import torch
import torch.nn as nn
import flash_attn_prologue.fused_qkv
import time

class TorchFusedQKVModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = True
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        
        q = qkv[..., 0, :, :].permute(0, 2, 1, 3).contiguous()
        k = qkv[..., 1, :, :].permute(0, 2, 1, 3).contiguous()
        v = qkv[..., 2, :, :].permute(0, 2, 1, 3).contiguous()
        return q, k, v

class CuteFusedQKVModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = True
        
    def forward(self, x):
        q, k, v = torch.ops.flash_prologue.fused_qkv(
            x, self.qkv_proj.weight
        )
        return q, k, v

def test_cute_fused_qkv_fwd_bwd_parity_and_benchmark():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n=======================================================")
    print(f"Setting up Fused QKV Fwd+Bwd Benchmarking (B={B}, N={N}, hidden={hidden}, num_heads={num_heads})")
    print(f"=======================================================")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    
    # Instantiate models
    torch_model = TorchFusedQKVModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    cute_model = CuteFusedQKVModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    # Synchronize weights
    cute_model.qkv_proj.weight.data.copy_(torch_model.qkv_proj.weight.data)
    
    # 1. Warmup and compile model forwards
    print("\n--- Compiling Models (Inductor) ---")
    compiled_torch = torch.compile(torch_model, backend="inductor")
    compiled_cute = torch.compile(cute_model, backend="inductor")
    
    # Prepare tensors with gradients
    x_t = x.clone().detach().requires_grad_(True)
    x_c = x.clone().detach().requires_grad_(True)
    
    # Warmup Torch Compiled (Inductor)
    q_t, k_t, v_t = compiled_torch(x_t)
    loss_t = q_t.sum() + k_t.sum() + v_t.sum()
    loss_t.backward()
    
    # Warmup CuTe Fused
    q_c, k_c, v_c = compiled_cute(x_c)
    loss_c = q_c.sum() + k_c.sum() + v_c.sum()
    loss_c.backward()
    
    torch.cuda.synchronize()
    print("Compilation and warmup successful!")
    
    # 2. Correctness Verification
    def print_accuracy_metrics(c_tensor, e_tensor, label):
        diff = (c_tensor.float() - e_tensor.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"--- Numerical Accuracy (CuTe vs Inductor) for {label} ---")
        print(f"Max Absolute Difference: {max_abs:.6f}")
        print(f"Mean Absolute Difference: {mean_abs:.6f}")
        
    print("\n[Accuracy Metrics - Forward]")
    print_accuracy_metrics(q_c, q_t, "Q Proj Output")
    print_accuracy_metrics(k_c, k_t, "K Proj Output")
    print_accuracy_metrics(v_c, v_t, "V Proj Output")
    
    print("\n[Accuracy Metrics - Backward]")
    print_accuracy_metrics(x_c.grad, x_t.grad, "Input Gradient (grad_x)")
    print_accuracy_metrics(cute_model.qkv_proj.weight.grad, torch_model.qkv_proj.weight.grad, "Weight Gradient (grad_weight)")
    
    # Assert close with precision-aware tolerances
    torch.testing.assert_close(q_c, q_t, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_c, k_t, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(v_c, v_t, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(x_c.grad, x_t.grad, rtol=2e-2, atol=1.0)
    torch.testing.assert_close(cute_model.qkv_proj.weight.grad, torch_model.qkv_proj.weight.grad, rtol=2e-2, atol=4.5)
    print("\nAll Forward and Backward Correctness Checks Passed!")
    
    # Reset gradients
    x_t.grad.zero_()
    x_c.grad.zero_()
    torch_model.qkv_proj.weight.grad.zero_()
    cute_model.qkv_proj.weight.grad.zero_()
    
    # 3. Benchmarking
    # Measure Forward-only
    def benchmark_fwd(compiled_fn, x_tensor, label):
        for _ in range(20):
            _ = compiled_fn(x_tensor)
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        mem_base = torch.cuda.memory_allocated()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            _ = compiled_fn(x_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_ms = start_event.elapsed_time(end_event) / 100.0
        peak_vram = (torch.cuda.max_memory_allocated() - mem_base) / (1024**2)
        print(f"[{label}] Forward Avg Time: {avg_ms:.4f} ms | Peak VRAM: {peak_vram:.2f} MB")
        return avg_ms, peak_vram

    # Measure Forward + Backward
    def benchmark_fwd_bwd(compiled_fn, x_tensor, weight_tensor, label):
        # Warmup
        for _ in range(20):
            q, k, v = compiled_fn(x_tensor)
            loss = q.sum() + k.sum() + v.sum()
            loss.backward()
            x_tensor.grad.zero_()
            weight_tensor.grad.zero_()
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        mem_base = torch.cuda.memory_allocated()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            q, k, v = compiled_fn(x_tensor)
            loss = q.sum() + k.sum() + v.sum()
            loss.backward()
            x_tensor.grad.zero_()
            weight_tensor.grad.zero_()
        end_event.record()
        torch.cuda.synchronize()
        
        avg_ms = start_event.elapsed_time(end_event) / 100.0
        peak_vram = (torch.cuda.max_memory_allocated() - mem_base) / (1024**2)
        print(f"[{label}] Fwd+Bwd Avg Time: {avg_ms:.4f} ms | Peak VRAM: {peak_vram:.2f} MB")
        return avg_ms, peak_vram

    print("\n--- Running Latency Benchmarks (100 iterations) ---")
    fwd_torch, fwd_torch_mem = benchmark_fwd(compiled_torch, x_t, "Torch Compiled (Inductor)")
    fwd_cute, fwd_cute_mem = benchmark_fwd(compiled_cute, x_c, "CuTe Fused QKV")
    
    fwd_bwd_torch, fwd_bwd_torch_mem = benchmark_fwd_bwd(compiled_torch, x_t, torch_model.qkv_proj.weight, "Torch Compiled (Inductor)")
    fwd_bwd_cute, fwd_bwd_cute_mem = benchmark_fwd_bwd(compiled_cute, x_c, cute_model.qkv_proj.weight, "CuTe Fused QKV")
    
    # Calculate backward and overhead by subtraction
    bwd_torch = fwd_bwd_torch - fwd_torch
    bwd_cute = fwd_bwd_cute - fwd_cute
    
    print("\n=======================================================")
    print("[Benchmark Breakdown Summary]")
    print("=======================================================")
    print(f"Forward Latency:      Torch = {fwd_torch:.4f} ms | CuTe = {fwd_cute:.4f} ms  --> Speedup: {fwd_torch / fwd_cute:.2f}x")
    print(f"Forward Peak VRAM:    Torch = {fwd_torch_mem:.2f} MB | CuTe = {fwd_cute_mem:.2f} MB")
    print(f"Backward+Overhead:    Torch = {bwd_torch:.4f} ms | CuTe = {bwd_cute:.4f} ms  --> Speedup: {bwd_torch / bwd_cute:.2f}x")
    print(f"Total Fwd+Bwd Latency: Torch = {fwd_bwd_torch:.4f} ms | CuTe = {fwd_bwd_cute:.4f} ms  --> Speedup: {fwd_bwd_torch / fwd_bwd_cute:.2f}x")
    print(f"Total Fwd+Bwd VRAM:   Torch = {fwd_bwd_torch_mem:.2f} MB | CuTe = {fwd_bwd_cute_mem:.2f} MB")
    print("=======================================================")

if __name__ == '__main__':
    test_cute_fused_qkv_fwd_bwd_parity_and_benchmark()
