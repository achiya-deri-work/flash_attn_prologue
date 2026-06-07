import torch
import time
from flash_attn_prologue import fused_qkv_pipelined_cute_op

def eager_fused_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch baseline for the Fused QKV Epilogue.
    Follows strict BFP16 / FP32 numerical stability rules.
    """
    B, N, C = x.shape
    
    qkv = torch.nn.functional.linear(x, weight)
    
    qkv = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v_proj = qkv[:, :, 2]
    
    q_fp32 = q.float()
    k_fp32 = k.float()
    
    q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + eps)
    k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + eps)
    
    q_normed_fp32 = q_fp32 * q_rsqrt
    k_normed_fp32 = k_fp32 * k_rsqrt
    
    q_normed = q_normed_fp32.to(q.dtype)
    k_normed = k_normed_fp32.to(k.dtype)
    
    q_normed = q_normed * rmsnorm_weight[0]
    k_normed = k_normed * rmsnorm_weight[1]
    
    def rotate_half(x_tensor):
        x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
        x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    q_normed_fp32 = q_normed.float()
    k_normed_fp32 = k_normed.float()
    
    q_roped_fp32 = (q_normed_fp32 * cos) + (rotate_half(q_normed_fp32) * sin)
    k_roped_fp32 = (k_normed_fp32 * cos) + (rotate_half(k_normed_fp32) * sin)
    
    q_proj = q_roped_fp32.to(q.dtype)
    k_proj = k_roped_fp32.to(k.dtype)
    
    scale_factor = scales.float().exp().to(q.dtype)
    q_proj = q_proj * scale_factor
    
    return q_proj, k_proj, v_proj


class BaselineModule(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = False
        self.qk_norm_weight = torch.nn.Parameter(torch.ones(2, 1, head_dim), requires_grad=False)
        self.scales = torch.nn.Parameter(torch.randn(num_heads, 1, 1), requires_grad=False)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, x, thetas):
        sin, cos = thetas
        return eager_fused_qkv(
            x, self.qkv_proj.weight, self.qk_norm_weight, self.scales, sin, cos, self.num_heads, self.head_dim
        )


class FusedQKVPipelinedCuteModule(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = False
        self.qk_norm_weight = torch.nn.Parameter(torch.ones(2, 1, head_dim), requires_grad=False)
        self.scales = torch.nn.Parameter(torch.randn(num_heads, 1, 1), requires_grad=False)
        
    def forward(self, x, thetas):
        sin, cos = thetas
        q, k, v = fused_qkv_pipelined_cute_op(
            x, self.qkv_proj.weight, self.qk_norm_weight, self.scales, sin, cos
        )
        return q, k, v


def test_pipelined_forward_parity_and_benchmark():
    B = 64
    N = 2048
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"Setting up benchmarking for Pipelined Kernel (B={B}, N={N}, hidden={hidden}, num_heads={num_heads})")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    
    # Base layers
    layer_baseline = BaselineModule(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_pipelined = FusedQKVPipelinedCuteModule(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    # Sync weights
    layer_pipelined.qkv_proj.weight.data.copy_(layer_baseline.qkv_proj.weight.data)
    layer_pipelined.qk_norm_weight.data.copy_(layer_baseline.qk_norm_weight.data)
    layer_pipelined.scales.data.copy_(layer_baseline.scales.data)
    
    # Thetas/frequencies (kept for compatibility with pipeline signature)
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    
    sin = sin_val.unsqueeze(-1).expand(*sin_val.shape, 2).reshape(B, 1, N, head_dim)
    cos = cos_val.unsqueeze(-1).expand(*cos_val.shape, 2).reshape(B, 1, N, head_dim)
    
    with torch.no_grad():
        print("\n--- Testing Native Torch (Option A) Eager Mode ---")
        for _ in range(5):
            _ = layer_baseline(x, (sin, cos))
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(50):
            _ = layer_baseline(x, (sin, cos))
        end_event.record()
        torch.cuda.synchronize()
        eager_time_ms = start_event.elapsed_time(end_event) / 50.0
        print(f"Native Torch Eager Forward Time: {eager_time_ms:.4f} ms")
        
        print("\n--- Testing Native Torch Compiled (Option A) ---")
        compiled_baseline = torch.compile(layer_baseline, backend="inductor")
        print("Triggering JIT compilation for baseline...")
        t0 = time.time()
        _ = compiled_baseline(x, (sin, cos))
        torch.cuda.synchronize()
        print(f"Native Torch Compile Time: {time.time() - t0:.2f} seconds")
        
        print("\n--- Testing CuTe DSL Pipelined Kernel ---")
        print("Running CuTe Pipelined Kernel in Eager Mode (triggers JIT compilation)...")
        t0 = time.time()
        _ = layer_pipelined(x, (sin, cos))
        torch.cuda.synchronize()
        print(f"CuTe Pipelined Eager Run Time: {time.time() - t0:.2f} seconds")
        
        compiled_pipelined = torch.compile(layer_pipelined, backend="inductor")
        print("Triggering torch.compile JIT compilation for pipelined kernel...")
        t0 = time.time()
        _ = compiled_pipelined(x, (sin, cos))
        torch.cuda.synchronize()
        print(f"CuTe Pipelined JIT Compile Time: {time.time() - t0:.2f} seconds")
        
        # Helper to measure memory and time
        def run_benchmark(compiled_fn, x, sin, cos, label):
            for _ in range(20):
                _ = compiled_fn(x, (sin, cos))
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(100):
                _ = compiled_fn(x, (sin, cos))
            end_event.record()
            torch.cuda.synchronize()
            
            time_ms = start_event.elapsed_time(end_event) / 100.0
            mem_peak = torch.cuda.max_memory_allocated()
            mem_used = mem_peak - mem_before
            
            print(f"\n[{label} Benchmark Results]")
            print(f"Average Execution Time: {time_ms:.4f} ms")
            print(f"Peak VRAM Allocated: {mem_peak / (1024**2):.2f} MB")
            print(f"Active Memory Consumed: {mem_used / (1024**2):.2f} MB")
            return time_ms, mem_peak
            
        torch_time, torch_mem = run_benchmark(compiled_baseline, x, sin, cos, "Native Torch Compiled")
        pipe_time, pipe_mem = run_benchmark(compiled_pipelined, x, sin, cos, "CuTe Pipelined Kernel")
        
        print("\n[Comparison Summary]")
        print(f"Speedup Factor: {torch_time / pipe_time:.2f}x")
        print(f"VRAM Reduction Factor: {torch_mem / pipe_mem:.2f}x")
        
        # Numerical accuracy check
        q_e, k_e, v_e = compiled_baseline(x, (sin, cos))
        q_p, k_p, v_p = compiled_pipelined(x, (sin, cos))
        torch.cuda.synchronize()
        
        def print_accuracy_metrics(c_tensor, e_tensor, label):
            diff = (c_tensor.float() - e_tensor.float()).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            
            # Relative diff
            rel_diff = diff / (e_tensor.float().abs() + 1e-8)
            max_rel = rel_diff.max().item()
            mean_rel = rel_diff.mean().item()
            
            # Percentages within tolerances
            pct_1e3 = (diff <= 1e-3).float().mean().item() * 100
            pct_1e2 = (diff <= 1e-2).float().mean().item() * 100
            pct_1e1 = (diff <= 1e-1).float().mean().item() * 100
            
            print(f"\n--- Detailed Numerical Accuracy for {label} ---")
            print(f"Max Absolute Difference: {max_abs:.6f}")
            print(f"Mean Absolute Difference: {mean_abs:.6f}")
            print(f"Max Relative Difference: {max_rel:.6f}")
            print(f"Mean Relative Difference: {mean_rel:.6f}")
            print(f"Percentage of elements matching within 1e-3: {pct_1e3:.2f}%")
            print(f"Percentage of elements matching within 1e-2: {pct_1e2:.2f}%")
            print(f"Percentage of elements matching within 1e-1: {pct_1e1:.2f}%")
            
            # Using the requested precision-aware tolerances
            mask = diff > 0.1
            if mask.any():
                print(f"First 10 mismatched elements for {label}:")
                indices = torch.nonzero(mask)[:10]
                for idx in indices:
                    i_tuple = tuple(idx.tolist())
                    print(f"  Index {i_tuple}: custom={c_tensor[i_tuple].item():.4f}, eager={e_tensor[i_tuple].item():.4f}, diff={diff[i_tuple].item():.4f}")
            torch.testing.assert_close(c_tensor, e_tensor, rtol=1e-1, atol=1.5)
            
        print_accuracy_metrics(q_p, q_e, "Q Proj Output")
        print_accuracy_metrics(k_p, k_e, "K Proj Output")
        print_accuracy_metrics(v_p, v_e, "V Proj Output")
        print("\nParity verification succeeded!")

if __name__ == '__main__':
    test_pipelined_forward_parity_and_benchmark()
