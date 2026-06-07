import torch
import flash_attn_prologue.kernel
from flash_attn_prologue.layers.FusedQKV import FusedQKV
import time

class FusedQKVCute(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = torch.nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = torch.nn.Parameter(torch.randn(num_heads, 1, 1))
        
    def forward(self, x, thetas):
        sin, cos = thetas
        q, k, v = torch.ops.flash_prologue.fused_qkv_cute(
            x, self.qkv_proj.weight, self.qk_norm_weight, self.scales, sin, cos
        )
        return q, k, v

def test_cute_forward_parity_and_benchmark():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"Setting up benchmarking (B={B}, N={N}, hidden={hidden}, num_heads={num_heads})")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    
    layer_baseline = FusedQKV(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_cute = FusedQKVCute(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    
    sin = sin_val.unsqueeze(-1).expand(*sin_val.shape, 2).reshape(B, 1, N, head_dim)
    cos = cos_val.unsqueeze(-1).expand(*cos_val.shape, 2).reshape(B, 1, N, head_dim)
    
    print("\n--- Testing Native Torch (Option A) Eager Mode (No Compile) ---")
    # Warmup Eager
    for _ in range(5):
        _ = layer_baseline(x, (sin, cos))
    torch.cuda.synchronize()
    
    # Benchmark Eager
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
    
    # Measure Compile Time
    print("Triggering JIT compilation...")
    t0 = time.time()
    _ = compiled_baseline(x, (sin, cos))
    torch.cuda.synchronize()
    t1 = time.time()
    baseline_compile_time = t1 - t0
    print(f"Native Torch Compile Time: {baseline_compile_time:.2f} seconds")
    
    # Sync weights
    layer_cute.qkv_proj.weight.data.copy_(layer_baseline.qkv_proj.weight.data)
    layer_cute.qk_norm_weight.data.copy_(layer_baseline.qk_norm_weight.data)
    layer_cute.scales.data.copy_(layer_baseline.scales.data)

    print("\n--- Testing CuTe DSL kernel (Option C) ---")
    compiled_cute = torch.compile(layer_cute, backend="inductor")
    
    # Measure Compile Time
    print("Triggering JIT compilation...")
    t0 = time.time()
    _ = compiled_cute(x, (sin, cos))
    torch.cuda.synchronize()
    t1 = time.time()
    cute_compile_time = t1 - t0
    print(f"CuTe DSL JIT Compile Time: {cute_compile_time:.2f} seconds")

    # Helper to measure memory and time
    def run_benchmark(compiled_fn, x, sin, cos, label):
        # Warmup
        for _ in range(20):
            _ = compiled_fn(x, (sin, cos))
        torch.cuda.synchronize()
        
        # Reset memory stats
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
        return time_ms, mem_peak, mem_used

    # Run benchmarks
    torch_time, torch_mem, _ = run_benchmark(compiled_baseline, x, sin, cos, "Native Torch Compiled")
    cute_time, cute_mem, _ = run_benchmark(compiled_cute, x, sin, cos, "CuTe DSL Kernel")
    
    print("\n[Comparison Summary]")
    print(f"Speedup Factor: {torch_time / cute_time:.2f}x")
    print(f"VRAM Reduction Factor: {torch_mem / cute_mem:.2f}x")

    # Measure detailed numerical accuracy
    q_e, k_e, v_e = compiled_baseline(x, (sin, cos))
    q_c, k_c, v_c = compiled_cute(x, (sin, cos))
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

    print_accuracy_metrics(q_c, q_e, "Q Proj Output")
    print_accuracy_metrics(k_c, k_e, "K Proj Output")
    print_accuracy_metrics(v_c, v_e, "V Proj Output")

if __name__ == '__main__':
    test_cute_forward_parity_and_benchmark()
