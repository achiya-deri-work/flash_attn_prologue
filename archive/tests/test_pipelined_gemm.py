import torch
import torch.nn as nn
import time
import flash_attn_prologue

class PipelinedGemmModule(torch.nn.Module):
    def forward(self, x, weight):
        return torch.ops.flash_prologue.pipelined_gemm_cute(x, weight)

class PureGemmModule(torch.nn.Module):
    def forward(self, x, weight):
        return torch.ops.flash_prologue.pure_gemm_cute(x, weight)

def test_correctness():
    # Tile size config: BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64
    # B * N = 512 (multiple of 128)
    B = 4
    N = 128
    hidden = 512
    out_features = 256 # multiple of 128
    device = "cuda"
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    weight = torch.randn(out_features, hidden, device=device, dtype=dtype)
    
    print("\n[Parity Check] Pipelined CuTe GEMM vs PyTorch Eager GEMM")
    
    # Run pipelined gemm kernel
    qkv_pipelined = torch.ops.flash_prologue.pipelined_gemm_cute(x, weight)
    
    # Eager reference: x @ weight.t()
    qkv_eager = x @ weight.t()
    
    diff_tensor = (qkv_pipelined - qkv_eager).abs()
    diff = diff_tensor.max().item()
    print(f"Max absolute difference: {diff:.6f}")
    if diff > 1.5:
        mismatch_indices = (diff_tensor > 1.5).nonzero()
        print("First 5 mismatched indices (B, N, out_features):")
        for idx in mismatch_indices[:5]:
            b, n, f = idx.tolist()
            print(f"Index ({b}, {n}, {f}): pipelined={qkv_pipelined[b, n, f].item():.6f}, eager={qkv_eager[b, n, f].item():.6f}")
    print("Sample eager:", qkv_eager[0, 0, :32])
    print("Sample pipelined:", qkv_pipelined[0, 0, :32])
    
    # Check parity with precision-aware tolerances for bfloat16
    torch.testing.assert_close(
        qkv_pipelined, 
        qkv_eager, 
        atol=1.5, 
        rtol=1e-1, 
        msg="Pipelined CuTe GEMM vs Eager mismatch"
    )
    print("Pipelined CuTe GEMM Parity Passed!")

def run_benchmarks():
    # Benchmark shapes: larger to saturate Tensor Cores
    # total_m = B * N = 64 * 2048 = 131072 (multiple of 128)
    B = 64
    N = 2048
    hidden = 512
    out_features = 3072
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n--- Setting up Benchmarking (B={B}, N={N}, hidden={hidden}, out_features={out_features}) ---")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    weight = torch.randn(out_features, hidden, device=device, dtype=dtype)
    
    mod_pipelined = PipelinedGemmModule()
    mod_pure = PureGemmModule()
    
    # PyTorch Eager Module for Compiled Baseline
    class EagerModule(torch.nn.Module):
        def forward(self, x, weight):
            return x @ weight.t()
    mod_eager = EagerModule()
    
    # Compile all modules with inductor
    print("Compiling all variants with torch.compile...")
    comp_pipelined = torch.compile(mod_pipelined, backend="inductor")
    comp_pure = torch.compile(mod_pure, backend="inductor")
    comp_eager = torch.compile(mod_eager, backend="inductor")
    
    # Warm up compilation
    print("Triggering compilation runs...")
    y_pipelined_warm = comp_pipelined(x, weight)
    y_pure_warm = comp_pure(x, weight)
    y_eager_warm = comp_eager(x, weight)
    torch.cuda.synchronize()
    
    # Interleaved Warmup
    print("Running interleaved warmup...")
    for _ in range(30):
        _ = comp_pipelined(x, weight)
        _ = comp_pure(x, weight)
        _ = comp_eager(x, weight)
    torch.cuda.synchronize()
    
    # Interleaved Measurements
    print("Running interleaved measurements (50 iterations)...")
    num_iters = 50
    times_pipelined = []
    times_pure = []
    times_eager = []
    
    for i in range(num_iters):
        # 1. Pipelined CuTe GEMM
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_pipelined(x, weight)
        end.record()
        torch.cuda.synchronize()
        times_pipelined.append(start.elapsed_time(end))
        
        # 2. Pure CuTe GEMM (Non-Pipelined)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_pure(x, weight)
        end.record()
        torch.cuda.synchronize()
        times_pure.append(start.elapsed_time(end))
        
        # 3. PyTorch Inductor Compiled Eager
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_eager(x, weight)
        end.record()
        torch.cuda.synchronize()
        times_eager.append(start.elapsed_time(end))
        
    def get_stats(t_list):
        t_tensor = torch.tensor(t_list, dtype=torch.float32)
        return t_tensor.mean().item(), t_tensor.median().item(), t_tensor.min().item()
        
    mean_pipe, med_pipe, min_pipe = get_stats(times_pipelined)
    mean_pure, med_pure, min_pure = get_stats(times_pure)
    mean_eager, med_eager, min_eager = get_stats(times_eager)
    
    print("\n================== BENCHMARK LATENCY BREAKDOWN ==================")
    print(f"[{'Configuration / Component':<38} | {'Mean (ms)':<10} | {'Median (ms)':<10} | {'Min (ms)':<10}]")
    print("-" * 79)
    print(f"{'1. Pipelined CuTe GEMM (Blackwell)':<38} | {mean_pipe:<10.4f} | {med_pipe:<10.4f} | {min_pipe:<10.4f}")
    print(f"{'2. Pure CuTe GEMM (Non-Pipelined)':<38} | {mean_pure:<10.4f} | {med_pure:<10.4f} | {min_pure:<10.4f}")
    print(f"{'3. PyTorch Inductor (Compiled Eager)':<38} | {mean_eager:<10.4f} | {med_eager:<10.4f} | {min_eager:<10.4f}")
    print("==========================================================================")
    
    # Analysis outputs
    print("\n--- Performance Analysis ---")
    if mean_pipe < mean_pure:
        speedup_pure = mean_pure / mean_pipe
        print(f"Pipelined CuTe speedup vs Non-Pipelined CuTe: {speedup_pure:.2f}x ({(1 - mean_pipe/mean_pure)*100:.1f}% reduction)")
    else:
        print("Pipelined CuTe is not faster than Non-Pipelined CuTe.")
        
    if mean_pipe < mean_eager:
        speedup_eager = mean_eager / mean_pipe
        print(f"Pipelined CuTe speedup vs PyTorch Inductor: {speedup_eager:.2f}x")
    else:
        slowdown = mean_pipe / mean_eager
        print(f"Pipelined CuTe is {slowdown:.2f}x slower than PyTorch Inductor (cuBLAS).")

if __name__ == "__main__":
    test_correctness()
    run_benchmarks()
