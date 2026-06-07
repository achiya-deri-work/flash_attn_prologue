import torch
import torch.nn as nn
import time
from flash_attn_prologue.layers.FusedQKV import FusedQKV
import flash_attn_prologue

class PureGemmModule(torch.nn.Module):
    def forward(self, x, weight):
        return torch.ops.flash_prologue.pure_gemm_cute(x, weight)

class TritonEpilogueModule(torch.nn.Module):
    def forward(self, qkv, rmsnorm_weight, scales, sin, cos):
        return torch.ops.flash_prologue.qknorm_rope_scale(qkv, rmsnorm_weight, scales, sin, cos)

class SequentialModule(torch.nn.Module):
    def forward(self, x, weight, rmsnorm_weight, scales, sin, cos):
        qkv = torch.ops.flash_prologue.pure_gemm_cute(x, weight)
        qkv_flat = qkv.view(x.shape[0], x.shape[1], -1)
        return torch.ops.flash_prologue.qknorm_rope_scale(qkv_flat, rmsnorm_weight, scales, sin, cos)

class FusedCuteBaseModule(torch.nn.Module):
    def forward(self, x, weight, rmsnorm_weight, scales, sin, cos):
        return torch.ops.flash_prologue.fused_qkv_cute(x, weight, rmsnorm_weight, scales, sin, cos)

class FusedCuteSplitNModule(torch.nn.Module):
    def forward(self, x, weight, rmsnorm_weight, scales, sin, cos):
        return torch.ops.flash_prologue.fused_qkv_cute_splitn(x, weight, rmsnorm_weight, scales, sin, cos)

def test_correctness():
    B = 4
    N = 128
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    weight = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=dtype)
    rmsnorm_weight = torch.ones(2, 1, head_dim, device=device, dtype=dtype)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=dtype)
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin = torch.sin(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    cos = torch.cos(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    
    print("\n[Parity Check] Pure CuTe GEMM vs PyTorch Eager GEMM")
    qkv_cute = torch.ops.flash_prologue.pure_gemm_cute(x, weight)
    qkv_eager = (x @ weight.t()).view(B, N, 3, num_heads, head_dim)
    
    diff = (qkv_cute - qkv_eager).abs().max().item()
    print(f"Max absolute difference: {diff:.6f}")
    torch.testing.assert_close(qkv_cute, qkv_eager, atol=1.5, rtol=1e-1, msg="CuTe GEMM vs Eager mismatch")
    print("Pure CuTe GEMM Parity Passed!")
    
    print("\n[Parity Check] Triton Epilogue vs PyTorch Eager Epilogue")
    qkv_flat = qkv_eager.view(B, N, -1)
    q_triton, k_triton, v_triton = torch.ops.flash_prologue.qknorm_rope_scale(qkv_flat, rmsnorm_weight, scales, sin, cos)
    
    # Eager FusedQKV as reference
    eager_layer = FusedQKV(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    eager_layer.qkv_proj.weight.data.copy_(weight)
    eager_layer.qk_norm_weight.data.copy_(rmsnorm_weight)
    eager_layer.scales.data.copy_(scales)
    
    q_eager, k_eager, v_eager = eager_layer(x, (sin, cos))
    
    torch.testing.assert_close(q_triton, q_eager, atol=1.5, rtol=1e-1, msg="Triton Epilogue vs Eager Q mismatch")
    torch.testing.assert_close(k_triton, k_eager, atol=1.5, rtol=1e-1, msg="Triton Epilogue vs Eager K mismatch")
    torch.testing.assert_close(v_triton, v_eager, atol=1.5, rtol=1e-1, msg="Triton Epilogue vs Eager V mismatch")
    print("Triton Epilogue Parity Passed!")
    
    print("\n[Parity Check] Sequential Module vs Eager Baseline")
    seq_mod = SequentialModule()
    q_seq, k_seq, v_seq = seq_mod(x, weight, rmsnorm_weight, scales, sin, cos)
    torch.testing.assert_close(q_seq, q_eager, atol=1.5, rtol=1e-1)
    torch.testing.assert_close(k_seq, k_eager, atol=1.5, rtol=1e-1)
    torch.testing.assert_close(v_seq, v_eager, atol=1.5, rtol=1e-1)
    print("Sequential Module Parity Passed!")

def run_benchmarks():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n--- Setting up Benchmarking (B={B}, N={N}, hidden={hidden}, heads={num_heads}) ---")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    weight = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=dtype)
    rmsnorm_weight = torch.ones(2, 1, head_dim, device=device, dtype=dtype)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=dtype)
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin = torch.sin(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    cos = torch.cos(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    
    # Pre-computed GEMM output for pure Triton Epilogue benchmarking
    qkv_gemm_out = torch.empty((B, N, 3 * num_heads * head_dim), dtype=dtype, device=device)
    
    # Instantiate modules
    mod_gemm = PureGemmModule()
    mod_epi = TritonEpilogueModule()
    mod_seq = SequentialModule()
    mod_fused_base = FusedCuteBaseModule()
    mod_fused_splitn = FusedCuteSplitNModule()
    
    layer_baseline = FusedQKV(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_baseline.qkv_proj.weight.data.copy_(weight)
    layer_baseline.qk_norm_weight.data.copy_(rmsnorm_weight)
    layer_baseline.scales.data.copy_(scales)
    
    # Compile all modules with inductor
    print("Compiling all variants with torch.compile...")
    comp_gemm = torch.compile(mod_gemm, backend="inductor")
    comp_epi = torch.compile(mod_epi, backend="inductor")
    comp_seq = torch.compile(mod_seq, backend="inductor")
    comp_fused_base = torch.compile(mod_fused_base, backend="inductor")
    comp_fused_splitn = torch.compile(mod_fused_splitn, backend="inductor")
    comp_baseline = torch.compile(layer_baseline, backend="inductor")
    
    # Warm up compilation
    print("Triggering compilation runs...")
    comp_gemm(x, weight)
    comp_epi(qkv_gemm_out, rmsnorm_weight, scales, sin, cos)
    comp_seq(x, weight, rmsnorm_weight, scales, sin, cos)
    comp_fused_base(x, weight, rmsnorm_weight, scales, sin, cos)
    comp_fused_splitn(x, weight, rmsnorm_weight, scales, sin, cos)
    comp_baseline(x, (sin, cos))
    torch.cuda.synchronize()
    
    # Interleaved Warmup
    print("Running interleaved warmup...")
    for _ in range(30):
        _ = comp_gemm(x, weight)
        _ = comp_epi(qkv_gemm_out, rmsnorm_weight, scales, sin, cos)
        _ = comp_seq(x, weight, rmsnorm_weight, scales, sin, cos)
        _ = comp_fused_base(x, weight, rmsnorm_weight, scales, sin, cos)
        _ = comp_fused_splitn(x, weight, rmsnorm_weight, scales, sin, cos)
        _ = comp_baseline(x, (sin, cos))
    torch.cuda.synchronize()
    
    # Interleaved Measurements
    print("Running interleaved measurements (50 iterations)...")
    num_iters = 50
    times_gemm = []
    times_epi = []
    times_seq = []
    times_fused_base = []
    times_fused_splitn = []
    times_baseline = []
    
    for i in range(num_iters):
        # 1. Pure CuTe GEMM
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_gemm(x, weight)
        end.record()
        torch.cuda.synchronize()
        times_gemm.append(start.elapsed_time(end))
        
        # 2. Triton Epilogue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_epi(qkv_gemm_out, rmsnorm_weight, scales, sin, cos)
        end.record()
        torch.cuda.synchronize()
        times_epi.append(start.elapsed_time(end))
        
        # 3. Sequential Combination
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_seq(x, weight, rmsnorm_weight, scales, sin, cos)
        end.record()
        torch.cuda.synchronize()
        times_seq.append(start.elapsed_time(end))
        
        # 4. Fused CuTe Base
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_fused_base(x, weight, rmsnorm_weight, scales, sin, cos)
        end.record()
        torch.cuda.synchronize()
        times_fused_base.append(start.elapsed_time(end))
        
        # 5. Fused CuTe Split-N
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_fused_splitn(x, weight, rmsnorm_weight, scales, sin, cos)
        end.record()
        torch.cuda.synchronize()
        times_fused_splitn.append(start.elapsed_time(end))
        
        # 6. PyTorch Inductor Compiled
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = comp_baseline(x, (sin, cos))
        end.record()
        torch.cuda.synchronize()
        times_baseline.append(start.elapsed_time(end))
        
    def get_stats(t_list):
        t_tensor = torch.tensor(t_list, dtype=torch.float32)
        return t_tensor.mean().item(), t_tensor.median().item(), t_tensor.min().item()
        
    mean_gemm, med_gemm, min_gemm = get_stats(times_gemm)
    mean_epi, med_epi, min_epi = get_stats(times_epi)
    mean_seq, med_seq, min_seq = get_stats(times_seq)
    mean_fb, med_fb, min_fb = get_stats(times_fused_base)
    mean_fs, med_fs, min_fs = get_stats(times_fused_splitn)
    mean_base, med_base, min_base = get_stats(times_baseline)
    
    print("\n================== DETAILED COMPONENT LATENCY BREAKDOWN ==================")
    print(f"[{'Configuration / Component':<38} | {'Mean (ms)':<10} | {'Median (ms)':<10} | {'Min (ms)':<10}]")
    print("-" * 79)
    print(f"{'1. Pure CuTe GEMM':<38} | {mean_gemm:<10.4f} | {med_gemm:<10.4f} | {min_gemm:<10.4f}")
    print(f"{'2. Triton Epilogue':<38} | {mean_epi:<10.4f} | {med_epi:<10.4f} | {min_epi:<10.4f}")
    print(f"{'3. Sequential (CuTe GEMM + Triton)':<38} | {mean_seq:<10.4f} | {med_seq:<10.4f} | {min_seq:<10.4f}")
    print(f"{'4. Fused CuTe Base':<38} | {mean_fb:<10.4f} | {med_fb:<10.4f} | {min_fb:<10.4f}")
    print(f"{'5. Fused CuTe Split-N':<38} | {mean_fs:<10.4f} | {med_fs:<10.4f} | {min_fs:<10.4f}")
    print(f"{'6. PyTorch Inductor (Compiled Eager)':<38} | {mean_base:<10.4f} | {med_base:<10.4f} | {min_base:<10.4f}")
    print("==========================================================================")
    
    # Analysis outputs
    print("\n--- Component Interaction Analysis ---")
    print(f"Sum of parts (GEMM + Epilogue): {mean_gemm + mean_epi:.4f} ms")
    print(f"Measured Sequential wrapper: {mean_seq:.4f} ms (overhead vs sum: {mean_seq - (mean_gemm + mean_epi):.4f} ms)")
    print(f"Base Fusion savings: {(mean_seq - mean_fb):.4f} ms ({(1 - mean_fb/mean_seq)*100:.1f}% reduction)")
    print(f"Split-N Fusion savings: {(mean_seq - mean_fs):.4f} ms ({(1 - mean_fs/mean_seq)*100:.1f}% reduction)")
    print(f"Split-N speedup vs PyTorch Inductor: {mean_base / mean_fs:.2f}x")

if __name__ == "__main__":
    test_correctness()
    run_benchmarks()
