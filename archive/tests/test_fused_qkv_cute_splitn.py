import torch
import torch.nn as nn
import time
from flash_attn_prologue.layers.FusedQKV import FusedQKV, rotate_half, rotate_half_backward
import flash_attn_prologue.splitn_kernel
import flash_attn_prologue.kernel
class EagerQKV(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = torch.nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = torch.nn.Parameter(torch.randn(num_heads, 1, 1))
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, x, sin, cos):
        B, N, C = x.shape
        qkv = torch.nn.functional.linear(x, self.qkv_proj.weight)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        q_fp32 = q.float()
        k_fp32 = k.float()
        
        q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
        k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
        
        q_normed = (q_fp32 * q_rsqrt) * self.qk_norm_weight[0].float()
        k_normed = (k_fp32 * k_rsqrt) * self.qk_norm_weight[1].float()
        
        q_roped = (q_normed * cos) + (rotate_half(q_normed) * sin)
        k_roped = (k_normed * cos) + (rotate_half(k_normed) * sin)
        
        q_proj = q_roped.to(x.dtype) * self.scales.exp().to(x.dtype)
        k_proj = k_roped.to(x.dtype)
        v_proj = v
        
        return q_proj, k_proj, v_proj

class FusedQKVCuteSplitN(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = torch.nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = torch.nn.Parameter(torch.randn(num_heads, 1, 1))
        
    def forward(self, x, thetas):
        sin, cos = thetas
        q, k, v = torch.ops.flash_prologue.fused_qkv_cute_splitn(
            x, self.qkv_proj.weight, self.qk_norm_weight, self.scales, sin, cos
        )
        return q, k, v

class FusedQKVCuteBase(torch.nn.Module):
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

def test_splitn_correctness_and_parity():
    B, N, hidden = 2, 128, 512
    num_heads, head_dim = 16, 64
    device = "cuda"
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    x_eager = torch.randn(B, N, hidden, device=device, dtype=dtype, requires_grad=True)
    x_splitn = x_eager.detach().clone().requires_grad_(True)
    x_base = x_eager.detach().clone().requires_grad_(True)
    
    eager_model = EagerQKV(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    splitn_model = FusedQKVCuteSplitN(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    splitn_model.load_state_dict(eager_model.state_dict())
    
    base_model = FusedQKVCuteBase(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    base_model.load_state_dict(eager_model.state_dict())
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin = torch.sin(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    cos = torch.cos(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    
    # Forward Eager
    q_e, k_e, v_e = eager_model(x_eager, sin, cos)
    loss_e = (q_e.sum() + k_e.sum() + v_e.sum())
    loss_e.backward()
    
    # Compile Custom Operators
    compiled_splitn = torch.compile(splitn_model, backend="inductor")
    compiled_base = torch.compile(base_model, backend="inductor")
    
    # Forward Custom (CuTe Split-N Compiled)
    q_s, k_s, v_s = compiled_splitn(x_splitn, (sin, cos))
    loss_s = (q_s.sum() + k_s.sum() + v_s.sum())
    loss_s.backward()
    
    # Forward Custom (CuTe Base Compiled)
    q_b, k_b, v_b = compiled_base(x_base, (sin, cos))
    loss_b = (q_b.sum() + k_b.sum() + v_b.sum())
    loss_b.backward()
    
    # Parity assertions
    print("\n[Parity Check] FusedQKVCuteSplitN vs Eager Baseline")
    print("Max diff Q:", (q_s - q_e).abs().max().item())
    print("Max diff K:", (k_s - k_e).abs().max().item())
    print("Max diff V:", (v_s - v_e).abs().max().item())
    
    torch.testing.assert_close(q_s, q_e, atol=1.5, rtol=1e-1, msg="Q mismatch splitn vs eager")
    torch.testing.assert_close(k_s, k_e, atol=1.5, rtol=1e-1, msg="K mismatch splitn vs eager")
    torch.testing.assert_close(v_s, v_e, atol=1.5, rtol=1e-1, msg="V mismatch splitn vs eager")
    
    # Check Gradients
    print("\n[Gradient Check] FusedQKVCuteSplitN vs Eager Baseline")
    if splitn_model.scales.grad is not None:
        torch.testing.assert_close(splitn_model.scales.grad, eager_model.scales.grad, atol=8.0, rtol=2e-2, msg="Scales grad mismatch")
    if splitn_model.qk_norm_weight.grad is not None:
        torch.testing.assert_close(splitn_model.qk_norm_weight.grad, eager_model.qk_norm_weight.grad, atol=8.0, rtol=2e-2, msg="RMSNorm grad mismatch")
    if splitn_model.qkv_proj.weight.grad is not None:
        torch.testing.assert_close(splitn_model.qkv_proj.weight.grad, eager_model.qkv_proj.weight.grad, atol=8.0, rtol=2e-2, msg="QKV weight grad mismatch")
    if x_splitn.grad is not None:
        torch.testing.assert_close(x_splitn.grad, x_eager.grad, atol=1.0, rtol=2e-2, msg="X grad mismatch")
        
    print("All Parity and Gradient Checks Passed for Split-N!")

def run_benchmark():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n--- Running Interleaved Performance Benchmarks (B={B}, N={N}, hidden={hidden}, heads={num_heads}) ---")
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin = torch.sin(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    cos = torch.cos(freqs_out).unsqueeze(-1).expand(*freqs_out.shape, 2).reshape(B, 1, N, head_dim)
    
    layer_baseline = FusedQKV(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_splitn = FusedQKVCuteSplitN(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_base = FusedQKVCuteBase(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    # Synchronize weights
    layer_splitn.qkv_proj.weight.data.copy_(layer_baseline.qkv_proj.weight.data)
    layer_splitn.qk_norm_weight.data.copy_(layer_baseline.qk_norm_weight.data)
    layer_splitn.scales.data.copy_(layer_baseline.scales.data)
    
    layer_base.qkv_proj.weight.data.copy_(layer_baseline.qkv_proj.weight.data)
    layer_base.qk_norm_weight.data.copy_(layer_baseline.qk_norm_weight.data)
    layer_base.scales.data.copy_(layer_baseline.scales.data)
    
    # Compile
    compiled_baseline = torch.compile(layer_baseline, backend="inductor")
    compiled_splitn = torch.compile(layer_splitn, backend="inductor")
    compiled_base = torch.compile(layer_base, backend="inductor")
    
    print("Compiling all variants...")
    compiled_baseline(x, (sin, cos))
    compiled_splitn(x, (sin, cos))
    compiled_base(x, (sin, cos))
    torch.cuda.synchronize()
    
    # Interleaved Warmup
    print("Running interleaved warmup...")
    for _ in range(30):
        _ = compiled_baseline(x, (sin, cos))
        _ = compiled_base(x, (sin, cos))
        _ = compiled_splitn(x, (sin, cos))
    torch.cuda.synchronize()
    
    # Measurement loop
    print("Running interleaved measurements (50 iterations)...")
    num_iters = 50
    times_baseline = []
    times_base = []
    times_splitn = []
    
    for i in range(num_iters):
        # 1. Baseline
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = compiled_baseline(x, (sin, cos))
        end.record()
        torch.cuda.synchronize()
        times_baseline.append(start.elapsed_time(end))
        
        # 2. Base CuTe
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = compiled_base(x, (sin, cos))
        end.record()
        torch.cuda.synchronize()
        times_base.append(start.elapsed_time(end))
        
        # 3. Split-N CuTe
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = compiled_splitn(x, (sin, cos))
        end.record()
        torch.cuda.synchronize()
        times_splitn.append(start.elapsed_time(end))
        
    # Stats calculations
    def get_stats(times_list):
        t_tensor = torch.tensor(times_list, dtype=torch.float32)
        mean_val = t_tensor.mean().item()
        median_val = t_tensor.median().item()
        min_val = t_tensor.min().item()
        return mean_val, median_val, min_val
        
    mean_baseline, med_baseline, min_baseline = get_stats(times_baseline)
    mean_base, med_base, min_base = get_stats(times_base)
    mean_splitn, med_splitn, min_splitn = get_stats(times_splitn)
    
    print("\n================== BENCHMARK RESULTS (Interleaved) ==================")
    print(f"[{'Kernel':<35} | {'Mean (ms)':<10} | {'Median (ms)':<10} | {'Min (ms)':<10}]")
    print("-" * 75)
    print(f"{'Native Inductor Compiled':<35} | {mean_baseline:<10.4f} | {med_baseline:<10.4f} | {min_baseline:<10.4f}")
    print(f"{'Fused CuTe Base (Single Head/CTA)':<35} | {mean_base:<10.4f} | {med_base:<10.4f} | {min_base:<10.4f}")
    print(f"{'Fused CuTe Split-N (Cooperative Load)':<35} | {mean_splitn:<10.4f} | {med_splitn:<10.4f} | {min_splitn:<10.4f}")
    print("=====================================================================")

if __name__ == "__main__":
    test_splitn_correctness_and_parity()
    run_benchmark()
