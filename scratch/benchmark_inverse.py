import torch
import torch.nn as nn
from flash_attn_prologue.layers.RoPE import apply_rope
from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op, qknorm_rope_scale_cute_inverse_op
import gc

# We'll use a realistic training shape: B=32, H=16, N=512, D=64.
B = 32
H = 16
N = 512
D = 64
device = "cuda"
dtype = torch.bfloat16
eps = 1e-5

def setup_tensors():
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    rmsnorm_weight = nn.Parameter(torch.ones(2, 1, D, device=device, dtype=dtype))
    scales = nn.Parameter(torch.full((H, 1, 1), -2.0, device=device, dtype=dtype))
    
    # Precompute sin/cos frequencies
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, D, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / D))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    return q, k, rmsnorm_weight, scales, sin, cos

def benchmark_standard():
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Fwd
    q_out, k_out = qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
    fwd_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    total_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return fwd_mem, total_mem

def benchmark_inverse():
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Fwd
    q_out, k_out, rms_q, rms_k = qknorm_rope_scale_cute_inverse_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
    fwd_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    total_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return fwd_mem, total_mem

def benchmark_eager():
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Fwd
    q_normed = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[0, 0].float()
    k_normed = k.float() * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[1, 0].float()
    q_rotated = apply_rope(q_normed.to(dtype), sin, cos)
    k_rotated = apply_rope(k_normed.to(dtype), sin, cos)
    q_out = q_rotated * scales.float().exp().to(dtype)
    k_out = k_rotated
    fwd_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    total_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return fwd_mem, total_mem

# Latency benchmark helpers
def measure_latency_standard(num_iters=100):
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    # Warmup
    for _ in range(10):
        q_out, k_out = qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
        loss = q_out.sum() + k_out.sum()
        if q.grad is not None:
            q.grad.zero_()
            k.grad.zero_()
            rmsnorm_weight.grad.zero_()
            scales.grad.zero_()
        loss.backward()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        q_out, k_out = qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
        loss = q_out.sum() + k_out.sum()
        q.grad.zero_()
        k.grad.zero_()
        rmsnorm_weight.grad.zero_()
        scales.grad.zero_()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters

def measure_latency_inverse(num_iters=100):
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    # Warmup
    for _ in range(10):
        q_out, k_out, rms_q, rms_k = qknorm_rope_scale_cute_inverse_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
        loss = q_out.sum() + k_out.sum()
        if q.grad is not None:
            q.grad.zero_()
            k.grad.zero_()
            rmsnorm_weight.grad.zero_()
            scales.grad.zero_()
        loss.backward()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        q_out, k_out, rms_q, rms_k = qknorm_rope_scale_cute_inverse_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
        loss = q_out.sum() + k_out.sum()
        q.grad.zero_()
        k.grad.zero_()
        rmsnorm_weight.grad.zero_()
        scales.grad.zero_()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters

def measure_latency_eager(num_iters=100):
    q, k, rmsnorm_weight, scales, sin, cos = setup_tensors()
    # Warmup
    for _ in range(10):
        q_normed = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[0, 0].float()
        k_normed = k.float() * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[1, 0].float()
        q_rotated = apply_rope(q_normed.to(dtype), sin, cos)
        k_rotated = apply_rope(k_normed.to(dtype), sin, cos)
        q_out = q_rotated * scales.float().exp().to(dtype)
        k_out = k_rotated
        loss = q_out.sum() + k_out.sum()
        if q.grad is not None:
            q.grad.zero_()
            k.grad.zero_()
            rmsnorm_weight.grad.zero_()
            scales.grad.zero_()
        loss.backward()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        q_normed = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[0, 0].float()
        k_normed = k.float() * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight[1, 0].float()
        q_rotated = apply_rope(q_normed.to(dtype), sin, cos)
        k_rotated = apply_rope(k_normed.to(dtype), sin, cos)
        q_out = q_rotated * scales.float().exp().to(dtype)
        k_out = k_rotated
        loss = q_out.sum() + k_out.sum()
        q.grad.zero_()
        k.grad.zero_()
        rmsnorm_weight.grad.zero_()
        scales.grad.zero_()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters

if __name__ == "__main__":
    print(f"=== Running Memory & Latency Benchmarks (B={B}, H={H}, N={N}, D={D}) ===")
    
    eager_fwd_mem, eager_total_mem = benchmark_eager()
    std_fwd_mem, std_total_mem = benchmark_standard()
    inv_fwd_mem, inv_total_mem = benchmark_inverse()
    
    eager_latency = measure_latency_eager()
    std_latency = measure_latency_standard()
    inv_latency = measure_latency_inverse()
    
    print("\n--- Memory Usage (MB) ---")
    print(f"Eager:    Peak Fwd: {eager_fwd_mem:.2f} MB, Peak Total: {eager_total_mem:.2f} MB")
    print(f"Standard: Peak Fwd: {std_fwd_mem:.2f} MB, Peak Total: {std_total_mem:.2f} MB")
    print(f"Inverse:  Peak Fwd: {inv_fwd_mem:.2f} MB, Peak Total: {inv_total_mem:.2f} MB")
    
    print("\n--- Latency (ms) ---")
    print(f"Eager:    {eager_latency:.3f} ms")
    print(f"Standard: {std_latency:.3f} ms")
    print(f"Inverse:  {inv_latency:.3f} ms")
