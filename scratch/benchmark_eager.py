import torch
import time
from flash_attn_prologue.epilogue_kernel import qknorm_rope_scale_eager

def benchmark_raw_eager():
    B = 32
    N = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Benchmarking RAW Eager Epilogue (B={B}, N={N}, num_heads={num_heads})")

    qkv = torch.randn(B, N, 3 * num_heads * head_dim, device=device, dtype=dtype)
    rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)

    # Generate frequencies of shape (B, N, head_dim // 2)
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)

    sin = torch.sin(freqs_out).unsqueeze(1).contiguous()
    cos = torch.cos(freqs_out).unsqueeze(1).contiguous()

    # Warmup
    qkv_bench = qkv.clone().detach().requires_grad_(True)
    rms_bench = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_bench = scales.clone().detach().requires_grad_(True)

    for _ in range(20):
        q_out, k_out, v_out = qknorm_rope_scale_eager(qkv_bench, rms_bench, s_bench, sin, cos)
        loss = q_out.sum() + k_out.sum()
        loss.backward()
        qkv_bench.grad.zero_()
        rms_bench.grad.zero_()
        s_bench.grad.zero_()

    torch.cuda.synchronize()

    # Benchmark iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        q_out, k_out, v_out = qknorm_rope_scale_eager(qkv_bench, rms_bench, s_bench, sin, cos)
        loss = q_out.sum() + k_out.sum()
        loss.backward()
        qkv_bench.grad.zero_()
        rms_bench.grad.zero_()
        s_bench.grad.zero_()
    end_event.record()
    torch.cuda.synchronize()

    raw_eager_time = start_event.elapsed_time(end_event) / 100.0
    print(f"Average Execution Time for RAW Eager: {raw_eager_time:.4f} ms")

if __name__ == "__main__":
    benchmark_raw_eager()
