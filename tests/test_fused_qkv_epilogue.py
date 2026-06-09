import torch
import flash_attn_prologue
import time

def test_fused_qkv_epilogue_fwd_bwd_parity_and_benchmark():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n=======================================================")
    print(f"Setting up Fused QKV + Epilogue Combined Test (B={B}, N={N})")
    print(f"=======================================================")
    
    # 1. Inputs
    x = torch.randn(B, N, hidden, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=dtype, requires_grad=True)
    rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32, requires_grad=True)
    
    # Precompute sin/cos frequencies
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = torch.cat((sin_val, sin_val), dim=-1).unsqueeze(1).contiguous()
    cos = torch.cat((cos_val, cos_val), dim=-1).unsqueeze(1).contiguous()
    
    # 2. Sequential Baseline (Eager/Torch.compile custom op chain)
    x_seq = x.clone().detach().requires_grad_(True)
    weight_seq = weight.clone().detach().requires_grad_(True)
    rms_seq = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_seq = scales.clone().detach().requires_grad_(True)
    
    def run_sequential(x_in, w_in, rms_in, s_in, sin_in, cos_in):
        q_proj, k_proj, v_proj = torch.ops.flash_prologue.fused_qkv(x_in, w_in)
        # Note: qknorm_rope_scale_cute clones internally, so this behaves correctly in eager
        q_final, k_final = torch.ops.flash_prologue.qknorm_rope_scale_cute(q_proj, k_proj, rms_in, s_in, sin_in, cos_in)
        return q_final, k_final, v_proj

    # 3. Combined/Fused Fwd/Bwd
    x_comb = x.clone().detach().requires_grad_(True)
    weight_comb = weight.clone().detach().requires_grad_(True)
    rms_comb = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_comb = scales.clone().detach().requires_grad_(True)
    
    def run_combined(x_in, w_in, rms_in, s_in, sin_in, cos_in):
        q, k, v, _, _ = torch.ops.flash_prologue.fused_qkv_epilogue(x_in, w_in, rms_in, s_in, sin_in, cos_in)
        return q, k, v
        
    # Compile both
    print("\n--- Compiling Kernels ---")
    compiled_seq = torch.compile(run_sequential, backend="inductor")
    compiled_comb = torch.compile(run_combined, backend="inductor")
    
    # Warmup Sequential
    q_s, k_s, v_s = compiled_seq(x_seq, weight_seq, rms_seq, scales_seq, sin, cos)
    (q_s.sum() + k_s.sum() + v_s.sum()).backward()
    
    # Warmup Combined
    q_c, k_c, v_c = compiled_comb(x_comb, weight_comb, rms_comb, scales_comb, sin, cos)
    (q_c.sum() + k_c.sum() + v_c.sum()).backward()
    
    torch.cuda.synchronize()
    print("Compilation and warmup successful!")
    
    # Parity check Forward outputs
    torch.testing.assert_close(q_c, q_s, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_c, k_s, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(v_c, v_s, rtol=1e-2, atol=1e-1)
    
    # Parity check Backward gradients
    torch.testing.assert_close(x_comb.grad, x_seq.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(weight_comb.grad, weight_seq.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(rms_comb.grad, rms_seq.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(scales_comb.grad, scales_seq.grad, rtol=1e-2, atol=6.0)
    
    print("All Combined Fused QKV + Epilogue Forward and Backward Parity Assertions Passed!")
    
    # Benchmarking
    def run_benchmark_fwd_bwd(compiled_fn, label):
        x_bench = x.clone().detach().requires_grad_(True)
        w_bench = weight.clone().detach().requires_grad_(True)
        rms_bench = rmsnorm_weight.clone().detach().requires_grad_(True)
        s_bench = scales.clone().detach().requires_grad_(True)
        
        # Warmup
        for _ in range(10):
            q_out, k_out, v_out = compiled_fn(x_bench, w_bench, rms_bench, s_bench, sin, cos)
            loss = q_out.sum() + k_out.sum() + v_out.sum()
            loss.backward()
            x_bench.grad.zero_()
            w_bench.grad.zero_()
            rms_bench.grad.zero_()
            s_bench.grad.zero_()
            
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            q_out, k_out, v_out = compiled_fn(x_bench, w_bench, rms_bench, s_bench, sin, cos)
            loss = q_out.sum() + k_out.sum() + v_out.sum()
            loss.backward()
            x_bench.grad.zero_()
            w_bench.grad.zero_()
            rms_bench.grad.zero_()
            s_bench.grad.zero_()
        end_event.record()
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / 100.0
        mem_peak = torch.cuda.max_memory_allocated()
        
        print(f"\n[{label} Fwd+Bwd Benchmark Results]")
        print(f"Average Execution Time: {time_ms:.4f} ms")
        print(f"Peak VRAM Allocated: {mem_peak / (1024**2):.2f} MB")
        return time_ms, mem_peak
        
    seq_time, seq_mem = run_benchmark_fwd_bwd(compiled_seq, "Sequential Autotuned Operators")
    comb_time, comb_mem = run_benchmark_fwd_bwd(compiled_comb, "Combined Fused QKV + Epilogue Operator")
    
    print("\n[Comparison Summary]")
    print(f"Speedup Factor: {seq_time / comb_time:.2f}x")
    print(f"VRAM Saving: {(seq_mem - comb_mem) / (1024**2):.2f} MB ({(seq_mem - comb_mem)/seq_mem*100:.1f}%)")

if __name__ == '__main__':
    test_fused_qkv_epilogue_fwd_bwd_parity_and_benchmark()
