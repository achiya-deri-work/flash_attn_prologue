import torch
import flash_attn_prologue.fused_qkv
import time

class PureGemmBaselineModel(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = False
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

class FusedQKVModel(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qkv_proj.weight.requires_grad = False
        
    def forward(self, x):
        q, k, v = torch.ops.flash_prologue.fused_qkv(
            x, self.qkv_proj.weight
        )
        return q, k, v

def test_cute_fused_qkv_forward_parity_and_benchmark():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"Setting up benchmarking (B={B}, N={N}, hidden={hidden}, num_heads={num_heads})")
    
    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    
    layer_baseline = PureGemmBaselineModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    layer_cute = FusedQKVModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    
    # Sync weights
    layer_cute.qkv_proj.weight.data.copy_(layer_baseline.qkv_proj.weight.data)

    with torch.no_grad():
        print("\n--- Testing Native Torch Compiled (Pure GEMM) ---")
        compiled_baseline = torch.compile(layer_baseline, backend="inductor")
        _ = compiled_baseline(x)
        torch.cuda.synchronize()
        
        print("\n--- Testing CuTe DSL Fused QKV Kernel ---")
        compiled_cute = torch.compile(layer_cute, backend="inductor")
        _ = compiled_cute(x)
        torch.cuda.synchronize()

    # Helper to measure memory and time
    def run_benchmark(compiled_fn, args, label):
        # Warmup
        for _ in range(20):
            _ = compiled_fn(*args)
        torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            _ = compiled_fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / 100.0
        mem_peak = torch.cuda.max_memory_allocated()
        
        print(f"\n[{label} Benchmark Results]")
        print(f"Average Execution Time: {time_ms:.4f} ms")
        print(f"Peak VRAM Allocated: {mem_peak / (1024**2):.2f} MB")
        return time_ms

    # Run benchmarks
    torch_time = run_benchmark(compiled_baseline, (x,), "Native Torch Compiled (Pure GEMM)")
    cute_time = run_benchmark(compiled_cute, (x,), "CuTe DSL Fused QKV Kernel")
    
    print("\n[Comparison Summary]")
    print(f"Speedup Factor: {torch_time / cute_time:.2f}x")

    # Measure detailed numerical accuracy
    q_e, k_e, v_e = compiled_baseline(x)
    q_c, k_c, v_c = compiled_cute(x)
    torch.cuda.synchronize()

    def print_accuracy_metrics(c_tensor, e_tensor, label):
        diff = (c_tensor.float() - e_tensor.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        
        # Relative diff
        rel_diff = diff / (e_tensor.float().abs() + 1e-8)
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()
        
        print(f"\n--- Detailed Numerical Accuracy for {label} ---")
        print(f"Max Absolute Difference: {max_abs:.6f}")
        print(f"Mean Absolute Difference: {mean_abs:.6f}")

    print_accuracy_metrics(q_c, q_e, "Q Proj Output")
    print_accuracy_metrics(k_c, k_e, "K Proj Output")
    print_accuracy_metrics(v_c, v_e, "V Proj Output")
    
    # Assert close with precision-aware tolerances for bfloat16
    torch.testing.assert_close(q_c, q_e, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_c, k_e, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(v_c, v_e, rtol=1e-2, atol=1e-1)

def qknorm_rope_scale_torch_comp(qkv, rmsnorm_weight, scales, freqs, eps=1e-5):
    B, N, total_dim = qkv.shape
    head_dim = rmsnorm_weight.shape[-1]
    num_heads = total_dim // 3 // head_dim
    
    qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim)
    qk = qkv_reshaped[..., 0:2, :, :].float()
    
    w = rmsnorm_weight.view(1, 1, 2, 1, head_dim)
    f = freqs.view(B, N, 1, 1, head_dim // 2)
    s = scales.float().exp().view(1, 1, num_heads, 1)
    
    qk_normed = qk * torch.rsqrt(torch.pow(qk, 2).mean(-1, keepdim=True) + eps) * w
    qk1, qk2 = qk_normed.chunk(2, dim=-1)
    
    out1 = qk1 * torch.cos(f) - qk2 * torch.sin(f)
    out2 = qk2 * torch.cos(f) + qk1 * torch.sin(f)
    
    out_fp32 = torch.cat((out1, out2), dim=-1)
    
    k_proj = out_fp32[:, :, 1, :, :].to(qkv.dtype).permute(0, 2, 1, 3).contiguous()
    q_proj = (out_fp32[:, :, 0, :, :] * s).to(qkv.dtype).permute(0, 2, 1, 3).contiguous()
    v_proj = qkv_reshaped[..., 2, :, :].permute(0, 2, 1, 3).contiguous()
    
    return q_proj, k_proj, v_proj

def test_cute_fused_epilogue_parity_and_benchmark():
    import flash_attn_prologue.fused_epilogue
    from flash_attn_prologue.epilogue_kernel import qknorm_rope_scale_triton, qknorm_rope_scale_eager
    
    B = 32
    N = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n=======================================================")
    print(f"Setting up Fused Epilogue Benchmarking (B={B}, N={N}, num_heads={num_heads})")
    print(f"=======================================================")
    
    qkv = torch.randn(B, N, 3 * num_heads * head_dim, device=device, dtype=dtype)
    rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)
    
    # Generate frequencies of shape (B, N, head_dim // 2)
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    # Precompute sin and cos for Triton/eager/CuTe baselines
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = torch.cat((sin_val, sin_val), dim=-1).unsqueeze(1).contiguous()
    cos = torch.cat((cos_val, cos_val), dim=-1).unsqueeze(1).contiguous()
    
    # Define functional wrappers for Triton, Eager, CuTe, and Torch Compiled (Inductor)
    def run_triton_fwd(qkv_in, rms_w, s, sin_t, cos_t):
        q_proj, k_proj, v_proj = torch.ops.flash_prologue.qknorm_rope_scale(qkv_in, rms_w, s, sin_t, cos_t)
        return q_proj, k_proj
        
    def run_eager_fwd(qkv_in, rms_w, s, sin_t, cos_t):
        q_proj, k_proj, v_proj = qknorm_rope_scale_eager(qkv_in, rms_w, s, sin_t, cos_t)
        return q_proj, k_proj
        
    def run_cute_fwd(qkv_in, rms_w, s, sin_t, cos_t):
        qkv_reshaped = qkv_in.view(B, N, 3, num_heads, head_dim)
        q_in = qkv_reshaped[..., 0, :, :].permute(0, 2, 1, 3).contiguous()
        k_in = qkv_reshaped[..., 1, :, :].permute(0, 2, 1, 3).contiguous()
        return torch.ops.flash_prologue.qknorm_rope_scale_cute(q_in, k_in, rms_w, s, sin_t, cos_t)

    def run_torch_comp_fwd(qkv_in, rms_w, s, sin_t, cos_t):
        q_proj, k_proj, v_proj = qknorm_rope_scale_torch_comp(qkv_in, rms_w, s, freqs_out)
        return q_proj, k_proj

    # Compile them
    print("\n--- Compiling Kernels with Autograd Support ---")
    compiled_triton = torch.compile(run_triton_fwd, backend="inductor")
    compiled_cute = torch.compile(run_cute_fwd, backend="inductor")
    compiled_eager = torch.compile(run_eager_fwd, backend="inductor")
    compiled_torch_comp = torch.compile(run_torch_comp_fwd, backend="inductor")
    
    # Warmup and verify autograd compiles
    qkv_warmup = qkv.clone().detach().requires_grad_(True)
    rms_warmup = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_warmup = scales.clone().detach().requires_grad_(True)
    
    # Warmup Triton
    q_t, k_t = compiled_triton(qkv_warmup, rms_warmup, s_warmup, sin, cos)
    (q_t.sum() + k_t.sum()).backward()
    qkv_warmup.grad.zero_()
    rms_warmup.grad.zero_()
    s_warmup.grad.zero_()
    
    # Warmup CuTe
    q_c, k_c = compiled_cute(qkv_warmup, rms_warmup, s_warmup, sin, cos)
    (q_c.sum() + k_c.sum()).backward()
    qkv_warmup.grad.zero_()
    rms_warmup.grad.zero_()
    s_warmup.grad.zero_()
    
    # Warmup Eager
    q_e, k_e = compiled_eager(qkv_warmup, rms_warmup, s_warmup, sin, cos)
    (q_e.sum() + k_e.sum()).backward()
    qkv_warmup.grad.zero_()
    rms_warmup.grad.zero_()
    s_warmup.grad.zero_()

    # Warmup Torch Compiled
    q_tc, k_tc = compiled_torch_comp(qkv_warmup, rms_warmup, s_warmup, sin, cos)
    (q_tc.sum() + k_tc.sum()).backward()
    
    torch.cuda.synchronize()
    print("Compilation and warmup successful!")

    # Helper to measure memory and time for Fwd + Bwd
    def run_benchmark_fwd_bwd(compiled_fn, label):
        qkv_bench = qkv.clone().detach().requires_grad_(True)
        rms_bench = rmsnorm_weight.clone().detach().requires_grad_(True)
        s_bench = scales.clone().detach().requires_grad_(True)
        
        # Warmup iterations
        for _ in range(10):
            q_out, k_out = compiled_fn(qkv_bench, rms_bench, s_bench, sin, cos)
            loss = q_out.sum() + k_out.sum()
            loss.backward()
            qkv_bench.grad.zero_()
            rms_bench.grad.zero_()
            s_bench.grad.zero_()
            
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            q_out, k_out = compiled_fn(qkv_bench, rms_bench, s_bench, sin, cos)
            loss = q_out.sum() + k_out.sum()
            loss.backward()
            qkv_bench.grad.zero_()
            rms_bench.grad.zero_()
            s_bench.grad.zero_()
        end_event.record()
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / 100.0
        mem_peak = torch.cuda.max_memory_allocated()
        
        print(f"\n[{label} Fwd+Bwd Benchmark Results]")
        print(f"Average Execution Time: {time_ms:.4f} ms")
        print(f"Peak VRAM Allocated: {mem_peak / (1024**2):.2f} MB")
        return time_ms

    # Run benchmarks
    triton_time = run_benchmark_fwd_bwd(compiled_triton, "Triton Epilogue")
    cute_time = run_benchmark_fwd_bwd(compiled_cute, "CuTe DSL Fused Epilogue")
    eager_time = run_benchmark_fwd_bwd(compiled_eager, "Eager Epilogue")
    torch_comp_time = run_benchmark_fwd_bwd(compiled_torch_comp, "Torch Compiled (Inductor)")
    
    print("\n[Comparison Summary (Fwd + Bwd)]")
    print(f"Speedup vs Triton: {triton_time / cute_time:.2f}x")
    print(f"Speedup vs Eager: {eager_time / cute_time:.2f}x")
    print(f"Speedup vs Torch Compiled (Inductor): {torch_comp_time / cute_time:.2f}x")

    # Measure detailed numerical accuracy for both Forward and Backward
    qkv_e = qkv.clone().detach().requires_grad_(True)
    rms_e = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_e = scales.clone().detach().requires_grad_(True)
    
    q_e, k_e = compiled_eager(qkv_e, rms_e, s_e, sin, cos)
    (q_e.sum() + k_e.sum()).backward()
    
    qkv_t = qkv.clone().detach().requires_grad_(True)
    rms_t = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_t = scales.clone().detach().requires_grad_(True)
    
    q_t, k_t = compiled_triton(qkv_t, rms_t, s_t, sin, cos)
    (q_t.sum() + k_t.sum()).backward()
    
    qkv_c = qkv.clone().detach().requires_grad_(True)
    rms_c = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_c = scales.clone().detach().requires_grad_(True)
    
    q_c, k_c = compiled_cute(qkv_c, rms_c, s_c, sin, cos)
    (q_c.sum() + k_c.sum()).backward()

    qkv_tc = qkv.clone().detach().requires_grad_(True)
    rms_tc = rmsnorm_weight.clone().detach().requires_grad_(True)
    s_tc = scales.clone().detach().requires_grad_(True)
    
    q_tc, k_tc = compiled_torch_comp(qkv_tc, rms_tc, s_tc, sin, cos)
    (q_tc.sum() + k_tc.sum()).backward()
    
    torch.cuda.synchronize()

    def print_accuracy_metrics(c_tensor, e_tensor, label):
        diff = (c_tensor.float() - e_tensor.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"--- Detailed Numerical Accuracy (CuTe vs Eager) for {label} ---")
        print(f"Max Absolute Difference: {max_abs:.6f}")
        print(f"Mean Absolute Difference: {mean_abs:.6f}")
        
    def print_triton_accuracy_metrics(c_tensor, t_tensor, label):
        diff = (c_tensor.float() - t_tensor.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"--- Detailed Numerical Accuracy (CuTe vs Triton) for {label} ---")
        print(f"Max Absolute Difference: {max_abs:.6f}")
        print(f"Mean Absolute Difference: {mean_abs:.6f}")
 
    def print_torch_comp_accuracy_metrics(c_tensor, tc_tensor, label):
        diff = (c_tensor.float() - tc_tensor.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"--- Detailed Numerical Accuracy (CuTe vs Torch Compiled) for {label} ---")
        print(f"Max Absolute Difference: {max_abs:.6f}")
        print(f"Mean Absolute Difference: {mean_abs:.6f}")

    print("\n[Accuracy Metrics - Forward]")
    print_accuracy_metrics(q_c, q_e, "Q Proj Output")
    print_accuracy_metrics(k_c, k_e, "K Proj Output")
    print_triton_accuracy_metrics(q_c, q_t, "Q Proj Output")
    print_triton_accuracy_metrics(k_c, k_t, "K Proj Output")
    print_torch_comp_accuracy_metrics(q_c, q_tc, "Q Proj Output")
    print_torch_comp_accuracy_metrics(k_c, k_tc, "K Proj Output")
    
    print("\n[Accuracy Metrics - Backward]")
    print_accuracy_metrics(qkv_c.grad, qkv_e.grad, "qkv Gradient")
    print_accuracy_metrics(rms_c.grad, rms_e.grad, "rmsnorm_weight Gradient")
    print_accuracy_metrics(s_c.grad, s_e.grad, "scales Gradient")
    
    print_triton_accuracy_metrics(qkv_c.grad, qkv_t.grad, "qkv Gradient")
    print_triton_accuracy_metrics(rms_c.grad, rms_t.grad, "rmsnorm_weight Gradient")
    print_triton_accuracy_metrics(s_c.grad, s_t.grad, "scales Gradient")

    print_torch_comp_accuracy_metrics(qkv_c.grad, qkv_tc.grad, "qkv Gradient")
    print_torch_comp_accuracy_metrics(rms_c.grad, rms_tc.grad, "rmsnorm_weight Gradient")
    print_torch_comp_accuracy_metrics(s_c.grad, s_tc.grad, "scales Gradient")

    # Assert close with precision-aware tolerances for bfloat16
    torch.testing.assert_close(q_c, q_e, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_c, k_e, rtol=1e-2, atol=1e-1)
    
    # Assert close for backward gradients
    torch.testing.assert_close(qkv_c.grad, qkv_e.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(rms_c.grad, rms_e.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(s_c.grad, s_e.grad, rtol=2e-2, atol=6.0)

    torch.testing.assert_close(q_c, q_tc, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_c, k_tc, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(qkv_c.grad, qkv_tc.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(rms_c.grad, rms_tc.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(s_c.grad, s_tc.grad, rtol=1e-2, atol=1e-1)

    print("\nAll Forward and Backward Parity Assertions Passed!")

if __name__ == '__main__':
    test_cute_fused_qkv_forward_parity_and_benchmark()
    test_cute_fused_epilogue_parity_and_benchmark()
