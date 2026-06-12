import os
import gc
import torch
import torch.nn as nn
import flash_attn_prologue
from mamba3 import Mamba3SingleStepUpdate, init_mamba_state
from cute.cute_mamba3 import cute_mamba3
import time

def benchmark_mamba3():
    # Setup shape dimensions matching standard temporal Mamba 3 profile
    B = 32
    S = 64
    NH = 4
    SD = 64  # N (state dimension)
    HD = 64  # P (head dimension)
    MR = 4   # R (MIMO rank)
    D = NH * HD
    device = "cuda"
    dtype = torch.bfloat16

    print(f"============================================================")
    print(f"Benchmarking Temporal Mamba 3 Kernel:")
    print(f"Shape: B={B}, S={S}, NH={NH}, SD={SD}, HD={HD}, MR={MR}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"============================================================")

    # Initialize PyTorch module
    model_eager = Mamba3SingleStepUpdate(
        hidden_dim=D,
        num_heads=NH,
        head_dim=HD,
        state_dim=SD,
        mimo_rank=MR,
        backend="native"
    ).to(device).bfloat16()
    model_eager.reset_parameters()

    model_cute = Mamba3SingleStepUpdate(
        hidden_dim=D,
        num_heads=NH,
        head_dim=HD,
        state_dim=SD,
        mimo_rank=MR,
        backend="cute"
    ).to(device).bfloat16()
    model_cute.reset_parameters()

    # Share weights
    with torch.no_grad():
        model_cute.in_proj.weight.copy_(model_eager.in_proj.weight)
        model_cute.out_proj.weight.copy_(model_eager.out_proj.weight)
        model_cute.W_X.copy_(model_eager.W_X)
        model_cute.W_Z.copy_(model_eager.W_Z)
        model_cute.W_O.copy_(model_eager.W_O)
        model_cute.B_bias.copy_(model_eager.B_bias)
        model_cute.C_bias.copy_(model_eager.C_bias)
        model_cute.dt_bias.copy_(model_eager.dt_bias)
        model_cute.D_skip.copy_(model_eager.D_skip)

    # Create inputs
    curr_mems = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    ssm_state, trap_mem, rope_angles = init_mamba_state(
        batch_size=B, seq_len=S, num_heads=NH, state_dim=SD, head_dim=HD, device=device, dtype=torch.float32
    )
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True

    # Clone inputs for each backend
    inputs_eager = (
        curr_mems.clone().detach().requires_grad_(True),
        ssm_state.clone().detach().requires_grad_(True),
        trap_mem.clone().detach().requires_grad_(True),
        rope_angles.clone().detach()
    )
    inputs_inductor = (
        curr_mems.clone().detach().requires_grad_(True),
        ssm_state.clone().detach().requires_grad_(True),
        trap_mem.clone().detach().requires_grad_(True),
        rope_angles.clone().detach()
    )
    inputs_cute = (
        curr_mems.clone().detach().requires_grad_(True),
        ssm_state.clone().detach().requires_grad_(True),
        trap_mem.clone().detach().requires_grad_(True),
        rope_angles.clone().detach()
    )

    # Warmup and correctness check
    print("Verifying numerical parity...")
    out_eager, state_eager, V_eager, angles_eager = model_eager(*inputs_eager)
    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)

    # Warmup CuTe
    print("Warming up CuTe...")
    out_cute, state_cute, V_cute, angles_cute = model_cute(*inputs_cute)
    out_cute.backward(grad_out)

    # Compile the eager module using Inductor
    # We define a helper that wraps the model forward call
    def inductor_fwd_step(curr_mems, ssm_state, trap_mem, rope_angles):
        return model_eager(curr_mems, ssm_state, trap_mem, rope_angles)

    print("Compiling model using PyTorch Inductor (torch.compile)...")
    inductor_compiled_fwd = torch.compile(inductor_fwd_step, fullgraph=True)
    
    # Warmup Inductor
    print("Warming up Inductor...")
    out_ind, state_ind, V_ind = inductor_compiled_fwd(inputs_inductor[0], inputs_inductor[1], inputs_inductor[2], inputs_inductor[3])[:3]
    out_ind.backward(grad_out)

    # Assert correctness
    torch.testing.assert_close(out_cute, out_eager, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_cute, state_eager, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_ind, out_eager, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_ind, state_eager, rtol=1e-3, atol=1e-3)

    # Check gradients
    torch.testing.assert_close(inputs_cute[0].grad, inputs_eager[0].grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(inputs_cute[1].grad, inputs_eager[1].grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(inputs_inductor[0].grad, inputs_eager[0].grad, rtol=1e-3, atol=1e-3)
    
    print("Numerical parity check passed successfully!")

    # Helper function to measure runtime
    def measure_time(fn, inputs, grad_out, num_iters=100):
        # Warmup
        for _ in range(10):
            out = fn(*inputs)
            if grad_out is not None:
                out.backward(grad_out, retain_graph=False)
                for inp in inputs:
                    if inp.grad is not None:
                        inp.grad.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            out = fn(*inputs)
            if grad_out is not None:
                out.backward(grad_out, retain_graph=False)
                
        torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) * 1000.0 / num_iters

    # Helper function to measure VRAM (in MB)
    def measure_vram(fn, inputs, grad_out):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        
        # Run FWD + BWD step
        out = fn(*inputs)
        if grad_out is not None:
            out.backward(grad_out)
            
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        net_bytes = peak_bytes - mem_before
        return net_bytes / (1024.0 * 1024.0)

    # Define runs
    def run_eager(*args):
        # Ignore returned angles for timing
        return model_eager(*args)[0]

    def run_inductor(*args):
        return inductor_compiled_fwd(*args)[0]

    def run_cute(*args):
        return model_cute(*args)[0]

    print("\nMeasuring Runtimes (average of 100 iterations)...")
    # Forward-only times
    fwd_eager_ms = measure_time(run_eager, inputs_eager, None)
    fwd_inductor_ms = measure_time(run_inductor, inputs_inductor, None)
    fwd_cute_ms = measure_time(run_cute, inputs_cute, None)

    # Forward + Backward times
    # Clear gradients first
    for inp in inputs_eager:
        if inp.grad is not None:
            inp.grad.zero_()
    for inp in inputs_inductor:
        if inp.grad is not None:
            inp.grad.zero_()
    for inp in inputs_cute:
        if inp.grad is not None:
            inp.grad.zero_()

    fwd_bwd_eager_ms = measure_time(run_eager, inputs_eager, grad_out)
    fwd_bwd_inductor_ms = measure_time(run_inductor, inputs_inductor, grad_out)
    fwd_bwd_cute_ms = measure_time(run_cute, inputs_cute, grad_out)

    print("\nMeasuring Peak VRAM consumption...")
    vram_eager = measure_vram(run_eager, inputs_eager, grad_out)
    vram_inductor = measure_vram(run_inductor, inputs_inductor, grad_out)
    vram_cute = measure_vram(run_cute, inputs_cute, grad_out)

    # Print results
    print("\n============================================================")
    print("BENCHMARK RESULTS TABLE:")
    print("============================================================")
    print(f"| Backend  | FWD Runtime (ms) | FWD+BWD Runtime (ms) | Peak VRAM (MB) |")
    print(f"| :--- | :---: | :---: | :---: |")
    print(f"| Eager    | {fwd_eager_ms:16.4f} | {fwd_bwd_eager_ms:20.4f} | {vram_eager:14.2f} |")
    print(f"| Inductor | {fwd_inductor_ms:16.4f} | {fwd_bwd_inductor_ms:20.4f} | {vram_inductor:14.2f} |")
    print(f"| CuTe     | {fwd_cute_ms:16.4f} | {fwd_bwd_cute_ms:20.4f} | {vram_cute:14.2f} |")
    print("============================================================")
    
    speedup_fwd = fwd_inductor_ms / fwd_cute_ms
    speedup_fwd_bwd = fwd_bwd_inductor_ms / fwd_bwd_cute_ms
    vram_saved = vram_inductor - vram_cute
    vram_saved_pct = (vram_saved / vram_inductor) * 100.0 if vram_inductor > 0 else 0
    
    print(f"CuTe speedup vs Inductor FWD: {speedup_fwd:.2f}x")
    print(f"CuTe speedup vs Inductor FWD+BWD: {speedup_fwd_bwd:.2f}x")
    print(f"CuTe VRAM savings vs Inductor: {vram_saved:.2f} MB ({vram_saved_pct:.1f}%)")

if __name__ == "__main__":
    benchmark_mamba3()
