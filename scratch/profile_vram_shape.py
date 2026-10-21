import os
import sys
import gc
import time
from typing import Dict, Any

# Ensure both workspace roots are in path
sys.path.insert(0, "/home/achiya/vs_code_main/flash_attn_prologue")
sys.path.insert(0, "/home/achiya/vs_code_main/Advantage-First-RL")

import torch
import torch.nn as nn
from temporal_mamba3.mamba3 import Mamba3SingleStepUpdate, init_mamba_state

def run_profiling():
    # User's exact training shape
    B = 16
    S = 64
    NH = 4
    SD = 32  # dstate
    HD = 64  # dhead
    MR = 4   # mimo rank
    D = 256  # hidden = NH * HD
    device = "cuda"
    dtype = torch.bfloat16

    print(f"============================================================")
    # Print CUDA device name
    print(f"Profiling Mamba-3 Kernel on: {torch.cuda.get_device_name(0)}")
    print(f"Shape: Batch={B}, Sequence={S}, Heads={NH}, dstate={SD}, dhead={HD}, MIMO Rank={MR}, Hidden={D}")
    print(f"============================================================")

    # Isolated helper to profile a single model backend
    def profile_backend(backend_name: str):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Initialize model
        model = Mamba3SingleStepUpdate(
            hidden_dim=D,
            num_heads=NH,
            head_dim=HD,
            state_dim=SD,
            mimo_rank=MR,
            backend=backend_name
        ).to(device).bfloat16()
        model.reset_parameters()

        # Generate inputs
        curr_mems = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
        ssm_state, trap_mem, rope_angles = init_mamba_state(
            batch_size=B, seq_len=S, num_heads=NH, state_dim=SD, head_dim=HD, device=device, dtype=torch.float32
        )
        ssm_state.requires_grad = True
        trap_mem.requires_grad = True
        grad_out = torch.randn(B, S, D, device=device, dtype=dtype)

        # Helper to compile if inductor
        if backend_name == "inductor":
            @torch.compile(fullgraph=True)
            def fwd_compiled(curr_mems, ssm_state, trap_mem, rope_angles):
                return model(curr_mems, ssm_state, trap_mem, rope_angles)
            fwd_fn = fwd_compiled
        else:
            fwd_fn = model

        # Warmup (necessary for compilation/JIT/autotuning)
        out, next_state, next_V, _ = fwd_fn(curr_mems, ssm_state, trap_mem, rope_angles)
        loss = out.sum() + next_state.sum() + next_V.sum()
        loss.backward()
        if curr_mems.grad is not None:
            curr_mems.grad.zero_()
        if ssm_state.grad is not None:
            ssm_state.grad.zero_()
        if trap_mem.grad is not None:
            trap_mem.grad.zero_()
        model.zero_grad(set_to_none=True)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 1. Profile Forward-Only Memory
        mem_before_fwd = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            out_fwd, state_fwd, V_fwd, _ = fwd_fn(curr_mems, ssm_state, trap_mem, rope_angles)
            torch.cuda.synchronize()
            peak_bytes_fwd = torch.cuda.max_memory_allocated()
            net_fwd_mb = (peak_bytes_fwd - mem_before_fwd) / (1024.0 * 1024.0)

        # Clean fwd outputs
        del out_fwd, state_fwd, V_fwd
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 2. Profile Forward + Backward Memory
        mem_before_bwd = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        out_bwd, state_bwd, V_bwd, _ = fwd_fn(curr_mems, ssm_state, trap_mem, rope_angles)
        loss_bwd = out_bwd.sum() + state_bwd.sum() + V_bwd.sum()
        loss_bwd.backward()
        torch.cuda.synchronize()

        peak_bytes_bwd = torch.cuda.max_memory_allocated()
        net_bwd_mb = (peak_bytes_bwd - mem_before_bwd) / (1024.0 * 1024.0)

        # 3. Profile Runtimes
        # Warmup for timing
        for _ in range(5):
            out_t, state_t, V_t, _ = fwd_fn(curr_mems, ssm_state, trap_mem, rope_angles)
            loss_t = out_t.sum() + state_t.sum() + V_t.sum()
            loss_t.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            out_t, state_t, V_t, _ = fwd_fn(curr_mems, ssm_state, trap_mem, rope_angles)
            loss_t = out_t.sum() + state_t.sum() + V_t.sum()
            loss_t.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        fwd_bwd_ms = (t_end - t_start) * 1000.0 / iters

        # Cleanup everything from GPU
        del model, curr_mems, ssm_state, trap_mem, rope_angles, grad_out
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return net_fwd_mb, net_bwd_mb, fwd_bwd_ms

    print("Profiling Eager (native)...")
    eager_fwd, eager_bwd, eager_time = profile_backend("native")

    print("Profiling Inductor (torch.compile)...")
    inductor_fwd, inductor_bwd, inductor_time = profile_backend("inductor")

    print("Profiling CuTe (custom CUDA/Triton FFI)...")
    cute_fwd, cute_bwd, cute_time = profile_backend("cute")

    # Print results
    print("\n==========================================================================")
    print("VRAM AND RUNTIME PROFILING RESULTS (B=16, S=64, N=32, NH=4, P=64, R=4, D=256):")
    print("==========================================================================")
    print(f"| Backend  | FWD VRAM (MB) | FWD+BWD VRAM (MB) | FWD+BWD Step Runtime (ms) |")
    print(f"| :---     | :---:         | :---:            | :---:                     |")
    print(f"| Eager    | {eager_fwd:12.2f}  | {eager_bwd:15.2f}  | {eager_time:24.3f}  |")
    print(f"| Inductor | {inductor_fwd:12.2f}  | {inductor_bwd:15.2f}  | {inductor_time:24.3f}  |")
    print(f"| CuTe     | {cute_fwd:12.2f}  | {cute_bwd:15.2f}  | {cute_time:24.3f}  |")
    print("==========================================================================")

    vram_saved = inductor_bwd - cute_bwd
    pct_saved = (vram_saved / inductor_bwd) * 100.0 if inductor_bwd > 0 else 0
    speedup = inductor_time / cute_time if cute_time > 0 else 0

    print(f"\nCuTe VRAM savings (FWD+BWD) vs Inductor: {vram_saved:.2f} MB ({pct_saved:.1f}% reduction)")
    print(f"CuTe Speedup (FWD+BWD) vs Inductor: {speedup:.2f}x")

if __name__ == "__main__":
    run_profiling()
