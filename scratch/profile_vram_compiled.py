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
from torch.utils.checkpoint import checkpoint
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
    print(f"Profiling Compiled Mamba-3 (torch.compile + custom op + checkpointing):")
    print(f"Shape: Batch={B}, Sequence={S}, Heads={NH}, dstate={SD}, dhead={HD}, MIMO Rank={MR}, Hidden={D}")
    print(f"============================================================")

    # Isolated helper to profile a single model backend
    def profile_compiled_backend(backend_name: str, use_checkpoint: bool):
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

        # Helper step function
        if use_checkpoint:
            def step_fn(curr_mems, ssm_state, trap_mem, rope_angles):
                # Use reentrant=False to avoid deprecation and ensure full compatibility with torch.compile
                return checkpoint(model, curr_mems, ssm_state, trap_mem, rope_angles, use_reentrant=False)
        else:
            def step_fn(curr_mems, ssm_state, trap_mem, rope_angles):
                return model(curr_mems, ssm_state, trap_mem, rope_angles)

        # Compile the step function
        compiled_step = torch.compile(step_fn, fullgraph=True)

        # Warmup and compile graph
        print(f"Warmup compiling backend={backend_name}, checkpointed={use_checkpoint}...")
        out, next_state, next_V, _ = compiled_step(curr_mems, ssm_state, trap_mem, rope_angles)
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
            out_fwd, state_fwd, V_fwd, _ = compiled_step(curr_mems, ssm_state, trap_mem, rope_angles)
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

        out_bwd, state_bwd, V_bwd, _ = compiled_step(curr_mems, ssm_state, trap_mem, rope_angles)
        loss_bwd = out_bwd.sum() + state_bwd.sum() + V_bwd.sum()
        loss_bwd.backward()
        torch.cuda.synchronize()

        peak_bytes_bwd = torch.cuda.max_memory_allocated()
        net_bwd_mb = (peak_bytes_bwd - mem_before_bwd) / (1024.0 * 1024.0)

        # 3. Profile Runtimes
        # Warmup for timing
        for _ in range(5):
            out_t, state_t, V_t, _ = compiled_step(curr_mems, ssm_state, trap_mem, rope_angles)
            loss_t = out_t.sum() + state_t.sum() + V_t.sum()
            loss_t.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            out_t, state_t, V_t, _ = compiled_step(curr_mems, ssm_state, trap_mem, rope_angles)
            loss_t = out_t.sum() + state_t.sum() + V_t.sum()
            loss_t.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        fwd_bwd_ms = (t_end - t_start) * 1000.0 / iters

        # Cleanup everything from GPU
        del model, compiled_step, curr_mems, ssm_state, trap_mem, rope_angles
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return net_fwd_mb, net_bwd_mb, fwd_bwd_ms

    print("\nProfiling Native...")
    native_fwd, native_bwd, native_time = profile_compiled_backend("native", use_checkpoint=False)

    print("\nProfiling Native + Checkpointed...")
    native_ckp_fwd, native_ckp_bwd, native_ckp_time = profile_compiled_backend("native", use_checkpoint=True)

    print("\nProfiling CuTe...")
    cute_fwd, cute_bwd, cute_time = profile_compiled_backend("cute", use_checkpoint=False)

    print("\nProfiling CuTe + Checkpointed...")
    cute_ckp_fwd, cute_ckp_bwd, cute_ckp_time = profile_compiled_backend("cute", use_checkpoint=True)

    # Print results
    print("\n====================================================================================")
    print("PROFILING RESULTS (B=16, S=64, N=32, NH=4, P=64, R=4, D=256):")
    print("====================================================================================")
    print(f"| Configuration           | FWD VRAM (MB) | FWD+BWD VRAM (MB) | FWD+BWD Runtime (ms) |")
    print(f"| :---                    | :---:         | :---:            | :---:                |")
    print(f"| Native Compiled         | {native_fwd:12.2f}  | {native_bwd:15.2f}  | {native_time:20.3f} |")
    print(f"| Native Compiled + CKP   | {native_ckp_fwd:12.2f}  | {native_ckp_bwd:15.2f}  | {native_ckp_time:20.3f} |")
    print(f"| CuTe Compiled           | {cute_fwd:12.2f}  | {cute_bwd:15.2f}  | {cute_time:20.3f} |")
    print(f"| CuTe Compiled + CKP     | {cute_ckp_fwd:12.2f}  | {cute_ckp_bwd:15.2f}  | {cute_ckp_time:20.3f} |")
    print("====================================================================================")

    vram_saved = native_bwd - cute_ckp_bwd
    pct_saved = (vram_saved / native_bwd) * 100.0 if native_bwd > 0 else 0

    print(f"\nMax VRAM savings (CuTe+CKP vs Native Compiled): {vram_saved:.2f} MB ({pct_saved:.1f}% reduction)")

if __name__ == "__main__":
    run_profiling()
