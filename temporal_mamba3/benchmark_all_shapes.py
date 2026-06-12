import os
import gc
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flash_attn_prologue
from temporal_mamba3.mamba3 import Mamba3SingleStepUpdate, init_mamba_state
from temporal_mamba3.cute.cute_mamba3 import cute_mamba3

def measure_time(fn, inputs, grad_out, num_iters=50):
    # Warmup
    for _ in range(5):
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

def measure_vram(fn, inputs, grad_out):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    
    out = fn(*inputs)
    if grad_out is not None:
        out.backward(grad_out)
        
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    net_bytes = peak_bytes - mem_before
    return net_bytes / (1024.0 * 1024.0)

def verify_correctness(model_eager, model_cute, inductor_compiled_fwd, B, S, NH, SD, HD, MR, device, dtype):
    curr_mems = torch.randn(B, S, NH * HD, device=device, dtype=dtype, requires_grad=True)
    ssm_state, trap_mem, rope_angles = init_mamba_state(
        batch_size=B, seq_len=S, num_heads=NH, state_dim=SD, head_dim=HD, device=device, dtype=torch.float32
    )
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True

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

    out_eager, state_eager, V_eager, angles_eager = model_eager(*inputs_eager)
    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)

    out_ind, state_ind, V_ind = inductor_compiled_fwd(
        inputs_inductor[0], inputs_inductor[1], inputs_inductor[2], inputs_inductor[3]
    )[:3]
    out_ind.backward(grad_out)

    out_cute, state_cute, V_cute, angles_cute = model_cute(*inputs_cute)
    out_cute.backward(grad_out)

    # Assert correctness
    torch.testing.assert_close(out_cute, out_eager, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_cute, state_eager, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_ind, out_eager, rtol=1e-2, atol=1e-2)

    # Check gradients
    torch.testing.assert_close(inputs_cute[0].grad, inputs_eager[0].grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inputs_cute[1].grad, inputs_eager[1].grad, rtol=1e-2, atol=1e-2)

def profile_backend(model_fn, B, S, NH, SD, HD, MR, device, dtype):
    curr_mems = torch.randn(B, S, NH * HD, device=device, dtype=dtype, requires_grad=True)
    ssm_state, trap_mem, rope_angles = init_mamba_state(
        batch_size=B, seq_len=S, num_heads=NH, state_dim=SD, head_dim=HD, device=device, dtype=torch.float32
    )
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True
    inputs = (curr_mems, ssm_state, trap_mem, rope_angles)
    
    with torch.no_grad():
        out_temp = model_fn(*inputs)
        if isinstance(out_temp, tuple):
            out_temp = out_temp[0]
        grad_out = torch.randn_like(out_temp)
        
    def run_fn(*args):
        res = model_fn(*args)
        if isinstance(res, tuple):
            return res[0]
        return res

    fwd_time = measure_time(run_fn, inputs, None)
    fwd_bwd_time = measure_time(run_fn, inputs, grad_out)
    vram_used = measure_vram(run_fn, inputs, grad_out)
    
    return fwd_time, fwd_bwd_time, vram_used

def run_single_benchmark(B, S, NH, SD, HD, MR):
    device = "cuda"
    dtype = torch.bfloat16
    D = NH * HD
    
    # Instantiate models
    model_eager = Mamba3SingleStepUpdate(
        hidden_dim=D, num_heads=NH, head_dim=HD, state_dim=SD, mimo_rank=MR, backend="native"
    ).to(device).bfloat16()
    model_eager.reset_parameters()

    model_cute = Mamba3SingleStepUpdate(
        hidden_dim=D, num_heads=NH, head_dim=HD, state_dim=SD, mimo_rank=MR, backend="cute"
    ).to(device).bfloat16()
    model_cute.reset_parameters()

    # Align weights
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

    # Compile the eager module using Inductor
    def inductor_fwd_step(curr_mems, ssm_state, trap_mem, rope_angles):
        return model_eager(curr_mems, ssm_state, trap_mem, rope_angles)
    
    inductor_compiled_fwd = torch.compile(inductor_fwd_step, fullgraph=True)

    # 1. Verify correctness
    verify_correctness(model_eager, model_cute, inductor_compiled_fwd, B, S, NH, SD, HD, MR, device, dtype)
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Profile Eager
    fwd_eager, fwd_bwd_eager, vram_eager = profile_backend(
        model_eager, B, S, NH, SD, HD, MR, device, dtype
    )
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Profile Inductor
    fwd_inductor, fwd_bwd_inductor, vram_inductor = profile_backend(
        inductor_compiled_fwd, B, S, NH, SD, HD, MR, device, dtype
    )
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Profile CuTe
    fwd_cute, fwd_bwd_cute, vram_cute = profile_backend(
        model_cute, B, S, NH, SD, HD, MR, device, dtype
    )
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fwd": {"eager": fwd_eager, "inductor": fwd_inductor, "cute": fwd_cute},
        "fwd_bwd": {"eager": fwd_bwd_eager, "inductor": fwd_bwd_inductor, "cute": fwd_bwd_cute},
        "vram": {"eager": vram_eager, "inductor": vram_inductor, "cute": vram_cute}
    }


def main():
    # Shape Configurations to profile:
    # HD = 64, NH = 4, MR in [2, 4], SD in [16, 32, 64]
    HD = 64
    NH = 4
    
    # We sweep across three workloads representing Small, Medium, Large, X-Large
    workloads = [
        {"B": 2, "S": 64, "name": "Small (BS=128)"},
        {"B": 8, "S": 128, "name": "Medium (BS=1024)"},
        {"B": 32, "S": 64, "name": "Large (BS=2048)"},
        {"B": 64, "S": 128, "name": "X-Large (BS=8192)"}
    ]

    shapes = [
        {"SD": 64, "MR": 4},
        {"SD": 32, "MR": 4},
        {"SD": 16, "MR": 4},
        {"SD": 64, "MR": 2},
        {"SD": 32, "MR": 2},
        {"SD": 16, "MR": 2}
    ]

    results = []

    print("=========================================================================")
    print("STARTING TEMPORAL MAMBA 3 AUTOTUNING AND BENCHMARKING SWEEP")
    print("=========================================================================")

    for shape in shapes:
        SD, MR = shape["SD"], shape["MR"]
        for wl in workloads:
            B, S = wl["B"], wl["S"]
            print(f"\n---> Profiling Shape P=64, N={SD}, R={MR} | Workload: {wl['name']}")
            try:
                import subprocess
                cmd = [
                    sys.executable,
                    __file__,
                    "--subprocess-run",
                    str(B), str(S), str(NH), str(SD), str(HD), str(MR)
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ":" + env.get("PYTHONPATH", "")
                
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                res = None
                for line in proc.stdout.splitlines():
                    if line.startswith("__RESULT__ "):
                        import json
                        res = json.loads(line[11:])
                    elif line.startswith("__ERROR__ "):
                        print(f"     [ERROR] Subprocess reported error: {line[10:]}")
                
                if res is None:
                    if proc.stderr:
                        print(f"     [ERROR] Subprocess crashed. Stderr:\n{proc.stderr}")
                    else:
                        print(f"     [ERROR] Subprocess failed with no result.")
                else:
                    speedup_fwd = res["fwd"]["inductor"] / res["fwd"]["cute"]
                    speedup_fwd_bwd = res["fwd_bwd"]["inductor"] / res["fwd_bwd"]["cute"]
                    vram_saved = res["vram"]["inductor"] - res["vram"]["cute"]
                    vram_saved_pct = (vram_saved / res["vram"]["inductor"]) * 100.0 if res["vram"]["inductor"] > 0 else 0
                    
                    print(f"     FWD Speedup: {speedup_fwd:.2f}x | FWD+BWD Speedup: {speedup_fwd_bwd:.2f}x | VRAM Saved: {vram_saved:.1f} MB ({vram_saved_pct:.1f}%)")
                    
                    results.append({
                        "SD": SD, "MR": MR, "B": B, "S": S, "wl_name": wl["name"],
                        "fwd": res["fwd"], "fwd_bwd": res["fwd_bwd"], "vram": res["vram"]
                    })
            except Exception as e:
                print(f"     [ERROR] Failed to run subprocess: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    # Print a beautiful markdown report table
    print("\n\n=========================================================================")
    print("BENCHMARK REPORT TABLE")
    print("=========================================================================")
    
    print("| Shape (N_R) | Workload | Backend | FWD (ms) | FWD+BWD (ms) | Peak VRAM (MB) | FWD Speedup | FWD+BWD Speedup | VRAM Savings |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for r in results:
        shape_str = f"{r['SD']}_{r['MR']}"
        wl_str = r["wl_name"]
        
        # Eager Row
        print(f"| {shape_str} | {wl_str} | Eager | {r['fwd']['eager']:.3f} | {r['fwd_bwd']['eager']:.3f} | {r['vram']['eager']:.1f} | - | - | - |")
        
        # Inductor Row
        print(f"| | | Inductor | {r['fwd']['inductor']:.3f} | {r['fwd_bwd']['inductor']:.3f} | {r['vram']['inductor']:.1f} | Ref | Ref | Ref |")
        
        # CuTe Row
        speedup_fwd = r["fwd"]["inductor"] / r["fwd"]["cute"]
        speedup_fwd_bwd = r["fwd_bwd"]["inductor"] / r["fwd_bwd"]["cute"]
        vram_saved = r["vram"]["inductor"] - r["vram"]["cute"]
        vram_saved_pct = (vram_saved / r["vram"]["inductor"]) * 100.0 if r["vram"]["inductor"] > 0 else 0
        
        print(f"| | | **CuTe** | **{r['fwd']['cute']:.3f}** | **{r['fwd_bwd']['cute']:.3f}** | **{r['vram']['cute']:.1f}** | **{speedup_fwd:.2f}x** | **{speedup_fwd_bwd:.2f}x** | **{vram_saved:.1f} MB ({vram_saved_pct:.1f}%)** |")
        print("|---|---|---|---|---|---|---|---|---|")

if __name__ == "__main__":
    if "--subprocess-run" in sys.argv:
        idx = sys.argv.index("--subprocess-run")
        B = int(sys.argv[idx+1])
        S = int(sys.argv[idx+2])
        NH = int(sys.argv[idx+3])
        SD = int(sys.argv[idx+4])
        HD = int(sys.argv[idx+5])
        MR = int(sys.argv[idx+6])
        try:
            res = run_single_benchmark(B, S, NH, SD, HD, MR)
            import json
            print(f"__RESULT__ {json.dumps(res)}")
        except Exception as e:
            import traceback
            print(f"__ERROR__ {str(e)}")
            traceback.print_exc()
        sys.exit(0)
    else:
        main()
