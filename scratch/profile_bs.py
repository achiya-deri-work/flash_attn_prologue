import sys
import os
import torch
import flash_attn_prologue

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_mamba3.cute.autotuner import get_smem_requirements
from flash_attn_prologue.coordinate_descent import CoordinateDescentAutotuner

def profile_shape_with_bs(B: int, S: int, H: int, P: int, N: int, R: int, device: str):
    from temporal_mamba3.cute.cute_mamba3 import (
        get_cute_mamba3_fwd_kernel,
        get_cute_mamba3_bwd_kernel
    )
    
    props = torch.cuda.get_device_properties(device)
    max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    
    # Setup inputs with custom B and S
    x_raw = torch.randn(B, S, H, P, device=device, dtype=torch.float32)
    z_raw = torch.randn(B, S, H, P, device=device, dtype=torch.float32)
    B_rot = torch.randn(B, S, H, N, R, device=device, dtype=torch.float32)
    C_rot = torch.randn(B, S, H, N, R, device=device, dtype=torch.float32)
    dt = torch.rand(B, S, H, device=device, dtype=torch.float32)
    A = torch.randn(B, S, H, device=device, dtype=torch.float32)
    lam = torch.rand(B, S, H, device=device, dtype=torch.float32)
    ssm_state = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    trap_mem = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    W_X = torch.randn(H, P, R, device=device, dtype=torch.float32)
    W_Z = torch.randn(H, P, R, device=device, dtype=torch.float32)
    W_O = torch.randn(H, P, R, device=device, dtype=torch.float32)
    D_skip = torch.randn(H, P, device=device, dtype=torch.float32)
    
    out = torch.empty((B, S, H, P), dtype=torch.bfloat16, device=device)
    curr_state = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    curr_V = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    
    grad_out = torch.randn(B, S, H, P, device=device, dtype=torch.float32)
    grad_curr_state = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    grad_curr_V = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    
    dx_raw = torch.empty((B, S, H, P), dtype=torch.float32, device=device)
    dz_raw = torch.empty((B, S, H, P), dtype=torch.float32, device=device)
    dB_rot = torch.zeros((B, S, H, N, R), dtype=torch.float32, device=device)
    dC_rot = torch.zeros((B, S, H, N, R), dtype=torch.float32, device=device)
    dt_grad = torch.zeros((B, S, H), dtype=torch.float32, device=device)
    A_grad = torch.zeros((B, S, H), dtype=torch.float32, device=device)
    lam_grad = torch.zeros((B, S, H), dtype=torch.float32, device=device)
    dssm_state = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    dtrap_mem = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    dW_X = torch.zeros((H, P, R), dtype=torch.float32, device=device)
    dW_Z = torch.zeros((H, P, R), dtype=torch.float32, device=device)
    dW_O = torch.zeros((H, P, R), dtype=torch.float32, device=device)
    dD_skip = torch.zeros((H, P), dtype=torch.float32, device=device)
    
    # Warmup / Objective/ Profiling Helper Functions
    def profile_config(fwd_kernel, bwd_kernel):
        for _ in range(3):
            fwd_kernel(x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip, out, curr_state, curr_V)
            bwd_kernel(
                grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                grad_curr_state, grad_curr_V,
                dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                dW_X, dW_Z, dW_O, dD_skip
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            fwd_kernel(x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip, out, curr_state, curr_V)
        end.record()
        torch.cuda.synchronize()
        fwd_ms = start.elapsed_time(end) / 10.0
        
        start.record()
        for _ in range(10):
            bwd_kernel(
                grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                grad_curr_state, grad_curr_V,
                dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                dW_X, dW_Z, dW_O, dD_skip
            )
        end.record()
        torch.cuda.synchronize()
        bwd_ms = start.elapsed_time(end) / 10.0
        return fwd_ms, bwd_ms

    def objective_fn(config) -> float:
        num_warps = config["num_warps"]
        layout_option = config["layout_option"]
        maxrregcount = config.get("maxrregcount", 255)
        try:
            fwd_k = get_cute_mamba3_fwd_kernel(
                num_warps=num_warps, P_val=P, N_val=N, R_val=R, layout_option=layout_option, maxrregcount=maxrregcount
            )
            bwd_k = get_cute_mamba3_bwd_kernel(
                num_warps=num_warps, P_val=P, N_val=N, R_val=R, layout_option=layout_option, maxrregcount=maxrregcount
            )
            fwd_ms, bwd_ms = profile_config(fwd_k, bwd_k)
            return fwd_ms + bwd_ms
        except Exception as e:
            import traceback
            traceback.print_exc()
            return float("inf")
            
    def is_valid_fn(config) -> bool:
        num_warps = config["num_warps"]
        layout_option = config["layout_option"]
        fwd_smem, bwd_smem = get_smem_requirements(num_warps, N, R, 1, 1, layout_option)
        return fwd_smem <= max_smem and bwd_smem <= max_smem
 
    param_space = {
        "num_warps": [4, 8],
        "layout_option": ["coprime", "multiple"],
        "maxrregcount": [128, 192, 255]
    }
    
    initial_config = {
        "num_warps": 4,
        "layout_option": "coprime",
        "maxrregcount": 255
    }
    
    autotuner = CoordinateDescentAutotuner(
        param_space=param_space,
        objective_fn=objective_fn,
        is_valid_fn=is_valid_fn,
        maximize=False,
        verbose=False
    )
    
    best_config, best_score = autotuner.tune(initial_config)
    return best_config, best_score

def sweep():
    device = "cuda"
    P, N, R = 64, 64, 4
    H = 4
    
    # Test different B and S combinations:
    workloads = [
        {"B": 2, "S": 64, "name": "Small (BS=128)"},
        {"B": 8, "S": 128, "name": "Medium (BS=1024)"},
        {"B": 32, "S": 64, "name": "Large (BS=2048)"},
        {"B": 64, "S": 128, "name": "X-Large (BS=8192)"}
    ]
    
    print("=================================================================")
    print(f"Sweeping Autotuner Configs over B x S (P={P}, N={N}, R={R})")
    print("=================================================================")
    for w in workloads:
        B, S = w["B"], w["S"]
        best_cfg, best_score = profile_shape_with_bs(B, S, H, P, N, R, device)
        print(f"Workload {w['name']:<18} (B={B:<2}, S={S:<3}): Best Config = {best_cfg}, Score = {best_score:.4f} ms")
    print("=================================================================")

if __name__ == "__main__":
    sweep()
