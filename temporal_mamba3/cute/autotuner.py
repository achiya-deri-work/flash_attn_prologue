import os
import json
import torch
from typing import Dict, Any, Optional
from CuTe_DSL_CONTEXT.coordinate_descent import CoordinateDescentAutotuner

CACHE_FILE = os.path.join(os.path.dirname(__file__), "autotune_cache.json")

_autotune_in_memory_cache = {}

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            for k, val in cache.items():
                if isinstance(val, dict):
                    if "copy_layout_shape" in val and isinstance(val["copy_layout_shape"], list):
                        val["copy_layout_shape"] = tuple(val["copy_layout_shape"])
                    if "mma_shape" in val and isinstance(val["mma_shape"], list):
                        val["mma_shape"] = tuple(val["mma_shape"])
            return cache
        except Exception:
            pass
    return {}

def save_cache(cache: Dict[str, Any]):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception:
        pass

def align_up(x: int) -> int:
    # 128-byte alignment for Float32 = 32 elements
    return ((x + 31) // 32) * 32

def get_smem_requirements(num_warps: int, N: int, R: int, TILE_S: int, buf_size: int, layout_option: str = "coprime"):
    # FWD: sState, sTrap, sC, sC_out
    smem_layout_size = buf_size * num_warps * N * 20
    if layout_option == "multiple":
        sC_size = num_warps * 10 * N
        sC_out_size = num_warps * 160
    else:
        sC_size = num_warps * 9 * N
        sC_out_size = num_warps * 136
    
    fwd_smem = (
        align_up(smem_layout_size) +  # sState
        align_up(smem_layout_size) +  # sTrap
        align_up(sC_size) +           # sC
        align_up(sC_out_size)         # sC_out
    ) * 4
    
    # BWD: sState, sTrap, sGradState, sGradV
    bwd_smem = (4 * align_up(smem_layout_size)) * 4
    
    return fwd_smem, bwd_smem

def profile_fwd(fwd_kernel, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip, out, curr_state, curr_V):
    # Warmup
    for _ in range(3):
        fwd_kernel(x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip, out, curr_state, curr_V)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Profile
    start_event.record()
    for _ in range(10):
        fwd_kernel(x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip, out, curr_state, curr_V)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 10.0

def profile_bwd(
    bwd_kernel,
    grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
    grad_curr_state, grad_curr_V,
    dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
    dW_X, dW_Z, dW_O, dD_skip
):
    # Warmup
    for _ in range(3):
        bwd_kernel(
            grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
            grad_curr_state, grad_curr_V,
            dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
            dW_X, dW_Z, dW_O, dD_skip
        )
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Profile
    start_event.record()
    for _ in range(10):
        bwd_kernel(
            grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
            grad_curr_state, grad_curr_V,
            dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
            dW_X, dW_Z, dW_O, dD_skip
        )
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 10.0

def autotune_shape(P: int, N: int, R: int, mode: str, device: str) -> Dict[str, Any]:
    # Lazy imports to avoid circular reference
    from advantage_first.layers.cute.cute_mamba3 import (
        get_cute_mamba3_fwd_kernel,
        get_cute_mamba3_bwd_kernel
    )
    
    props = torch.cuda.get_device_properties(device)
    max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    
    # Setup small tensors for profiling
    B, S, H = 2, 64, 4
    x_raw = torch.randn(B, S, H * P, device=device, dtype=torch.float32)
    z_raw = torch.randn(B, S, H * P, device=device, dtype=torch.float32)
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
    
    out = torch.empty((B, S, H * P), dtype=torch.bfloat16, device=device)
    curr_state = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    curr_V = torch.empty((B, S, H, N, P), dtype=torch.float32, device=device)
    
    grad_out = torch.randn(B, S, H * P, device=device, dtype=torch.float32)
    grad_curr_state = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    grad_curr_V = torch.randn(B, S, H, N, P, device=device, dtype=torch.float32)
    
    dx_raw = torch.empty((B, S, H * P), dtype=torch.float32, device=device)
    dz_raw = torch.empty((B, S, H * P), dtype=torch.float32, device=device)
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
    
    # Objective function
    def objective_fn(config: Dict[str, Any]) -> float:
        num_warps = config["num_warps"]
        TILE_S = config["TILE_S"]
        buf_size = config["buf_size"]
        layout_option = config["layout_option"]
        maxrregcount = config.get("maxrregcount", 255)
        copy_layout_shape = config.get("copy_layout_shape", (8, 4))
        mma_shape = config.get("mma_shape", (16, 8, 16))
        
        try:
            if mode == "fwd":
                fwd_k = get_cute_mamba3_fwd_kernel(
                    num_warps=num_warps, P_val=P, N_val=N, R_val=R, TILE_S=TILE_S, buf_size=buf_size, layout_option=layout_option, maxrregcount=maxrregcount,
                    copy_layout_shape=copy_layout_shape, mma_shape=mma_shape
                )
                return profile_fwd(
                    fwd_k, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                    W_X, W_Z, W_O, D_skip, out, curr_state, curr_V
                )
            else:
                bwd_k = get_cute_mamba3_bwd_kernel(
                    num_warps=num_warps, P_val=P, N_val=N, R_val=R, TILE_S=TILE_S, buf_size=buf_size, layout_option=layout_option, maxrregcount=maxrregcount,
                    copy_layout_shape=copy_layout_shape, mma_shape=mma_shape
                )
                return profile_bwd(
                    bwd_k,
                    grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                    grad_curr_state, grad_curr_V,
                    dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                    dW_X, dW_Z, dW_O, dD_skip
                )
        except Exception:
            return float("inf")
            
    # Validator function
    def is_valid_fn(config: Dict[str, Any]) -> bool:
        num_warps = config["num_warps"]
        TILE_S = config["TILE_S"]
        buf_size = config["buf_size"]
        layout_option = config["layout_option"]
        
        if TILE_S == 1 and buf_size == 2:
            return False
            
        fwd_smem, bwd_smem = get_smem_requirements(num_warps, N, R, TILE_S, buf_size, layout_option)
        if mode == "fwd":
            return fwd_smem <= max_smem
        else:
            return bwd_smem <= max_smem

    # Search space
    param_space = {
        "num_warps": [4, 8],
        "TILE_S": [1, 2, 4, 8],
        "buf_size": [1, 2],
        "layout_option": ["coprime", "multiple"],
        "maxrregcount": [128, 192, 255],
        "copy_layout_shape": [(8, 4), (16, 2)],
        "mma_shape": [(16, 8, 16)]
    }
    
    # Safe starting configuration
    initial_config = {
        "num_warps": 4,
        "TILE_S": 1,
        "buf_size": 1,
        "layout_option": "coprime",
        "maxrregcount": 255,
        "copy_layout_shape": (8, 4),
        "mma_shape": (16, 8, 16)
    }
    
    autotuner = CoordinateDescentAutotuner(
        param_space=param_space,
        objective_fn=objective_fn,
        is_valid_fn=is_valid_fn,
        maximize=False,
        verbose=True
    )
    
    try:
        best_config, best_score = autotuner.tune(initial_config)
        return best_config
    except ValueError as e:
        if autotuner.verbose:
            print(f"[CoordinateDescentAutotuner] No valid configuration found for shape P={P}, N={N}, R={R}, mode={mode} due to hardware constraints. Falling back. Error: {e}")
        return {"fallback": True}

def get_autotuned_config(P: int, N: int, R: int, mode: str, device: str) -> Dict[str, Any]:
    env_warps = os.environ.get("MAMBA3_WARPS")
    if env_warps:
        # Override via environment variable
        w = int(env_warps)
        t_s = 8 if (N <= 16 or (N == 32 and R == 2)) else 1
        return {"num_warps": w, "TILE_S": t_s, "buf_size": 1, "layout_option": "coprime", "maxrregcount": 255, "copy_layout_shape": (8, 4), "mma_shape": (16, 8, 16)}
        
    key = f"{P}_{N}_{R}_{mode}"
    if key in _autotune_in_memory_cache:
        return _autotune_in_memory_cache[key]
        
    cache = load_cache()
    if key in cache:
        _autotune_in_memory_cache[key] = cache[key]
        return cache[key]
        
    # Profile shape dynamically using coordinate descent
    best_config = autotune_shape(P, N, R, mode, device)
    
    # Save cache
    cache[key] = best_config
    save_cache(cache)
    _autotune_in_memory_cache[key] = best_config
    return best_config
