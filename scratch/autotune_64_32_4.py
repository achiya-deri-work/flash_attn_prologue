import os
import sys
import json
import torch
import time
import traceback
from typing import Dict, Any

sys.path.append("/home/achiya/vs_code_main/Advantage-First-RL")
sys.path.append("/home/achiya/vs_code_main/flash_attn_prologue")

import flash_attn_prologue
from temporal_mamba3.cute.cute_mamba3 import (
    get_cute_mamba3_fwd_kernel,
    get_cute_mamba3_bwd_kernel
)
from temporal_mamba3.cute.autotuner import (
    get_smem_requirements,
    profile_fwd,
    profile_bwd,
    CACHE_FILE as MAMBA3_CACHE_FILE
)

ADV_FIRST_CACHE_FILE = "/home/achiya/vs_code_main/Advantage-First-RL/advantage_first/layers/cute/autotune_cache.json"

def run_exhaustive_autotune():
    device = "cuda"
    P, N, R = 64, 32, 4
    B, S, H = 4, 64, 4
    
    props = torch.cuda.get_device_properties(device)
    max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    print(f"Max shared memory per block: {max_smem / 1024:.2f} KB")
    
    # Initialize 3D tensors for the new cute_mamba3 custom operators
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

    # Search space definition
    num_warps_list = [2, 4, 8]
    TILE_S_list = [1, 2, 4, 8, 16]
    buf_size_list = [1, 2]
    layout_option_list = ["coprime", "multiple"]
    maxrregcount_list = [80, 128, 168, 192, 255]
    copy_layout_shape_list = [(8, 4), (16, 2)]
    mma_shape_list = [(16, 8, 16)]

    print("Generating all candidate configurations...")
    configs = []
    for num_warps in num_warps_list:
        for TILE_S in TILE_S_list:
            for buf_size in buf_size_list:
                for layout_option in layout_option_list:
                    for maxrregcount in maxrregcount_list:
                        for copy_layout_shape in copy_layout_shape_list:
                            for mma_shape in mma_shape_list:
                                # Validation rules
                                if TILE_S == 1 and buf_size == 2:
                                    continue
                                fwd_smem, bwd_smem = get_smem_requirements(num_warps, N, R, TILE_S, buf_size, layout_option)
                                configs.append({
                                    "num_warps": num_warps,
                                    "TILE_S": TILE_S,
                                    "buf_size": buf_size,
                                    "layout_option": layout_option,
                                    "maxrregcount": maxrregcount,
                                    "copy_layout_shape": copy_layout_shape,
                                    "mma_shape": mma_shape,
                                    "fwd_smem": fwd_smem,
                                    "bwd_smem": bwd_smem
                                })

    print(f"Total candidate configurations: {len(configs)}")
    
    results_fwd = []
    results_bwd = []
    
    for idx, config in enumerate(configs):
        print(f"[{idx+1}/{len(configs)}] Profiling: warps={config['num_warps']}, TILE_S={config['TILE_S']}, buf={config['buf_size']}, layout={config['layout_option']}, reg={config['maxrregcount']}, copy={config['copy_layout_shape']}")
        
        # Profile FWD
        if config["fwd_smem"] <= max_smem:
            try:
                torch.cuda.empty_cache()
                fwd_k = get_cute_mamba3_fwd_kernel(
                    num_warps=config['num_warps'], P_val=P, N_val=N, R_val=R,
                    TILE_S=config['TILE_S'], buf_size=config['buf_size'],
                    layout_option=config['layout_option'], maxrregcount=config['maxrregcount'],
                    copy_layout_shape=config['copy_layout_shape'], mma_shape=config['mma_shape']
                )
                fwd_ms = profile_fwd(
                    fwd_k, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                    W_X, W_Z, W_O, D_skip, out, curr_state, curr_V
                )
                results_fwd.append((fwd_ms, config))
                print(f"      FWD Success -> {fwd_ms:.4f} ms")
            except Exception as e:
                pass

        # Profile BWD
        if config["bwd_smem"] <= max_smem:
            try:
                torch.cuda.empty_cache()
                bwd_k = get_cute_mamba3_bwd_kernel(
                    num_warps=config['num_warps'], P_val=P, N_val=N, R_val=R,
                    TILE_S=config['TILE_S'], buf_size=config['buf_size'],
                    layout_option=config['layout_option'], maxrregcount=config['maxrregcount'],
                    copy_layout_shape=config['copy_layout_shape'], mma_shape=config['mma_shape']
                )
                bwd_ms = profile_bwd(
                    bwd_k,
                    grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                    grad_curr_state, grad_curr_V,
                    dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                    dW_X, dW_Z, dW_O, dD_skip
                )
                results_bwd.append((bwd_ms, config))
                print(f"      BWD Success -> {bwd_ms:.4f} ms")
            except Exception as e:
                pass

    results_fwd.sort(key=lambda x: x[0])
    results_bwd.sort(key=lambda x: x[0])
    
    print("\n" + "="*60)
    print("TOP 5 FWD CONFIGURATIONS:")
    print("="*60)
    for i, (fwd_ms, config) in enumerate(results_fwd[:5]):
        print(f"{i+1}. FWD: {fwd_ms:.4f} ms")
        print(f"   warps={config['num_warps']}, TILE_S={config['TILE_S']}, buf_size={config['buf_size']}, layout={config['layout_option']}, maxrregcount={config['maxrregcount']}, copy={config['copy_layout_shape']}")
    
    print("\n" + "="*60)
    print("TOP 5 BWD CONFIGURATIONS:")
    print("="*60)
    for i, (bwd_ms, config) in enumerate(results_bwd[:5]):
        print(f"{i+1}. BWD: {bwd_ms:.4f} ms")
        print(f"   warps={config['num_warps']}, TILE_S={config['TILE_S']}, buf_size={config['buf_size']}, layout={config['layout_option']}, maxrregcount={config['maxrregcount']}, copy={config['copy_layout_shape']}")
    print("="*60)

    if not results_fwd or not results_bwd:
        print("Error: Could not find valid FWD and BWD configurations!")
        return

    best_fwd_ms, best_fwd_config = results_fwd[0]
    best_bwd_ms, best_bwd_config = results_bwd[0]
    
    save_cfg_fwd = {
        "num_warps": best_fwd_config["num_warps"],
        "TILE_S": best_fwd_config["TILE_S"],
        "buf_size": best_fwd_config["buf_size"],
        "layout_option": best_fwd_config["layout_option"],
        "maxrregcount": best_fwd_config["maxrregcount"],
        "copy_layout_shape": best_fwd_config["copy_layout_shape"],
        "mma_shape": best_fwd_config["mma_shape"]
    }
    
    save_cfg_bwd = {
        "num_warps": best_bwd_config["num_warps"],
        "TILE_S": best_bwd_config["TILE_S"],
        "buf_size": best_bwd_config["buf_size"],
        "layout_option": best_bwd_config["layout_option"],
        "maxrregcount": best_bwd_config["maxrregcount"],
        "copy_layout_shape": best_bwd_config["copy_layout_shape"],
        "mma_shape": best_bwd_config["mma_shape"]
    }
    
    # 1. Update temporal_mamba3 cache
    for path in [MAMBA3_CACHE_FILE, ADV_FIRST_CACHE_FILE]:
        if os.path.exists(path):
            with open(path, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        cache["64_32_4_fwd"] = save_cfg_fwd
        cache["64_32_4_bwd"] = save_cfg_bwd
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Saved best configs to {path}")

if __name__ == "__main__":
    run_exhaustive_autotune()
