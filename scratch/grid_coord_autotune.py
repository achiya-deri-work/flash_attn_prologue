import os
import sys
import json
import torch
import time

sys.path.append("/home/achiya/vs_code_main/Advantage-First-RL")
sys.path.append("/home/achiya/vs_code_main/flash_attn_prologue")

# Import first to initialize CUTE_DSL_LIBS
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

def run_grid_coord_tuning():
    device = "cuda"
    P, N, R = 32, 64, 4
    # User's exact batch, tokens, heads:
    B, S, H = 16, 64, 4
    
    props = torch.cuda.get_device_properties(device)
    max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    print(f"Max shared memory per block: {max_smem / 1024:.2f} KB")
    
    # Initialize tensors
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

    # Search space dimensions
    num_warps_list = [2, 4, 8]
    TILE_S_list = [1, 2, 4, 8]
    buf_size_list = [1, 2]
    layout_option_list = ["coprime", "multiple"]
    maxrregcount_list = [128, 192, 255]
    copy_layout_shape_list = [(8, 4), (16, 2)]
    mma_shape_list = [(16, 8, 16)]

    # Generate configurations
    configs = []
    for num_warps in num_warps_list:
        for TILE_S in TILE_S_list:
            for buf_size in buf_size_list:
                for layout_option in layout_option_list:
                    for maxrregcount in maxrregcount_list:
                        for copy_layout_shape in copy_layout_shape_list:
                            for mma_shape in mma_shape_list:
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

    # Profile FWD Grid Search
    print("\n--- STARTING FWD GRID SEARCH ---")
    fwd_candidates = [c for c in configs if c["fwd_smem"] <= max_smem]
    print(f"Total compilable FWD configurations: {len(fwd_candidates)}")
    fwd_results = []
    for idx, c in enumerate(fwd_candidates):
        try:
            torch.cuda.empty_cache()
            fwd_k = get_cute_mamba3_fwd_kernel(
                num_warps=c['num_warps'], P_val=P, N_val=N, R_val=R,
                TILE_S=c['TILE_S'], buf_size=c['buf_size'],
                layout_option=c['layout_option'], maxrregcount=c['maxrregcount'],
                copy_layout_shape=c['copy_layout_shape'], mma_shape=c['mma_shape']
            )
            ms = profile_fwd(
                fwd_k, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                W_X, W_Z, W_O, D_skip, out, curr_state, curr_V
            )
            fwd_results.append((ms, c))
            print(f"[{idx+1}/{len(fwd_candidates)}] warps={c['num_warps']}, TILE_S={c['TILE_S']}, buf={c['buf_size']}, layout={c['layout_option']}, reg={c['maxrregcount']}, copy={c['copy_layout_shape']} -> {ms:.4f} ms")
        except Exception:
            pass

    # Profile BWD Grid Search
    print("\n--- STARTING BWD GRID SEARCH ---")
    bwd_candidates = [c for c in configs if c["bwd_smem"] <= max_smem]
    print(f"Total compilable BWD configurations: {len(bwd_candidates)}")
    bwd_results = []
    for idx, c in enumerate(bwd_candidates):
        try:
            torch.cuda.empty_cache()
            bwd_k = get_cute_mamba3_bwd_kernel(
                num_warps=c['num_warps'], P_val=P, N_val=N, R_val=R,
                TILE_S=c['TILE_S'], buf_size=c['buf_size'],
                layout_option=c['layout_option'], maxrregcount=c['maxrregcount'],
                copy_layout_shape=c['copy_layout_shape'], mma_shape=c['mma_shape']
            )
            ms = profile_bwd(
                bwd_k,
                grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                grad_curr_state, grad_curr_V,
                dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                dW_X, dW_Z, dW_O, dD_skip
            )
            bwd_results.append((ms, c))
            print(f"[{idx+1}/{len(bwd_candidates)}] warps={c['num_warps']}, TILE_S={c['TILE_S']}, buf={c['buf_size']}, layout={c['layout_option']}, reg={c['maxrregcount']}, copy={c['copy_layout_shape']} -> {ms:.4f} ms")
        except Exception:
            pass

    fwd_results.sort(key=lambda x: x[0])
    bwd_results.sort(key=lambda x: x[0])

    if not fwd_results or not bwd_results:
        print("Error: No valid FWD or BWD configurations successfully compiled!")
        return

    best_fwd_ms, best_fwd_config = fwd_results[0]
    best_bwd_ms, best_bwd_config = bwd_results[0]
    
    print("\n" + "="*60)
    print(f"GRID SEARCH WINNER FWD: {best_fwd_ms:.4f} ms -> {best_fwd_config}")
    print(f"GRID SEARCH WINNER BWD: {best_bwd_ms:.4f} ms -> {best_bwd_config}")
    print("="*60)

    # Perform Coordinate Descent on FWD
    print("\n--- COORDINATE DESCENT FWD REFINE ---")
    current_fwd = best_fwd_config.copy()
    current_fwd_score = best_fwd_ms
    
    keys_to_tune = ["num_warps", "TILE_S", "buf_size", "layout_option", "maxrregcount", "copy_layout_shape"]
    space = {
        "num_warps": num_warps_list,
        "TILE_S": TILE_S_list,
        "buf_size": buf_size_list,
        "layout_option": layout_option_list,
        "maxrregcount": maxrregcount_list,
        "copy_layout_shape": copy_layout_shape_list
    }
    
    improved = True
    while improved:
        improved = False
        for key in keys_to_tune:
            for val in space[key]:
                if val == current_fwd[key]:
                    continue
                test_cfg = current_fwd.copy()
                test_cfg[key] = val
                
                # Verify SMEM
                fwd_smem, _ = get_smem_requirements(test_cfg["num_warps"], N, R, test_cfg["TILE_S"], test_cfg["buf_size"], test_cfg["layout_option"])
                if fwd_smem > max_smem or (test_cfg["TILE_S"] == 1 and test_cfg["buf_size"] == 2):
                    continue
                    
                try:
                    torch.cuda.empty_cache()
                    test_k = get_cute_mamba3_fwd_kernel(
                        num_warps=test_cfg['num_warps'], P_val=P, N_val=N, R_val=R,
                        TILE_S=test_cfg['TILE_S'], buf_size=test_cfg['buf_size'],
                        layout_option=test_cfg['layout_option'], maxrregcount=test_cfg['maxrregcount'],
                        copy_layout_shape=test_cfg['copy_layout_shape'], mma_shape=test_cfg['mma_shape']
                    )
                    score = profile_fwd(
                        test_k, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                        W_X, W_Z, W_O, D_skip, out, curr_state, curr_V
                    )
                    if score < current_fwd_score:
                        print(f"      FWD improved: {key} -> {val} (score {score:.4f} ms vs {current_fwd_score:.4f} ms)")
                        current_fwd_score = score
                        current_fwd = test_cfg
                        improved = True
                except Exception:
                    pass

    # Perform Coordinate Descent on BWD
    print("\n--- COORDINATE DESCENT BWD REFINE ---")
    current_bwd = best_bwd_config.copy()
    current_bwd_score = best_bwd_ms
    
    improved = True
    while improved:
        improved = False
        for key in keys_to_tune:
            for val in space[key]:
                if val == current_bwd[key]:
                    continue
                test_cfg = current_bwd.copy()
                test_cfg[key] = val
                
                # Verify SMEM
                _, bwd_smem = get_smem_requirements(test_cfg["num_warps"], N, R, test_cfg["TILE_S"], test_cfg["buf_size"], test_cfg["layout_option"])
                if bwd_smem > max_smem or (test_cfg["TILE_S"] == 1 and test_cfg["buf_size"] == 2):
                    continue
                    
                try:
                    torch.cuda.empty_cache()
                    test_k = get_cute_mamba3_bwd_kernel(
                        num_warps=test_cfg['num_warps'], P_val=P, N_val=N, R_val=R,
                        TILE_S=test_cfg['TILE_S'], buf_size=test_cfg['buf_size'],
                        layout_option=test_cfg['layout_option'], maxrregcount=test_cfg['maxrregcount'],
                        copy_layout_shape=test_cfg['copy_layout_shape'], mma_shape=test_cfg['mma_shape']
                    )
                    score = profile_bwd(
                        test_k,
                        grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
                        grad_curr_state, grad_curr_V,
                        dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
                        dW_X, dW_Z, dW_O, dD_skip
                    )
                    if score < current_bwd_score:
                        print(f"      BWD improved: {key} -> {val} (score {score:.4f} ms vs {current_bwd_score:.4f} ms)")
                        current_bwd_score = score
                        current_bwd = test_cfg
                        improved = True
                except Exception:
                    pass

    print("\n" + "="*60)
    print(f"FINAL OPTIMAL CONFIG FWD: {current_fwd_score:.4f} ms -> {current_fwd}")
    print(f"FINAL OPTIMAL CONFIG BWD: {current_bwd_score:.4f} ms -> {current_bwd}")
    print("="*60)

    # Save to caches
    save_cfg_fwd = {
        "num_warps": current_fwd["num_warps"],
        "TILE_S": current_fwd["TILE_S"],
        "buf_size": current_fwd["buf_size"],
        "layout_option": current_fwd["layout_option"],
        "maxrregcount": current_fwd["maxrregcount"],
        "copy_layout_shape": current_fwd["copy_layout_shape"],
        "mma_shape": current_fwd["mma_shape"]
    }
    
    save_cfg_bwd = {
        "num_warps": current_bwd["num_warps"],
        "TILE_S": current_bwd["TILE_S"],
        "buf_size": current_bwd["buf_size"],
        "layout_option": current_bwd["layout_option"],
        "maxrregcount": current_bwd["maxrregcount"],
        "copy_layout_shape": current_bwd["copy_layout_shape"],
        "mma_shape": current_bwd["mma_shape"]
    }
    
    for path in [MAMBA3_CACHE_FILE, ADV_FIRST_CACHE_FILE]:
        if os.path.exists(path):
            with open(path, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        cache[f"{P}_{N}_{R}_fwd"] = save_cfg_fwd
        cache[f"{P}_{N}_{R}_bwd"] = save_cfg_bwd
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Saved best configs to {path}")

if __name__ == "__main__":
    run_grid_coord_tuning()
