import os
import json
import torch
import math
from typing import Dict, Any, Optional, List

CACHE_FILE = os.path.join(os.path.dirname(__file__), "autotune_cache.json")

_autotune_in_memory_cache = {}

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache: Dict[str, Any]):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception:
        pass

def get_valid_warp_layouts(warps):
    layouts = []
    # Find all factorizations of warps = m * n (with k=1 to avoid runtime illegal memory access)
    for m in range(1, warps + 1):
        if warps % m != 0:
            continue
        n = warps // m
        layouts.append((m, n, 1))
    return layouts


def is_config_valid(config: Dict[str, Any], mode: str) -> bool:
    # 1. Base warp layouts extraction
    num_warps = config.get("num_warps", 4)
    num_threads = num_warps * 32
    
    # 2. Extract layouts
    w_layout = config.get("atom_layout_mnk", None)
    if w_layout is not None:
        if isinstance(w_layout, list):
            w_layout = tuple(w_layout)
        m, n, k = w_layout
    else:
        bm = config.get("BLOCK_M", 64)
        if mode == "fused_qkv" and bm == 16:
            if num_warps == 4:
                m, n, k = 1, 4, 1
            elif num_warps == 2:
                m, n, k = 1, 2, 1
            elif num_warps == 6:
                m, n, k = 1, 6, 1
            elif num_warps == 8:
                m, n, k = 1, 8, 1
            else:
                m, n, k = 1, num_warps, 1
        else:
            if num_warps == 4:
                m, n, k = 2, 2, 1
            elif num_warps == 2:
                m, n, k = 2, 1, 1
            elif num_warps == 6:
                m, n, k = 3, 2, 1
            elif num_warps == 8:
                m, n, k = 4, 2, 1
            elif num_warps == 12:
                m, n, k = 4, 3, 1
            elif num_warps == 16:
                m, n, k = 4, 4, 1
            else:
                m, n, k = num_warps, 1, 1
            
    # 3. K-dimension layouts matching constraints (k=1)
    if k > 1:
        return False
        
    # 4. Divisibility check for warp layout sizes
    # BLOCK_M (or 3 * BLOCK_M) must be a multiple of m * 16 (MMA shape M)
    bm = config.get("BLOCK_M", 64)
    val_m = 3 * bm if mode == "bwd_dw" else bm
    if val_m % (m * 16) != 0:
        return False
        
    # BLOCK_N must be a multiple of n * 8 (MMA shape N)
    if mode in ("bwd_dx", "bwd_dw", "fused_qkv", "fused"):
        if mode in ("fused_qkv", "fused"):
            bn = config.get("BLOCK_N", config.get("HEAD_GROUP", 2) * 3 * 64)
        else:
            bn = config.get("BLOCK_N", 64)
        if bn % (n * 8) != 0:
            return False
            
    # 5. Global store boundary constraints for tiled copy (for bwd_dx)
    if mode == "bwd_dx":
        bn = config.get("BLOCK_N", 64)
        store_thr_m = min(16, bm)
        store_thr_n = num_threads // store_thr_m
        if store_thr_n * 8 > bn:
            return False
        # K-Dimension Layout Matching: if n > 1, BLOCK_K must be 16
        bk = config.get("BLOCK_K", 16)
        if n > 1 and bk != 16:
            return False
            
    # 6. Thread occupancy constraints for bwd_dw
    if mode == "bwd_dw":
        bn = config.get("BLOCK_N", 64)
        threads_col = min(bn, num_threads)
        if num_threads % threads_col != 0 or bn % threads_col != 0:
            return False
        threads_row = num_threads // threads_col
        if val_m % threads_row != 0:
            return False
        # Global store boundary constraints for tiled copy
        store_thr_m = min(16, bm)
        store_thr_n = num_threads // store_thr_m
        if store_thr_n * 8 > bn:
            return False
            
    # 7. Additional constraints for epilogue mode
    if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
        hg = config.get("HEAD_GROUP", 1)
        stage = config.get("ab_stage", 1)
        if mode == "epilogue_bwd":
            required_threads = bm * hg
        elif mode == "epilogue_fwd":
            required_threads = (bm // stage) * hg
        else:
            required_threads = max((bm // stage) * hg, bm * hg)
        if num_threads < required_threads:
            return False
            
    # 8. Thread occupancy constraints for fused_qkv
    if mode in ("fused_qkv", "fused"):
        bm = config.get("BLOCK_M", 64)
        hg = config.get("HEAD_GROUP", 1)
        if num_threads < bm * hg:
            return False
            
    return True


def get_b_n_seq(total_elements):
    # Find N_seq as the largest divisor of total_elements in [512, 256, 128, 64, 32, 16]
    for n_seq in [512, 256, 128, 64, 32, 16]:
        if total_elements % n_seq == 0:
            return total_elements // n_seq, n_seq
    # Fallback
    return 1, total_elements

def get_smem_bytes(mode, bm, bn, bk, hg, stage, pad=0, K=64):
    if mode == "fused_qkv":
        block_n = hg * 3 * 64
        mainloop_bytes = (stage * bm * bk * 2) + (stage * block_n * bk * 2)
        epilogue_bytes = bm * (block_n + 8) * 4
        return max(mainloop_bytes, epilogue_bytes)
    elif mode == "bwd_dx":
        sX_bytes = stage * bm * bk * 3 * 2
        sW_bytes = stage * bn * (bk + pad) * 3 * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = bm * (bn + 8) * 2
        return max(mainloop_bytes, epilogue_bytes)
    elif mode == "bwd_dw":
        sX_bytes = stage * bm * bk * 3 * 2
        sW_bytes = stage * bn * bk * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = 3 * bm * (bn + 8) * 2
        return max(mainloop_bytes, epilogue_bytes)
    elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
        if mode == "epilogue_bwd":
            s_rmsnorm_reduction_elems = bm * 2 * hg * K
            s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
            total_elems = s_scales_offset + bm * hg
            flat_smem_elements = ((total_elems + 31) // 32) * 32
            return flat_smem_elements * 4
        else:
            block_n = hg * K
            block_n_padded = block_n + 8
            smem_bytes = 2 * (bm * block_n_padded * 2)
            if mode == "epilogue":
                s_rmsnorm_reduction_elems = bm * 2 * hg * K
                s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                total_elems = s_scales_offset + bm * hg
                flat_smem_elements = ((total_elems + 31) // 32) * 32
                bwd_smem_bytes = flat_smem_elements * 4
                smem_bytes = max(smem_bytes, bwd_smem_bytes)
            return smem_bytes
    return 0

def profile_shapes(M: int, N: int, K: int, is_fused: Any, device: str, initial_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    if isinstance(is_fused, bool):
        mode = "fused" if is_fused else "gemm"
    else:
        mode = is_fused

    prof_M = min(M, 16384)
    prof_N = N
    prof_K = K

    # Setup reference data and outputs for correctness validation
    ref_data = {}
    from cutlass import BFloat16, Float32
    from quack.compile_utils import make_fake_tensor
    
    if mode == "fused_qkv":
        num_heads = N // 3 // 64
        head_dim = 64
        hidden = K
        B_prof, seq_prof = get_b_n_seq(prof_M)
        
        x_prof = torch.randn(B_prof, seq_prof, hidden, device=device, dtype=torch.bfloat16) * 0.1
        weight_prof = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=torch.bfloat16) * 0.1
        
        num_heads_x_dim = num_heads * head_dim
        weight_q_prof = weight_prof[0 : num_heads_x_dim, :].clone()
        weight_k_prof = weight_prof[num_heads_x_dim : 2 * num_heads_x_dim, :].clone()
        weight_v_prof = weight_prof[2 * num_heads_x_dim : 3 * num_heads_x_dim, :].clone()
        
        q_ref = (x_prof.view(-1, hidden) @ weight_q_prof.t()).view(B_prof, seq_prof, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k_ref = (x_prof.view(-1, hidden) @ weight_k_prof.t()).view(B_prof, seq_prof, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v_ref = (x_prof.view(-1, hidden) @ weight_v_prof.t()).view(B_prof, seq_prof, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        
        ref_data = {
            "x_prof": x_prof, "weight_prof": weight_prof,
            "weight_q_prof": weight_q_prof, "weight_k_prof": weight_k_prof, "weight_v_prof": weight_v_prof,
            "q_ref": q_ref, "k_ref": k_ref, "v_ref": v_ref,
            "num_heads": num_heads, "head_dim": head_dim, "hidden": hidden,
            "B_prof": B_prof, "seq_prof": seq_prof
        }
    elif mode == "bwd_dx":
        B_prof, N_seq_prof = get_b_n_seq(prof_M)
        num_heads = prof_K // 3 // 64
        head_dim = 64
        grad_q_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        grad_k_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        grad_v_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        weight_prof = torch.randn(prof_K, prof_N, device=device, dtype=torch.bfloat16) * 0.1
        
        grad_qkv = torch.empty((B_prof * N_seq_prof, 3 * num_heads * head_dim), device=device, dtype=torch.bfloat16)
        grad_q_flat = grad_q_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_k_flat = grad_k_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_v_flat = grad_v_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_qkv[:, 0 : num_heads * head_dim] = grad_q_flat
        grad_qkv[:, num_heads * head_dim : 2 * num_heads * head_dim] = grad_k_flat
        grad_qkv[:, 2 * num_heads * head_dim : 3 * num_heads * head_dim] = grad_v_flat
        
        ref_grad_x = grad_qkv @ weight_prof
        
        ref_data = {
            "grad_q_prof": grad_q_prof, "grad_k_prof": grad_k_prof, "grad_v_prof": grad_v_prof,
            "weight_prof": weight_prof, "ref_grad_x": ref_grad_x,
            "B_prof": B_prof, "N_seq_prof": N_seq_prof,
            "num_heads": num_heads
        }
    elif mode == "bwd_dw":
        B_prof, N_seq_prof = get_b_n_seq(prof_K)
        num_heads = prof_M // 3 // 64
        head_dim = 64
        grad_q_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        grad_k_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        grad_v_prof = torch.randn(B_prof, num_heads, N_seq_prof, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        x_prof = torch.randn(prof_K, prof_N, device=device, dtype=torch.bfloat16) * 0.1
        
        grad_qkv = torch.empty((B_prof * N_seq_prof, 3 * num_heads * head_dim), device=device, dtype=torch.bfloat16)
        grad_q_flat = grad_q_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_k_flat = grad_k_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_v_flat = grad_v_prof.permute(0, 2, 1, 3).reshape(B_prof * N_seq_prof, num_heads * head_dim)
        grad_qkv[:, 0 : num_heads * head_dim] = grad_q_flat
        grad_qkv[:, num_heads * head_dim : 2 * num_heads * head_dim] = grad_k_flat
        grad_qkv[:, 2 * num_heads * head_dim : 3 * num_heads * head_dim] = grad_v_flat
        
        ref_grad_weight = grad_qkv.t() @ x_prof
        
        ref_data = {
            "grad_q_prof": grad_q_prof, "grad_k_prof": grad_k_prof, "grad_v_prof": grad_v_prof,
            "x_prof": x_prof, "ref_grad_weight": ref_grad_weight,
            "B_prof": B_prof, "N_seq_prof": N_seq_prof,
            "num_heads": num_heads
        }

    # Inner profile helper with build & runtime try-catches and correctness validation
    def profile_and_verify(config):
        bm = config["BLOCK_M"]
        if bm > M:
            return float("inf")
        bn = config.get("BLOCK_N", config.get("HEAD_GROUP", 1) * 3 * 64)
        if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            if config.get("HEAD_GROUP", 1) > N:
                return float("inf")
        else:
            if bn > N:
                return float("inf")
        
        # Check divisibility of dimensions
        if mode == "fused":
            if prof_M % bm != 0:
                return float("inf")
        elif mode == "fused_qkv":
            if prof_M % bm != 0:
                return float("inf")
        elif mode == "bwd_dx":
            if prof_M % bm != 0 or prof_N % bn != 0:
                return float("inf")
        elif mode == "bwd_dw":
            if (prof_M // 3) % bm != 0 or prof_N % bn != 0:
                return float("inf")
        elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            if prof_M % bm != 0:
                return float("inf")
                
        try:
            if mode == "fused":
                from archive.flash_attn_prologue.fused_pipelined_kernel import get_fused_pipelined_fwd_kernel
                hg = config["HEAD_GROUP"]
                if prof_M % bm != 0:
                    return float("inf")
                
                num_heads = 16
                head_dim = 64
                hidden = K
                
                fwd_kernel = get_fused_pipelined_fwd_kernel(
                    num_warps=4,
                    BLOCK_M=bm,
                    BLOCK_K_STAGE=config["BLOCK_K"],
                    NUM_HEADS=num_heads,
                    HEAD_DIM=head_dim,
                    HEAD_GROUP=hg,
                    hidden=hidden
                )
                
                B_prof = 2
                seq_prof = prof_M // 2
                x_prof = torch.randn(B_prof, seq_prof, hidden, device=device, dtype=torch.bfloat16)
                weight_raw = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=torch.bfloat16)
                w_reshaped = weight_raw.view(3, num_heads, head_dim, hidden)
                w_split = w_reshaped.view(3, num_heads // hg, hg, head_dim, hidden)
                w_permuted = w_split.permute(1, 2, 0, 3, 4)
                weight_prof = w_permuted.reshape(num_heads * 3 * head_dim, hidden).contiguous()
                
                rmsnorm_weight_prof = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
                scales_prof = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)
                sin_prof = torch.randn(B_prof, 1, seq_prof, head_dim // 2, device=device, dtype=torch.float32)
                cos_prof = torch.randn(B_prof, 1, seq_prof, head_dim // 2, device=device, dtype=torch.float32)
                
                q_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                k_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                v_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                
                for _ in range(3):
                    fwd_kernel(x_prof, weight_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof, q_out, k_out, v_out)
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    fwd_kernel(x_prof, weight_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof, q_out, k_out, v_out)
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event) / 15.0

            elif mode == "fused_qkv":
                from flash_attn_prologue.fused_qkv import get_fused_qkv_kernel
                fwd_kernel = get_fused_qkv_kernel(
                    num_warps=config["num_warps"],
                    BLOCK_M=config["BLOCK_M"],
                    BLOCK_K_STAGE=config["BLOCK_K"],
                    NUM_HEADS=ref_data["num_heads"],
                    HEAD_DIM=ref_data["head_dim"],
                    HEAD_GROUP=config["HEAD_GROUP"],
                    hidden=ref_data["hidden"],
                    use_sync_in_loop=True,
                    ab_stage=config["ab_stage"],
                    maxrregcount=config["maxrregcount"],
                    swizzle_x=config.get("swizzle_x", 8),
                    atom_layout_mnk=config.get("atom_layout_mnk", None)
                )
                q_out = torch.empty_like(ref_data["q_ref"])
                k_out = torch.empty_like(ref_data["k_ref"])
                v_out = torch.empty_like(ref_data["v_ref"])
                
                # Warmup and correctness check
                fwd_kernel(ref_data["x_prof"], ref_data["weight_q_prof"], ref_data["weight_k_prof"], ref_data["weight_v_prof"], q_out, k_out, v_out)
                
                diff_q = (q_out - ref_data["q_ref"]).abs().max().item()
                diff_k = (k_out - ref_data["k_ref"]).abs().max().item()
                diff_v = (v_out - ref_data["v_ref"]).abs().max().item()
                if diff_q > 0.1 or diff_k > 0.1 or diff_v > 0.1 or torch.isnan(q_out).any() or torch.isinf(q_out).any():
                    print(f"[Autotune] fused_qkv config failed correctness check: diff_q={diff_q}, diff_k={diff_k}, diff_v={diff_v}")
                    return float("inf")
                
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    fwd_kernel(ref_data["x_prof"], ref_data["weight_q_prof"], ref_data["weight_k_prof"], ref_data["weight_v_prof"], q_out, k_out, v_out)
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event) / 15.0
                
            elif mode == "bwd_dx":
                from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dx_kernel
                dx_kernel = get_matmul_bwd_dx_kernel(
                    num_warps=config["num_warps"],
                    BLOCK_M=config["BLOCK_M"],
                    BLOCK_N=config["BLOCK_N"],
                    BLOCK_K_STAGE=config["BLOCK_K"],
                    ab_stage=config["ab_stage"],
                    B=ref_data["B_prof"],
                    N=ref_data["N_seq_prof"],
                    K_dim=prof_K,
                    N_dim=prof_N,
                    NUM_HEADS=ref_data["num_heads"],
                    HEAD_DIM=64,
                    q_strides=ref_data["grad_q_prof"].stride(),
                    k_strides=ref_data["grad_k_prof"].stride(),
                    v_strides=ref_data["grad_v_prof"].stride(),
                    maxrregcount=config["maxrregcount"],
                    swizzle_x=config.get("swizzle_x", 8),
                    atom_layout_mnk=config.get("atom_layout_mnk", None)
                )
                grad_x_prof = torch.empty(prof_M, prof_N, device=device, dtype=torch.bfloat16)
                
                dx_kernel(ref_data["grad_q_prof"], ref_data["grad_k_prof"], ref_data["grad_v_prof"], ref_data["weight_prof"], grad_x_prof)
                
                diff = (grad_x_prof - ref_data["ref_grad_x"]).abs().max().item()
                if diff > 1e-1 or torch.isnan(grad_x_prof).any() or torch.isinf(grad_x_prof).any():
                    print(f"[Autotune] bwd_dx config {config} failed correctness: diff={diff}, has_nan={torch.isnan(grad_x_prof).any()}, has_inf={torch.isinf(grad_x_prof).any()}")
                    return float("inf")
                
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    dx_kernel(ref_data["grad_q_prof"], ref_data["grad_k_prof"], ref_data["grad_v_prof"], ref_data["weight_prof"], grad_x_prof)
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event) / 15.0
                
            elif mode == "bwd_dw":
                from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dw_kernel
                split_k = config.get("SPLIT_K", 1)
                dw_kernel = get_matmul_bwd_dw_kernel(
                    num_warps=config["num_warps"],
                    BLOCK_M=config["BLOCK_M"],
                    BLOCK_N=config["BLOCK_N"],
                    BLOCK_K_STAGE=config["BLOCK_K"],
                    ab_stage=config["ab_stage"],
                    B=ref_data["B_prof"],
                    N=ref_data["N_seq_prof"],
                    M_dim=prof_M,
                    K_dim=prof_K,
                    N_dim=prof_N,
                    NUM_HEADS=ref_data["num_heads"],
                    HEAD_DIM=64,
                    q_strides=ref_data["grad_q_prof"].stride(),
                    k_strides=ref_data["grad_k_prof"].stride(),
                    v_strides=ref_data["grad_v_prof"].stride(),
                    maxrregcount=config["maxrregcount"],
                    SPLIT_K=split_k,
                    swizzle_x=config.get("swizzle_x", 8),
                    accumulate=False,
                    atom_layout_mnk=config.get("atom_layout_mnk", None)
                )
                grad_weight_prof = torch.zeros((split_k * prof_M, prof_N), device=device, dtype=torch.bfloat16)
                
                dw_kernel(ref_data["grad_q_prof"], ref_data["grad_k_prof"], ref_data["grad_v_prof"], ref_data["x_prof"], grad_weight_prof)
                
                if split_k > 1:
                    grad_weight_prof_summed = grad_weight_prof.view(split_k, prof_M, prof_N).float().sum(dim=0).to(torch.bfloat16)
                else:
                    grad_weight_prof_summed = grad_weight_prof.to(torch.bfloat16)
                    
                diff = (grad_weight_prof_summed - ref_data["ref_grad_weight"]).abs().max().item()
                if diff > 0.3 or torch.isnan(grad_weight_prof_summed).any() or torch.isinf(grad_weight_prof_summed).any():
                    print(f"[Autotune] bwd_dw config {config} failed correctness: diff={diff}, has_nan={torch.isnan(grad_weight_prof_summed).any()}, has_inf={torch.isinf(grad_weight_prof_summed).any()}")
                    return float("inf")
                
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    dw_kernel(ref_data["grad_q_prof"], ref_data["grad_k_prof"], ref_data["grad_v_prof"], ref_data["x_prof"], grad_weight_prof)
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event) / 15.0
                
            elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
                from flash_attn_prologue.fused_epilogue import get_fused_epilogue_kernel, get_fused_epilogue_back_kernel
                
                hg = config["HEAD_GROUP"]
                bm = config["BLOCK_M"]
                warps = config["num_warps"]
                regs = config["maxrregcount"]
                
                num_heads = N
                head_dim = K
                B_prof = 2
                seq_prof = prof_M // 2
                
                q_prof = torch.randn(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                k_prof = torch.randn(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                rmsnorm_weight_prof = torch.randn(2, 1, head_dim, device=device, dtype=torch.bfloat16)
                scales_prof = torch.randn(num_heads, 1, 1, device=device, dtype=torch.bfloat16)
                sin_prof = torch.randn(B_prof, 1, seq_prof, head_dim // 2, device=device, dtype=torch.float32)
                cos_prof = torch.randn(B_prof, 1, seq_prof, head_dim // 2, device=device, dtype=torch.float32)
                
                fwd_kernel = None
                back_kernel = None
                
                if mode in ("epilogue", "epilogue_fwd"):
                    fwd_kernel = get_fused_epilogue_kernel(
                        num_warps=warps,
                        BLOCK_M=bm,
                        NUM_HEADS=num_heads,
                        HEAD_DIM=head_dim,
                        HEAD_GROUP=hg,
                        ab_stage=config.get("ab_stage", 1),
                        q_in_strides=q_prof.stride(),
                        k_in_strides=k_prof.stride(),
                        q_out_strides=q_prof.stride(),
                        k_out_strides=k_prof.stride(),
                        sin_strides=sin_prof.stride(),
                        cos_strides=cos_prof.stride(),
                        maxrregcount=regs,
                        B=B_prof,
                        N=seq_prof,
                        q_dtype=q_prof.dtype,
                        k_dtype=k_prof.dtype,
                        weight_dtype=rmsnorm_weight_prof.dtype,
                        scales_dtype=scales_prof.dtype,
                        sin_dtype=sin_prof.dtype,
                        cos_dtype=cos_prof.dtype
                    )
                
                if mode in ("epilogue", "epilogue_bwd"):
                    back_kernel = get_fused_epilogue_back_kernel(
                        num_warps=warps,
                        BLOCK_M=bm,
                        NUM_HEADS=num_heads,
                        HEAD_DIM=head_dim,
                        HEAD_GROUP=hg,
                        ab_stage=1,
                        q_strides=q_prof.stride(),
                        k_strides=k_prof.stride(),
                        sin_strides=sin_prof.stride(),
                        cos_strides=cos_prof.stride(),
                        maxrregcount=regs,
                        B=B_prof,
                        N=seq_prof,
                        q_dtype=q_prof.dtype,
                        k_dtype=k_prof.dtype,
                        weight_dtype=rmsnorm_weight_prof.dtype,
                        scales_dtype=scales_prof.dtype,
                        sin_dtype=sin_prof.dtype,
                        cos_dtype=cos_prof.dtype
                    )
                
                grad_q_prof = torch.randn_like(q_prof)
                grad_k_prof = torch.randn_like(k_prof)
                dq_prof = torch.empty_like(q_prof)
                dk_prof = torch.empty_like(k_prof)
                dw_prof = torch.zeros((2, 1, head_dim), device=device, dtype=torch.float32)
                ds_prof = torch.zeros((num_heads, 1, 1), device=device, dtype=torch.float32)
                q_out_prof = torch.empty_like(q_prof)
                k_out_prof = torch.empty_like(k_prof)
                
                # Warmup
                if fwd_kernel is not None:
                    fwd_kernel(q_prof, k_prof, q_out_prof, k_out_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof)
                if back_kernel is not None:
                    back_kernel(
                        q_prof, k_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof,
                        grad_q_prof, grad_k_prof, dq_prof, dk_prof, dw_prof, ds_prof
                    )
                
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    if fwd_kernel is not None:
                        fwd_kernel(q_prof, k_prof, q_out_prof, k_out_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof)
                    if back_kernel is not None:
                        back_kernel(
                            q_prof, k_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof,
                            grad_q_prof, grad_k_prof, dq_prof, dk_prof, dw_prof, ds_prof
                        )
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event) / 15.0
            else:
                return float("inf")
        except BaseException as e:
            import traceback
            print(f"[Autotune] Profiling failed for config {config}. Error: {e}")
            traceback.print_exc()
            return float("inf")

    # ------------------ STAGE 1: TILE SWEEP ------------------
    tile_candidates = []
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import cpasync
    
    if mode == "fused_qkv":
        for hg in [1, 2, 4, 8, 16]:
            if N % hg != 0:
                continue
            for bm in [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
                if prof_M % bm != 0:
                    continue
                for bk in [16, 32, 48, 64, 80, 96, 112, 128]:
                    if prof_K % bk != 0:
                        continue
                    
                    # occupancy / static footprint checks
                    num_threads = 128  # default Stage 1 warps=4
                    if num_threads < bm * hg:
                        continue
                    
                    best_stage = None
                    for stage in [4, 3, 2]:
                        smem_bytes = get_smem_bytes(mode, bm, None, bk, hg, stage)
                        if smem_bytes <= 99 * 1024:
                            best_stage = stage
                            break
                    
                    if best_stage is not None:
                        candidate = {
                            "HEAD_GROUP": hg,
                            "BLOCK_M": bm,
                            "BLOCK_K": bk,
                            "ab_stage": best_stage,
                            "num_warps": 4,
                            "maxrregcount": 255,
                            "swizzle_x": 8,
                            "atom_layout_mnk": None
                        }
                        if is_config_valid(candidate, mode):
                            tile_candidates.append(candidate)
    elif mode == "bwd_dx":
        for bm in [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
            if prof_M % bm != 0:
                continue
            for bn in [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
                if prof_N % bn != 0:
                    continue
                for bk in [16, 32, 48, 64, 80, 96, 112, 128]:
                    if prof_K % bk != 0:
                        continue
                    
                    best_stage = None
                    for stage in [3, 2]:
                        pad = 8 if (bk // 16) > 1 else 0
                        smem_bytes = get_smem_bytes(mode, bm, bn, bk, None, stage, pad)
                        if smem_bytes <= 99 * 1024:
                            best_stage = stage
                            break
                            
                    if best_stage is not None:
                        candidate = {
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "ab_stage": best_stage,
                            "num_warps": 4,
                            "maxrregcount": 255,
                            "swizzle_x": 8,
                            "atom_layout_mnk": None
                        }
                        if is_config_valid(candidate, mode):
                            tile_candidates.append(candidate)
    elif mode == "bwd_dw":
        for bm in [16, 32, 64]:  # restricted to <= 64 (HEAD_DIM limit)
            if (prof_M // 3) % bm != 0:
                continue
            for bn in [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
                if prof_N % bn != 0:
                    continue
                for bk in [16, 32, 48, 64, 80, 96, 112, 128]:
                    if prof_K % bk != 0:
                        continue
                        
                    best_stage = None
                    for stage in [3, 2]:
                        smem_bytes = get_smem_bytes(mode, bm, bn, bk, None, stage)
                        if smem_bytes <= 99 * 1024:
                            best_stage = stage
                            break
                            
                    if best_stage is not None:
                        candidate = {
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "ab_stage": best_stage,
                            "num_warps": 4,
                            "maxrregcount": 255,
                            "SPLIT_K": 4,
                            "swizzle_x": 8,
                            "atom_layout_mnk": None
                        }
                        if is_config_valid(candidate, mode):
                            tile_candidates.append(candidate)
    elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
        for hg in [1, 2, 4, 8, 16]:
            if mode in ("epilogue", "epilogue_fwd") and hg > 2:
                continue
            if N % hg != 0:
                continue
            for bm in [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
                if prof_M % bm != 0:
                    continue
                stage = 1
                warps = 4
                regs = 255
                
                if mode == "epilogue_bwd":
                    s_rmsnorm_reduction_elems = bm * 2 * hg * K
                    s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                    total_elems = s_scales_offset + bm * hg
                    flat_smem_elements = ((total_elems + 31) // 32) * 32
                    smem_bytes = flat_smem_elements * 4
                else:
                    block_n = hg * K
                    block_n_padded = block_n + 8
                    smem_bytes = 2 * (bm * block_n_padded * 2)
                    if mode == "epilogue":
                        s_rmsnorm_reduction_elems = bm * 2 * hg * K
                        s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                        total_elems = s_scales_offset + bm * hg
                        flat_smem_elements = ((total_elems + 31) // 32) * 32
                        bwd_smem_bytes = flat_smem_elements * 4
                        smem_bytes = max(smem_bytes, bwd_smem_bytes)
                        
                if smem_bytes <= 99 * 1024:
                    candidate = {
                        "HEAD_GROUP": hg,
                        "BLOCK_M": bm,
                        "BLOCK_K": K,
                        "ab_stage": stage,
                        "num_warps": warps,
                        "maxrregcount": regs,
                        "atom_layout_mnk": None
                    }
                    if is_config_valid(candidate, mode):
                        tile_candidates.append(candidate)
    else:
        tile_candidates = [
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128},
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
        ]
        
    if not tile_candidates:
        if mode == "fused_qkv":
            return {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_K": 16, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255, "swizzle_x": 8, "atom_layout_mnk": None}
        elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            return {"HEAD_GROUP": 2, "BLOCK_M": 64, "ab_stage": 1, "num_warps": 4, "maxrregcount": 255, "atom_layout_mnk": None}
        elif mode == "bwd_dx":
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255, "swizzle_x": 8, "atom_layout_mnk": None}
        elif mode == "bwd_dw":
            return {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "ab_stage": 2, "num_warps": 4, "maxrregcount": 255, "SPLIT_K": 16, "swizzle_x": 8, "atom_layout_mnk": None}
        else:
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}

    if initial_config is not None:
        print(f"[Autotune] Running Coordinate Descent starting from: {initial_config}")
        
        # 1. Define validator function that includes SMEM size checks
        def is_valid(config):
            if not is_config_valid(config, mode):
                return False
                
            bm = config.get("BLOCK_M", 64)
            bn = config.get("BLOCK_N", 64)
            bk = config.get("BLOCK_K", 32)
            hg = config.get("HEAD_GROUP", 1)
            stage = config.get("ab_stage", 1)
            
            # Enforce limits on epilogue stage/warps/hg
            if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
                if mode == "epilogue_bwd":
                    if stage != 1:
                        return False
                else: # epilogue, epilogue_fwd
                    if stage not in (1, 2):
                        return False
                    if bm % stage != 0:
                        return False
                if N % hg != 0:
                    return False
                if mode in ("epilogue", "epilogue_fwd") and hg > 2:
                    return False
                    
            smem_bytes = get_smem_bytes(mode, bm, bn, bk, hg, stage, pad=0, K=bk)
            if smem_bytes > 99 * 1024:
                return False
                
            return True
            
        # 2. Define coordinate descent param_space
        if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            param_space = {
                "HEAD_GROUP": {"type": "power_of_2", "min": 1, "max": 16},
                "BLOCK_M": {"type": "both", "min": 16, "max": 256},
                "BLOCK_K": [K],
                "ab_stage": {"type": "step_1", "min": 1, "max": 2 if mode in ("epilogue", "epilogue_fwd") else 1},
                "num_warps": {"type": "both", "min": 2, "max": 16},
                "maxrregcount": {"type": "maxrregcount", "min": 128, "max": 255},
                "atom_layout_mnk": [None]
            }
        elif mode == "bwd_dx":
            param_space = {
                "BLOCK_M": {"type": "both", "min": 16, "max": 256},
                "BLOCK_N": {"type": "power_of_2", "min": 16, "max": 256},
                "BLOCK_K": {"type": "power_of_2", "min": 16, "max": 128},
                "ab_stage": {"type": "step_1", "min": 2, "max": 10},
                "num_warps": {"type": "both", "min": 2, "max": 16},
                "maxrregcount": {"type": "maxrregcount", "min": 128, "max": 255},
                "swizzle_x": {"type": "power_of_2", "min": 1, "max": 16},
                "atom_layout_mnk": {"type": "custom", "fn": lambda config: get_valid_warp_layouts(config.get("num_warps", 4))}
            }
        elif mode == "bwd_dw":
            param_space = {
                "BLOCK_M": {"type": "both", "min": 16, "max": 256},
                "BLOCK_N": {"type": "power_of_2", "min": 16, "max": 256},
                "BLOCK_K": {"type": "power_of_2", "min": 16, "max": 128},
                "ab_stage": {"type": "step_1", "min": 2, "max": 10},
                "num_warps": {"type": "both", "min": 2, "max": 16},
                "maxrregcount": {"type": "maxrregcount", "min": 128, "max": 255},
                "SPLIT_K": {"type": "both", "min": 1, "max": 128},
                "swizzle_x": {"type": "power_of_2", "min": 1, "max": 16},
                "atom_layout_mnk": {"type": "custom", "fn": lambda config: get_valid_warp_layouts(config.get("num_warps", 4))}
            }
        elif mode == "fused_qkv":
            param_space = {
                "HEAD_GROUP": {"type": "power_of_2", "min": 1, "max": 16},
                "BLOCK_M": {"type": "both", "min": 16, "max": 256},
                "BLOCK_K": {"type": "power_of_2", "min": 16, "max": 128},
                "ab_stage": {"type": "step_1", "min": 2, "max": 10},
                "num_warps": {"type": "both", "min": 2, "max": 16},
                "maxrregcount": {"type": "maxrregcount", "min": 128, "max": 255},
                "swizzle_x": {"type": "power_of_2", "min": 1, "max": 16},
                "atom_layout_mnk": {"type": "custom", "fn": lambda config: get_valid_warp_layouts(config.get("num_warps", 4)) + [None]}
            }
        else:
            param_space = {
                "BLOCK_M": {"type": "both", "min": 16, "max": 256},
                "BLOCK_N": {"type": "power_of_2", "min": 16, "max": 256},
                "BLOCK_K": {"type": "power_of_2", "min": 16, "max": 128}
            }
            
        import copy
        from flash_attn_prologue.coordinate_descent import CoordinateDescentAutotuner
        tuner = CoordinateDescentAutotuner(
            param_space=param_space,
            objective_fn=profile_and_verify,
            is_valid_fn=is_valid,
            maximize=False,
            max_iterations=3,
            verbose=True
        )
        # Verify and coerce initial_config parameters just in case
        start_config = copy.deepcopy(initial_config)
        if "BLOCK_K" not in start_config:
            start_config["BLOCK_K"] = K
        best_config, best_time = tuner.tune(start_config)
        print(f"[Autotune] Coordinate Descent completed. Best config: {best_config} with time: {best_time:.4f} ms")
        return best_config

    print(f"[Autotune] Stage 1: Sweep over {len(tile_candidates)} tile configurations...")
    tile_results = []
    for t_config in tile_candidates:
        t_ms = profile_and_verify(t_config)
        if t_ms < float("inf"):
            tile_results.append((t_ms, t_config))
            
    tile_results.sort(key=lambda x: x[0])
    top_tiles = [item[1] for item in tile_results[:5]]
    
    if not top_tiles:
        if mode == "fused_qkv":
            return {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_K": 16, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255, "swizzle_x": 8, "atom_layout_mnk": None}
        elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            return {"HEAD_GROUP": 2, "BLOCK_M": 64, "ab_stage": 1, "num_warps": 4, "maxrregcount": 255, "atom_layout_mnk": None}
        elif mode == "bwd_dx":
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255, "swizzle_x": 8, "atom_layout_mnk": None}
        elif mode == "bwd_dw":
            return {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "ab_stage": 2, "num_warps": 4, "maxrregcount": 255, "SPLIT_K": 16, "swizzle_x": 8, "atom_layout_mnk": None}
        else:
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
            
    print(f"[Autotune] Stage 1 Completed. Top tile configurations: {top_tiles}")
    
    # ------------------ STAGE 2: WARP COUNT & LAYOUT SWEEP ------------------
    stage_2_candidates = []
    for tile in top_tiles:
        warp_choices = [2, 4, 8, 12, 16] if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd") else [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
        for warps in warp_choices:
            layouts = (get_valid_warp_layouts(warps) if mode not in ("epilogue", "epilogue_fwd", "epilogue_bwd") else [None])
            for w_layout in layouts:
                candidate = tile.copy()
                candidate["num_warps"] = warps
                candidate["atom_layout_mnk"] = w_layout
                if is_config_valid(candidate, mode):
                    stage_2_candidates.append(candidate)
                
    print(f"[Autotune] Stage 2: Sweep over {len(stage_2_candidates)} warp count & layout configurations...")
    stage_2_results = []
    for exec_config in stage_2_candidates:
        exec_ms = profile_and_verify(exec_config)
        if exec_ms < float("inf"):
            stage_2_results.append((exec_ms, exec_config))
            
    stage_2_results.sort(key=lambda x: x[0])
    if not stage_2_results:
        best_stage_2 = top_tiles[0]
    else:
        best_stage_2 = stage_2_results[0][1]
        
    print(f"[Autotune] Stage 2 Completed. Best warp/layout setup: {best_stage_2}")
    
    # ------------------ STAGE 3: PIPELINING & PARALLELISM SWEEP ------------------
    stage_3_candidates = []
    stages_to_sweep = [2, 3, 4, 5, 6, 7, 8, 9, 10] if mode not in ("epilogue", "epilogue_fwd", "epilogue_bwd") else ([1, 2] if mode in ("epilogue", "epilogue_fwd") else [1])
    split_k_to_sweep = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128] if mode == "bwd_dw" else [1]
    
    for stage in stages_to_sweep:
        if mode in ("epilogue", "epilogue_fwd"):
            if best_stage_2["BLOCK_M"] % stage != 0:
                continue
            num_threads = best_stage_2["num_warps"] * 32
            if mode == "epilogue_fwd":
                required_threads = (best_stage_2["BLOCK_M"] // stage) * best_stage_2["HEAD_GROUP"]
            else:
                required_threads = max((best_stage_2["BLOCK_M"] // stage) * best_stage_2["HEAD_GROUP"], best_stage_2["BLOCK_M"] * best_stage_2["HEAD_GROUP"])
            if num_threads < required_threads:
                continue

        for split_k in split_k_to_sweep:
            # Check dynamic shared memory size limit
            if mode == "fused_qkv":
                block_n = best_stage_2["HEAD_GROUP"] * 3 * 64
                mainloop_bytes = (stage * best_stage_2["BLOCK_M"] * best_stage_2["BLOCK_K"] * 2) + (stage * block_n * best_stage_2["BLOCK_K"] * 2)
                epilogue_bytes = best_stage_2["BLOCK_M"] * (block_n + 8) * 4
                smem_bytes = max(mainloop_bytes, epilogue_bytes)
            elif mode == "bwd_dx":
                sX_bytes = stage * best_stage_2["BLOCK_M"] * best_stage_2["BLOCK_K"] * 3 * 2
                sW_bytes = stage * best_stage_2["BLOCK_N"] * best_stage_2["BLOCK_K"] * 3 * 2
                mainloop_bytes = sX_bytes + sW_bytes
                epilogue_bytes = best_stage_2["BLOCK_M"] * (best_stage_2["BLOCK_N"] + 8) * 2
                smem_bytes = max(mainloop_bytes, epilogue_bytes)
            elif mode == "bwd_dw":
                sX_bytes = stage * best_stage_2["BLOCK_M"] * best_stage_2["BLOCK_K"] * 3 * 2
                sW_bytes = stage * best_stage_2["BLOCK_N"] * best_stage_2["BLOCK_K"] * 2
                mainloop_bytes = sX_bytes + sW_bytes
                epilogue_bytes = 3 * best_stage_2["BLOCK_M"] * (best_stage_2["BLOCK_N"] + 8) * 2
                smem_bytes = max(mainloop_bytes, epilogue_bytes)
            elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
                if mode == "epilogue_bwd":
                    s_rmsnorm_reduction_elems = best_stage_2["BLOCK_M"] * 2 * best_stage_2["HEAD_GROUP"] * K
                    s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                    total_elems = s_scales_offset + best_stage_2["BLOCK_M"] * best_stage_2["HEAD_GROUP"]
                    flat_smem_elements = ((total_elems + 31) // 32) * 32
                    smem_bytes = flat_smem_elements * 4
                else:
                    block_n = best_stage_2["HEAD_GROUP"] * K
                    block_n_padded = block_n + 8
                    smem_bytes = 2 * (best_stage_2["BLOCK_M"] * block_n_padded * 2)
                    if mode == "epilogue":
                        s_rmsnorm_reduction_elems = best_stage_2["BLOCK_M"] * 2 * best_stage_2["HEAD_GROUP"] * K
                        s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                        total_elems = s_scales_offset + best_stage_2["BLOCK_M"] * best_stage_2["HEAD_GROUP"]
                        flat_smem_elements = ((total_elems + 31) // 32) * 32
                        bwd_smem_bytes = flat_smem_elements * 4
                        smem_bytes = max(smem_bytes, bwd_smem_bytes)
            else:
                smem_bytes = 0
                
            if smem_bytes <= 99 * 1024:
                candidate = best_stage_2.copy()
                candidate["ab_stage"] = stage
                if mode == "bwd_dw":
                    candidate["SPLIT_K"] = split_k
                if is_config_valid(candidate, mode):
                    stage_3_candidates.append(candidate)
                
    print(f"[Autotune] Stage 3: Sweep over {len(stage_3_candidates)} pipelining & parallelism configurations...")
    stage_3_results = []
    for exec_config in stage_3_candidates:
        exec_ms = profile_and_verify(exec_config)
        if exec_ms < float("inf"):
            stage_3_results.append((exec_ms, exec_config))
            
    stage_3_results.sort(key=lambda x: x[0])
    if not stage_3_results:
        best_stage_3 = best_stage_2
    else:
        best_stage_3 = stage_3_results[0][1]
        
    print(f"[Autotune] Stage 3 Completed. Best stage/split-K setup: {best_stage_3}")
    
    # ------------------ STAGE 4: COMPILER & SWIZZLE CONFIGURATIONS ------------------
    stage_4_candidates = []
    regs_to_sweep = [128, 192, 255] if mode not in ("epilogue", "epilogue_fwd", "epilogue_bwd") else [128, 255]
    swizzles_to_sweep = [1, 4, 8, 16] if mode not in ("epilogue", "epilogue_fwd", "epilogue_bwd") else [1]
    
    for regs in regs_to_sweep:
        for swizzle in swizzles_to_sweep:
            candidate = best_stage_3.copy()
            candidate["maxrregcount"] = regs
            if mode not in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
                candidate["swizzle_x"] = swizzle
            stage_4_candidates.append(candidate)
            
    print(f"[Autotune] Stage 4: Sweep over {len(stage_4_candidates)} compiler & swizzle configurations...")
    stage_4_results = []
    for exec_config in stage_4_candidates:
        exec_ms = profile_and_verify(exec_config)
        if exec_ms < float("inf"):
            stage_4_results.append((exec_ms, exec_config))
            
    stage_4_results.sort(key=lambda x: x[0])
    if not stage_4_results:
        best_config = best_stage_3
        best_time = float("inf")
    else:
        best_config = stage_4_results[0][1]
        best_time = stage_4_results[0][0]
        
    print(f"[Autotune] Selection completed. Best config for mode {mode}: {best_config} with time: {best_time:.4f} ms")
    return best_config
 
def get_autotuned_config(M: int, N: int, K: int, is_fused: Any, device: str = "cuda", initial_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    if isinstance(is_fused, bool):
        mode = "fused" if is_fused else "gemm"
    else:
        mode = is_fused
    key = f"{M}_{N}_{K}_{mode}"
    if key in _autotune_in_memory_cache:
        return _autotune_in_memory_cache[key]
        
    cache = load_cache()
    if key in cache and initial_config is None:
        return cache[key]
        
    print(f"[Autotune] Shape key '{key}' not found in cache or autotuning triggered. Aggressively profiling {mode} config candidates...")
    best_config = profile_shapes(M, N, K, mode, device, initial_config)
    print(f"[Autotune] Selected best config for '{key}': {best_config}")
    
    cache[key] = best_config
    save_cache(cache)
    _autotune_in_memory_cache[key] = best_config
    return best_config
