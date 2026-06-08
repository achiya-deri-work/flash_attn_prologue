import os
import json
import torch
import math
from typing import Dict, Any

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

def profile_shapes(M: int, N: int, K: int, is_fused: Any, device: str) -> Dict[str, int]:
    if isinstance(is_fused, bool):
        mode = "fused" if is_fused else "gemm"
    else:
        mode = is_fused

    if mode == "fused":
        candidates = [
            {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_N": 192, "BLOCK_K": 32},
            {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_N": 192, "BLOCK_K": 16},
            {"HEAD_GROUP": 1, "BLOCK_M": 32, "BLOCK_N": 192, "BLOCK_K": 32},
            {"HEAD_GROUP": 1, "BLOCK_M": 32, "BLOCK_N": 192, "BLOCK_K": 16},
            {"HEAD_GROUP": 2, "BLOCK_M": 64, "BLOCK_N": 384, "BLOCK_K": 32},
            {"HEAD_GROUP": 2, "BLOCK_M": 64, "BLOCK_N": 384, "BLOCK_K": 16},
            {"HEAD_GROUP": 2, "BLOCK_M": 32, "BLOCK_N": 384, "BLOCK_K": 32},
            {"HEAD_GROUP": 2, "BLOCK_M": 32, "BLOCK_N": 384, "BLOCK_K": 16},
            {"HEAD_GROUP": 4, "BLOCK_M": 32, "BLOCK_N": 768, "BLOCK_K": 16},
            {"HEAD_GROUP": 4, "BLOCK_M": 16, "BLOCK_N": 768, "BLOCK_K": 32},
            {"HEAD_GROUP": 4, "BLOCK_M": 16, "BLOCK_N": 768, "BLOCK_K": 16},
            {"HEAD_GROUP": 8, "BLOCK_M": 16, "BLOCK_N": 1536, "BLOCK_K": 16},
            {"HEAD_GROUP": 8, "BLOCK_M": 16, "BLOCK_N": 1536, "BLOCK_K": 8},
        ]
    elif mode == "fused_qkv":
        candidates = []
        for hg in [1, 2]:
            for bm in [32, 64]:
                for bk in [16, 32]:
                    for stage in [2, 3]:
                        for warps in [4, 8]:
                            for regs in [128, 255]:
                                # Calculate SMEM bytes
                                block_n = hg * 3 * 64
                                mainloop_bytes = (stage * bm * bk * 2) + (stage * block_n * bk * 2)
                                epilogue_bytes = bm * (block_n + 8) * 4
                                smem_bytes = max(mainloop_bytes, epilogue_bytes)
                                if smem_bytes <= 99 * 1024:
                                    candidates.append({
                                        "HEAD_GROUP": hg,
                                        "BLOCK_M": bm,
                                        "BLOCK_K": bk,
                                        "ab_stage": stage,
                                        "num_warps": warps,
                                        "maxrregcount": regs
                                    })
    elif mode in ("bwd_dx", "bwd_dw"):
        candidates = []
        if mode == "bwd_dx":
            # For dx, symmetric block shape 128x128 is correct and performant.
            # Avoid asymmetric block shapes or smaller 64x64 shapes to prevent layout corruption or race conditions.
            for block_size in [128]:
                for bk in [32, 64]:
                    for regs in [255]:
                        candidates.append({
                            "BLOCK_M": block_size,
                            "BLOCK_N": block_size,
                            "BLOCK_K": bk,
                            "ab_stage": 3,
                            "num_warps": 4,
                            "maxrregcount": regs
                        })
        else: # mode == "bwd_dw"
            # For dw, 128x128x32 is correct and performant under Split-K.
            candidates.append({
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "ab_stage": 3,
                "num_warps": 4,
                "maxrregcount": 255
            })
    elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
        candidates = []
        for hg in [1, 2, 4, 8]:
            if mode in ("epilogue", "epilogue_fwd") and hg > 2:
                continue
            if N % hg != 0: # N is NUM_HEADS
                continue
            for bm in [16, 32, 64, 128, 256]:
                for warps in [2, 4, 8]:
                    for regs in [128, 255]:
                        # Sweep ab_stage in [1, 2] for fwd/combined, 1 for bwd
                        stages = [1, 2] if mode in ("epilogue", "epilogue_fwd") else [1]
                        for stage in stages:
                            if bm % stage != 0:
                                continue
                            # Ensure the thread count is sufficient to cover the block shape
                            # Forward requires self.SUB_M * self.HEAD_GROUP threads (where SUB_M = bm // stage)
                            # Backward requires self.BLOCK_M * self.HEAD_GROUP threads
                            num_threads = warps * 32
                            if mode == "epilogue_bwd":
                                required_threads = bm * hg
                            elif mode == "epilogue_fwd":
                                required_threads = (bm // stage) * hg
                            else: # mode == "epilogue"
                                required_threads = max((bm // stage) * hg, bm * hg)
                                
                            if num_threads < required_threads:
                                continue
                                
                            # Calculate SMEM bytes depending on the mode
                            if mode == "epilogue_bwd":
                                # Backward SMEM calculation
                                s_rmsnorm_reduction_elems = bm * 2 * hg * K
                                s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                                total_elems = s_scales_offset + bm * hg
                                flat_smem_elements = ((total_elems + 31) // 32) * 32
                                smem_bytes = flat_smem_elements * 4
                            else:
                                # Forward (or combined) SMEM calculation
                                block_n = hg * K
                                block_n_padded = block_n + 8
                                smem_bytes = 2 * (bm * block_n_padded * 2)
                                if mode == "epilogue": # needs both to fit
                                    s_rmsnorm_reduction_elems = bm * 2 * hg * K
                                    s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
                                    total_elems = s_scales_offset + bm * hg
                                    flat_smem_elements = ((total_elems + 31) // 32) * 32
                                    bwd_smem_bytes = flat_smem_elements * 4
                                    smem_bytes = max(smem_bytes, bwd_smem_bytes)
                                    
                            if smem_bytes <= 99 * 1024:
                                candidates.append({
                                    "HEAD_GROUP": hg,
                                    "BLOCK_M": bm,
                                    "BLOCK_K": K,
                                    "ab_stage": stage,
                                    "num_warps": warps,
                                    "maxrregcount": regs
                                })
    else:
        # Candidates: (BLOCK_M, BLOCK_N, BLOCK_K)
        candidates_raw = [
            (128, 128, 64),
            (128, 64, 128),
            (64, 128, 128),
            (64, 64, 128),
            (128, 64, 64),
            (64, 128, 64),
            (64, 64, 64),
            (128, 128, 32),
            (128, 64, 32),
            (64, 128, 32),
            (64, 64, 32),
        ]
        candidates = []
        for bm, bn, bk in candidates_raw:
            # Check active SMEM footprint for 2-stage double-buffered pipeline
            smem_bytes = (bm * bk * 2 * 2) + (bn * bk * 2 * 2)
            if smem_bytes <= 99 * 1024:  # Blackwell sm_120a SMEM limit (99 KB)
                candidates.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
            
    if not candidates:
        if mode == "fused_qkv":
            return {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_K": 16, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            return {"HEAD_GROUP": 2, "BLOCK_M": 64, "ab_stage": 1, "num_warps": 4, "maxrregcount": 255}
        elif mode == "fused":
            return {"HEAD_GROUP": 2, "BLOCK_M": 64, "BLOCK_N": 384, "BLOCK_K": 32}
        elif mode == "bwd_dx":
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        elif mode == "bwd_dw":
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        else:
            return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
        
    best_config = None
    best_time = float("inf")
    
    prof_M = min(M, 16384)  # Clamp sequence length to keep compilation/profiling latency small
    prof_N = N
    prof_K = K
    
    for config in candidates:
        bm = config["BLOCK_M"]
        if bm > M:
            continue
        bn = config.get("BLOCK_N", config.get("HEAD_GROUP", 1) * 3 * 64)
        if mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            if config.get("HEAD_GROUP", 1) > N:
                continue
        else:
            if bn > N:
                continue
        bk = config["BLOCK_K"]
        
        try:
            if mode == "fused":
                from archive.flash_attn_prologue.fused_pipelined_kernel import get_fused_pipelined_fwd_kernel
                
                hg = config["HEAD_GROUP"]
                
                # Check if candidate matches shape dimension divisibility
                if prof_M % bm != 0:
                    continue
                
                # Generate contiguous weight test matrix with correct permuted shape
                num_heads = 16
                head_dim = 64
                hidden = K
                
                fwd_kernel = get_fused_pipelined_fwd_kernel(
                    num_warps=4,
                    BLOCK_M=bm,
                    BLOCK_K_STAGE=bk,
                    NUM_HEADS=num_heads,
                    HEAD_DIM=head_dim,
                    HEAD_GROUP=hg,
                    hidden=hidden
                )
                
                B_prof = 2
                seq_prof = prof_M // 2
                x_prof = torch.randn(B_prof, seq_prof, hidden, device=device, dtype=torch.bfloat16)
                
                # Weight pre-permutation matching the specific HEAD_GROUP
                weight_raw = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=torch.bfloat16)
                w_reshaped = weight_raw.view(3, num_heads, head_dim, hidden)
                w_split = w_reshaped.view(3, num_heads // hg, hg, head_dim, hidden)
                w_permuted = w_split.permute(1, 2, 0, 3, 4)
                weight_prof = w_permuted.reshape(num_heads * 3 * head_dim, hidden).contiguous()
                
                rmsnorm_weight_prof = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
                scales_prof = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)
                sin_prof = torch.randn(B_prof, 1, seq_prof, head_dim, device=device, dtype=torch.float32)
                cos_prof = torch.randn(B_prof, 1, seq_prof, head_dim, device=device, dtype=torch.float32)
                
                q_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                k_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                v_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                
                # Warmup
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
                curr_ms = start_event.elapsed_time(end_event) / 15.0
            elif mode == "fused_qkv":
                from flash_attn_prologue.fused_qkv import get_fused_qkv_kernel
                
                hg = config["HEAD_GROUP"]
                stage = config["ab_stage"]
                warps = config["num_warps"]
                regs = config["maxrregcount"]
                
                # Check if candidate matches shape dimension divisibility
                if prof_M % bm != 0:
                    continue
                
                num_heads = 16
                head_dim = 64
                hidden = K
                
                fwd_kernel = get_fused_qkv_kernel(
                    num_warps=warps,
                    BLOCK_M=bm,
                    BLOCK_K_STAGE=bk,
                    NUM_HEADS=num_heads,
                    HEAD_DIM=head_dim,
                    HEAD_GROUP=hg,
                    hidden=hidden,
                    use_sync_in_loop=True,
                    ab_stage=stage,
                    maxrregcount=regs
                )
                
                B_prof = 2
                seq_prof = prof_M // 2
                x_prof = torch.randn(B_prof, seq_prof, hidden, device=device, dtype=torch.bfloat16)
                weight_prof = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=torch.bfloat16)
                
                num_heads_x_dim = num_heads * head_dim
                weight_q_prof = weight_prof[0 : num_heads_x_dim, :]
                weight_k_prof = weight_prof[num_heads_x_dim : 2 * num_heads_x_dim, :]
                weight_v_prof = weight_prof[2 * num_heads_x_dim : 3 * num_heads_x_dim, :]
                
                q_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                k_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                v_out = torch.empty(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                
                # Warmup
                for _ in range(3):
                    fwd_kernel(x_prof, weight_q_prof, weight_k_prof, weight_v_prof, q_out, k_out, v_out)
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    fwd_kernel(x_prof, weight_q_prof, weight_k_prof, weight_v_prof, q_out, k_out, v_out)
                end_event.record()
                torch.cuda.synchronize()
                curr_ms = start_event.elapsed_time(end_event) / 15.0
            elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
                from flash_attn_prologue.fused_epilogue import get_fused_epilogue_kernel, get_fused_epilogue_back_kernel
                
                hg = config["HEAD_GROUP"]
                bm = config["BLOCK_M"]
                warps = config["num_warps"]
                regs = config["maxrregcount"]
                
                if prof_M % bm != 0:
                    continue
                
                num_heads = N
                head_dim = K
                
                B_prof = 2
                seq_prof = prof_M // 2
                
                q_prof = torch.randn(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                k_prof = torch.randn(B_prof, num_heads, seq_prof, head_dim, device=device, dtype=torch.bfloat16)
                rmsnorm_weight_prof = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
                scales_prof = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)
                sin_prof = torch.randn(B_prof, 1, seq_prof, head_dim, device=device, dtype=torch.float32)
                cos_prof = torch.randn(B_prof, 1, seq_prof, head_dim, device=device, dtype=torch.float32)
                
                q_strides = q_prof.stride()
                k_strides = k_prof.stride()
                sin_strides = sin_prof.stride()
                cos_strides = cos_prof.stride()
                
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
                        q_strides=q_strides,
                        k_strides=k_strides,
                        sin_strides=sin_strides,
                        cos_strides=cos_strides,
                        maxrregcount=regs
                    )
                
                if mode in ("epilogue", "epilogue_bwd"):
                    back_kernel = get_fused_epilogue_back_kernel(
                        num_warps=warps,
                        BLOCK_M=bm,
                        NUM_HEADS=num_heads,
                        HEAD_DIM=head_dim,
                        HEAD_GROUP=hg,
                        ab_stage=1,
                        q_strides=q_strides,
                        k_strides=k_strides,
                        sin_strides=sin_strides,
                        cos_strides=cos_strides,
                        maxrregcount=regs
                    )
                
                grad_q_prof = torch.randn_like(q_prof)
                grad_k_prof = torch.randn_like(k_prof)
                dq_prof = torch.empty_like(q_prof)
                dk_prof = torch.empty_like(k_prof)
                dw_prof = torch.zeros((2, 1, head_dim), device=device, dtype=torch.float32)
                ds_prof = torch.zeros((num_heads, 1, 1), device=device, dtype=torch.float32)
                
                # Warmup
                for _ in range(3):
                    if fwd_kernel is not None:
                        fwd_kernel(q_prof, k_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof)
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
                        fwd_kernel(q_prof, k_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof)
                    if back_kernel is not None:
                        back_kernel(
                            q_prof, k_prof, rmsnorm_weight_prof, scales_prof, sin_prof, cos_prof,
                            grad_q_prof, grad_k_prof, dq_prof, dk_prof, dw_prof, ds_prof
                        )
                end_event.record()
                torch.cuda.synchronize()
                curr_ms = start_event.elapsed_time(end_event) / 15.0
            elif mode == "bwd_dx":
                from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dx_kernel
                
                stage = config["ab_stage"]
                warps = config["num_warps"]
                regs = config["maxrregcount"]
                
                if prof_M % bm != 0 or prof_N % bn != 0:
                    continue
                    
                dx_kernel = get_matmul_bwd_dx_kernel(
                    num_warps=warps,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K_STAGE=bk,
                    ab_stage=stage,
                    K_dim=prof_K,
                    N_dim=prof_N,
                    maxrregcount=regs
                )
                
                grad_output_prof = torch.randn(prof_M, prof_K, device=device, dtype=torch.bfloat16)
                weight_prof = torch.randn(prof_K, prof_N, device=device, dtype=torch.bfloat16)
                grad_x_prof = torch.empty(prof_M, prof_N, device=device, dtype=torch.bfloat16)
                
                for _ in range(3):
                    dx_kernel(grad_output_prof, weight_prof, grad_x_prof)
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    dx_kernel(grad_output_prof, weight_prof, grad_x_prof)
                end_event.record()
                torch.cuda.synchronize()
                curr_ms = start_event.elapsed_time(end_event) / 15.0
            elif mode == "bwd_dw":
                from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dw_kernel
                
                stage = config["ab_stage"]
                warps = config["num_warps"]
                regs = config["maxrregcount"]
                
                if prof_M % bm != 0 or prof_N % bn != 0:
                    continue
                    
                dw_kernel = get_matmul_bwd_dw_kernel(
                    num_warps=warps,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K_STAGE=bk,
                    ab_stage=stage,
                    M_dim=prof_M,
                    K_dim=prof_K,
                    N_dim=prof_N,
                    maxrregcount=regs,
                    SPLIT_K=16
                )
                
                grad_output_t_prof = torch.randn(prof_K, prof_M, device=device, dtype=torch.bfloat16).t()
                x_prof = torch.randn(prof_K, prof_N, device=device, dtype=torch.bfloat16)
                grad_weight_prof = torch.zeros(prof_M, prof_N, device=device, dtype=torch.float32)
                
                for _ in range(3):
                    dw_kernel(grad_output_t_prof, x_prof, grad_weight_prof)
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    dw_kernel(grad_output_t_prof, x_prof, grad_weight_prof)
                end_event.record()
                torch.cuda.synchronize()
                curr_ms = start_event.elapsed_time(end_event) / 15.0
            else:
                from archive.flash_attn_prologue.pipelined_gemm import get_pipelined_gemm_fwd_kernel
                
                if prof_M % bm != 0 or prof_N % bn != 0:
                    continue
                
                fwd_kernel = get_pipelined_gemm_fwd_kernel(
                    num_warps=4,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    hidden=prof_K,
                    out_features=prof_N
                )
                
                B_prof = 2
                seq_prof = prof_M // 2
                x_prof = torch.randn(B_prof, seq_prof, prof_K, device=device, dtype=torch.bfloat16)
                weight_prof = torch.randn(prof_N, prof_K, device=device, dtype=torch.bfloat16)
                out_prof = torch.empty(B_prof, seq_prof, prof_N, device=device, dtype=torch.bfloat16)
                
                # Warmup
                for _ in range(3):
                    fwd_kernel(x_prof, weight_prof, out_prof)
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(15):
                    fwd_kernel(x_prof, weight_prof, out_prof)
                end_event.record()
                torch.cuda.synchronize()
                curr_ms = start_event.elapsed_time(end_event) / 15.0
                
            if curr_ms < best_time:
                best_time = curr_ms
                best_config = config
        except Exception:
            continue
            
    if best_config is None:
        if mode == "fused_qkv":
            best_config = {"HEAD_GROUP": 1, "BLOCK_M": 64, "BLOCK_K": 16, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        elif mode in ("epilogue", "epilogue_fwd", "epilogue_bwd"):
            best_config = {"HEAD_GROUP": 2, "BLOCK_M": 64, "ab_stage": 1, "num_warps": 4, "maxrregcount": 255}
        elif mode == "fused":
            best_config = {"HEAD_GROUP": 2, "BLOCK_M": 64, "BLOCK_N": 384, "BLOCK_K": 32}
        elif mode == "bwd_dx":
            best_config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        elif mode == "bwd_dw":
            best_config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "ab_stage": 3, "num_warps": 4, "maxrregcount": 255}
        else:
            best_config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
        
    return best_config
 
def get_autotuned_config(M: int, N: int, K: int, is_fused: Any, device: str = "cuda") -> Dict[str, int]:
    if isinstance(is_fused, bool):
        mode = "fused" if is_fused else "gemm"
    else:
        mode = is_fused
    key = f"{M}_{N}_{K}_{mode}"
    if key in _autotune_in_memory_cache:
        return _autotune_in_memory_cache[key]
        
    cache = load_cache()
    if key in cache:
        return cache[key]
        
    print(f"[Autotune] Shape key '{key}' not found in cache. Aggressively profiling {mode} config candidates...")
    best_config = profile_shapes(M, N, K, mode, device)
    print(f"[Autotune] Selected best config for '{key}': {best_config}")
    
    cache[key] = best_config
    save_cache(cache)
    _autotune_in_memory_cache[key] = best_config
    return best_config
