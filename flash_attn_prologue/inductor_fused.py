from typing import Tuple

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

options = {
    # Your current settings
    "epilogue_fusion": True,
    "prologue_fusion": True,
    "max_autotune": True,
    "triton.cudagraphs": False,
    "max_autotune_gemm": True,
    "aggressive_fusion": True,
    "max_autotune_pointwise": True,
    "online_softmax": True,
    "layout_optimization": True,
    "b2b_gemm_pass": True,
    "joint_graph_constant_folding": True,
    # "batch_fusion": True,
    # "split_cat_fx_passes": True,
    # "combo_kernels": True,
    "benchmark_epilogue_fusion": True,
    # "benchmark_fusion": True,

    # Extreme Autotuning (Slow Covmpile)
    "coordinate_descent_tuning": False,
    "triton.autotune_cublasLt": False,

    # Memory & Cache Optimization
    "memory_planning": True,
    "inplace_buffers": True,
    "allow_buffer_reuse": True,
    "reorder_for_locality": True,

    # Advanced Triton Fusions
    "group_fusion": True,
    "triton.persistent_reductions": True,

    # Fast Math
    "use_fast_math": True,
}

# @torch.compile(fullgraph=True, dynamic=False, options=options)
# def fused_qkv_op(
#     x: torch.Tensor,
#     rmsnorm_weight: torch.Tensor,
#     scales: torch.Tensor,
#     freqs: torch.Tensor,
#     num_heads: int,
#     head_dim: int,
#     eps: float
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Forward Pass (Eager execution until CuTe kernel is active).
#     Returns (q_proj, k_proj, v_proj, q_rsqrt, k_rsqrt)
#     """

#     # overlap casts and shape manipulation with gemm
#     rmsnorm_weight = rmsnorm_weight.float() # (2, 1, dhead)
#     # f1, f2 = freqs.chunk(2, dim=-1) # (batch, seq, 1, 1, dhead/2) which broadcasts on the nheads and qk dim
#     scales = scales.float().exp().to(x.dtype) # [nheads, 1]

#     qkv = qkv.view(B, N, 3, num_heads, head_dim)
#     qk, v_proj = qkv[..., 0:2, :, :].float(), qkv[..., 2, :, :]

#     # overlap shape ops and save V while we compute RMS Norm and RoPE on QK
#     v_proj = v_proj.permute(0, 2, 1, 3).contiguous() # (b, h, n, d)

#     # parallelize QKRMSNorm and RoPE across Q and K at once

#     qk_normed = qk * torch.rsqrt(torch.pow(qk, 2).mean(-1, keepdim=True) + eps) * rmsnorm_weight

#     qk1, qk2 = qk_normed.chunk(2, dim=-1)

#     # Calc cos and sin from Freqs on the fly to save on loading and SMEM/RMEM pressure by half while not forching memory copy and reshapes with cat
#     out1 = qk1 * torch.cos(freqs) - qk2 * torch.sin(freqs)
#     out2 = qk2 * torch.cos(freqs) + qk1 * torch.sin(freqs)

#     out = torch.cat((out1, out2), dim=-1).to(x.dtype)

#     # overlap shape ops and save K while we compute Q scaling
#     k_proj = out[:, :, 1, :, :].permute(0, 2, 1, 3).contiguous() # (b, h, n, d)

#     q_proj = (out[:, :, 0, :, :] * scales).permute(0, 2, 1, 3).contiguous() # (b, h, n, d)

#     return q_proj, k_proj, v_proj




def qknorm_rope_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused QK RMSNorm and RoPE with scaling.
    """

    # Cast parameters to float for computation
    rmsnorm_weight = rmsnorm_weight.float()
    scales = scales.float().exp()

    qk = torch.stack([q, k], dim=2).float()

    # [batch, nheads, 2, seq, dhead]

    qk_normed = qk * torch.rsqrt(qk.square().mean(-1, keepdim=True) + eps) * rmsnorm_weight

    qk1, qk2 = qk_normed.chunk(2, dim=-1)

    # Standardize sin/cos to 5D shape [B, H, 1, N, D // 2] for broadcasting
    if cos.ndim == 4:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1).unsqueeze(2)
        sin = sin.unsqueeze(1).unsqueeze(2)
    elif cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(1).unsqueeze(2)

    out1 = qk1 * cos - qk2 * sin
    out2 = qk2 * cos + qk1 * sin

    out = torch.cat((out1, out2), dim=-1)

    return (out[:, :, 0, :, :] * scales).to(q.dtype), out[:, :, 1, :, :].to(k.dtype)
