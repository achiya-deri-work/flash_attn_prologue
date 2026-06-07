from typing import Tuple

# pyrefly: ignore [missing-import]
import torch 
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


def fused_qkv_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward Pass (Eager execution until CuTe kernel is active).
    Returns (q_proj, k_proj, v_proj, q_rsqrt, k_rsqrt)
    """

    # overlap casts and shape manipulation with gemm
    rmsnorm_weight = rmsnorm_weight.float() # (2, 1, dhead)
    # f1, f2 = freqs.chunk(2, dim=-1) # (batch, seq, 1, 1, dhead/2) which broadcasts on the nheads and qk dim
    scales = scales.float().exp().to(x.dtype) # [nheads, 1]

    B, N, C = x.shape
    qkv = F.linear(x, weight)
    
    qkv = qkv.view(B, N, 3, num_heads, head_dim)
    qk, v_proj = qkv[..., 0:2, :, :].float(), qkv[..., 2, :, :]

    # overlap shape ops and save V while we compute RMS Norm and RoPE on QK
    v_proj = v_proj.permute(0, 2, 1, 3).contiguous() # (b, h, n, d) 

    # parallelize QKRMSNorm and RoPE across Q and K at once 

    qk_normed = qk * torch.rsqrt(torch.pow(qk, 2).mean(-1, keepdim=True) + eps) * rmsnorm_weight

    qk1, qk2 = qk_normed.chunk(2, dim=-1)

    # Calc cos and sin from Freqs on the fly to save on loading and SMEM/RMEM pressure by half while not forching memory copy and reshapes with cat
    out1 = qk1 * torch.cos(freqs) - qk2 * torch.sin(freqs)
    out2 = qk2 * torch.cos(freqs) + qk1 * torch.sin(freqs)

    out = torch.cat((out1, out2), dim=-1).to(x.dtype)
    
    # overlap shape ops and save K while we compute Q scaling
    k_proj = out[:, :, 1, :, :].permute(0, 2, 1, 3).contiguous() # (b, h, n, d)
    
    q_proj = (out[:, :, 0, :, :] * scales).permute(0, 2, 1, 3).contiguous() # (b, h, n, d)
    
    return q_proj, k_proj, v_proj