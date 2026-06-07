import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
import torch.nn.functional as F

def rotate_half(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_half_backward(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)

@torch.library.custom_op("flash_prologue::fused_qkv", mutates_args=())
def fused_qkv_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward Pass (Eager execution until CuTe kernel is active).
    Returns (q_proj, k_proj, v_proj, q_rsqrt, k_rsqrt)
    """
    B, N, C = x.shape
    qkv = F.linear(x, weight)
    
    qkv = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
    q, k, v_proj = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    
    q_fp32 = q.float()
    k_fp32 = k.float()
    
    q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + eps)
    k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + eps)
    
    q_normed = (q_fp32 * q_rsqrt) * rmsnorm_weight[0].float()
    k_normed = (k_fp32 * k_rsqrt) * rmsnorm_weight[1].float()
    
    q_roped_fp32 = (q_normed * cos) + (rotate_half(q_normed) * sin)
    k_roped_fp32 = (k_normed * cos) + (rotate_half(k_normed) * sin)
    
    q_proj = (q_roped_fp32.to(q.dtype) * scales.float().exp().to(q.dtype)).contiguous()
    k_proj = k_roped_fp32.to(k.dtype).contiguous()
    v_proj = v_proj.contiguous()
    
    return q_proj, k_proj, v_proj, q_rsqrt, k_rsqrt

@torch.library.custom_op("flash_prologue::fused_qkv_backward", mutates_args=())
def fused_qkv_backward_op(
    grad_q_proj: torch.Tensor,
    grad_k_proj: torch.Tensor,
    grad_v_proj: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_rsqrt: torch.Tensor,
    k_rsqrt: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward Pass (Eager Recomputation + Inverse Epilogue).
    """
    B, N, C = x.shape
    
    with torch.no_grad():
        qkv = F.linear(x, weight)
        qkv = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
        q, k = qkv[:, :, 0], qkv[:, :, 1]
        
        q_fp32 = q.float()
        k_fp32 = k.float()
    
    # Exact Autograd emulation with BFP16 truncation boundaries
    scale_factor_bf16 = scales.float().exp().to(x.dtype)
    grad_q_scaled = grad_q_proj * scale_factor_bf16
    
    grad_q_scaled_fp32 = grad_q_scaled.float()
    grad_k_proj_fp32 = grad_k_proj.float()
    
    grad_q_normed_fp32 = (grad_q_scaled_fp32 * cos) + rotate_half_backward(grad_q_scaled_fp32 * sin)
    grad_k_normed_fp32 = (grad_k_proj_fp32 * cos) + rotate_half_backward(grad_k_proj_fp32 * sin)
    
    grad_q_normed = grad_q_normed_fp32.to(x.dtype)
    grad_k_normed = grad_k_normed_fp32.to(x.dtype)
    
    q_normed = (q_fp32 * q_rsqrt).to(x.dtype)
    k_normed = (k_fp32 * k_rsqrt).to(x.dtype)
    
    grad_rmsnorm_weight_q = (grad_q_normed * q_normed).float().sum(dim=(0, 1, 2)).view(1, 1, head_dim)
    grad_rmsnorm_weight_k = (grad_k_normed * k_normed).float().sum(dim=(0, 1, 2)).view(1, 1, head_dim)
    grad_rmsnorm_weight = torch.cat((grad_rmsnorm_weight_q, grad_rmsnorm_weight_k), dim=0).to(rmsnorm_weight.dtype)
    
    grad_q_prenorm = grad_q_normed * rmsnorm_weight[0]
    grad_k_prenorm = grad_k_normed * rmsnorm_weight[1]
    
    grad_q_prenorm_fp32 = grad_q_prenorm.float()
    grad_k_prenorm_fp32 = grad_k_prenorm.float()
    
    q_sum = (grad_q_prenorm_fp32 * q_fp32).sum(-1, keepdim=True)
    grad_q_original = q_rsqrt * (grad_q_prenorm_fp32 - q_fp32 * q_sum * (q_rsqrt.pow(2) / head_dim))
    
    k_sum = (grad_k_prenorm_fp32 * k_fp32).sum(-1, keepdim=True)
    grad_k_original = k_rsqrt * (grad_k_prenorm_fp32 - k_fp32 * k_sum * (k_rsqrt.pow(2) / head_dim))
    
    grad_q_original = grad_q_original.to(x.dtype)
    grad_k_original = grad_k_original.to(x.dtype)
    grad_v_original = grad_v_proj
    
    grad_qkv = torch.stack((grad_q_original, grad_k_original, grad_v_original), dim=2)
    grad_qkv = grad_qkv.permute(0, 3, 2, 1, 4).reshape(B, N, 3 * num_heads * head_dim)
    
    grad_x = F.linear(grad_qkv, weight.t())
    grad_weight = F.linear(grad_qkv.reshape(-1, 3 * num_heads * head_dim).t(), x.reshape(-1, C).t())
    
    q_roped = (q_normed.float() * cos) + (rotate_half(q_normed.float()) * sin)
    q_roped = q_roped.to(x.dtype)
    grad_scales = (grad_q_proj * q_roped).float().sum(dim=(0, 2, 3)).view(num_heads, 1, 1).to(scales.dtype)
    grad_scales = (grad_scales.float() * scale_factor_bf16.float()).to(scales.dtype)

    
    return grad_x, grad_weight, grad_rmsnorm_weight, grad_scales

def setup_fused_qkv_context(ctx, inputs, output):
    x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim, eps = inputs
    q, k, v, q_rsqrt, k_rsqrt = output
    ctx.save_for_backward(x, weight, rmsnorm_weight, scales, sin, cos, q_rsqrt, k_rsqrt)
    ctx.num_heads = num_heads
    ctx.head_dim = head_dim
    ctx.eps = eps

def fused_qkv_backward_hook(ctx, grad_q, grad_k, grad_v, grad_q_rsqrt, grad_k_rsqrt):
    x, weight, rmsnorm_weight, scales, sin, cos, q_rsqrt, k_rsqrt = ctx.saved_tensors
    grad_x, grad_w, grad_norm, grad_scales = fused_qkv_backward_op(
        grad_q, grad_k, grad_v, x, weight, rmsnorm_weight, scales, sin, cos, 
        q_rsqrt, k_rsqrt, ctx.num_heads, ctx.head_dim, ctx.eps
    )
    return grad_x, grad_w, grad_norm, grad_scales, None, None, None, None, None

torch.library.register_autograd(
    "flash_prologue::fused_qkv",
    fused_qkv_backward_hook,
    setup_context=setup_fused_qkv_context
)

@fused_qkv_op.register_fake
def fused_qkv_fake(x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim, eps):
    B, N, C = x.shape
    q_proj = torch.empty(B, num_heads, N, head_dim, dtype=x.dtype, device=x.device)
    k_proj = torch.empty(B, num_heads, N, head_dim, dtype=x.dtype, device=x.device)
    v_proj = torch.empty(B, num_heads, N, head_dim, dtype=x.dtype, device=x.device)
    q_rsqrt = torch.empty(B, num_heads, N, 1, dtype=torch.float32, device=x.device)
    k_rsqrt = torch.empty(B, num_heads, N, 1, dtype=torch.float32, device=x.device)
    return q_proj, k_proj, v_proj, q_rsqrt, k_rsqrt


class FusedQKV(nn.Module):
    """
    Fused QKV Projection + Epilogue Module.
    Replaces: qkv_proj, QKRMSNorm, apply_rope, and q_proj scaling.
    """
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = nn.Parameter(torch.randn(num_heads, 1, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.constant_(self.qk_norm_weight, 1.0)
        nn.init.constant_(self.scales, math.log(self.head_dim ** -0.5))
        
    def forward(self, x: torch.Tensor, thetas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if thetas is None:
            raise ValueError("FusedQKV requires precomputed RoPE thetas (sin, cos).")
            
        sin, cos = thetas
        q_proj, k_proj, v_proj, _, _ = torch.ops.flash_prologue.fused_qkv(
            x, self.qkv_proj.weight, self.qk_norm_weight, self.scales, 
            sin, cos, self.num_heads, self.head_dim, self.eps
        )
        
        return q_proj, k_proj, v_proj

@fused_qkv_backward_op.register_fake
def fused_qkv_backward_fake(
    grad_q_proj, grad_k_proj, grad_v_proj, x, weight, rmsnorm_weight, scales, sin, cos, q_rsqrt, k_rsqrt, num_heads, head_dim, eps
):
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    grad_rmsnorm_weight = torch.empty_like(rmsnorm_weight)
    grad_scales = torch.empty_like(scales)
    return grad_x, grad_weight, grad_rmsnorm_weight, grad_scales
