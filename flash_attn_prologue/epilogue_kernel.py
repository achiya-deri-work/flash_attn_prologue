import torch
from typing import Tuple

# Set architecture targets
import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

def rotate_half(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_backward(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def qknorm_rope_scale_eager(qkv, rmsnorm_weight, scales, sin, cos, eps=1e-5):
    # Vectorized PyTorch eager equivalent
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    
    # Reshape and split Q, K, V
    qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
    q, k, v = qkv_reshaped[:, :, 0], qkv_reshaped[:, :, 1], qkv_reshaped[:, :, 2]
    
    q_fp32 = q.float()
    k_fp32 = k.float()
    
    q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    
    w_q = rmsnorm_weight[0] if rmsnorm_weight.dim() == 3 else rmsnorm_weight[0:1]
    w_k = rmsnorm_weight[1] if rmsnorm_weight.dim() == 3 else rmsnorm_weight[1:2]
    
    q_normed = (q_fp32 * q_rsqrt).to(q.dtype) * w_q
    k_normed = (k_fp32 * k_rsqrt).to(k.dtype) * w_k
        
    q_normed_fp32 = q_normed.float()
    k_normed_fp32 = k_normed.float()
    
    q_roped_fp32 = (q_normed_fp32 * cos) + (rotate_half(q_normed_fp32) * sin)
    k_roped_fp32 = (k_normed_fp32 * cos) + (rotate_half(k_normed_fp32) * sin)
    
    q_proj = q_roped_fp32.to(q.dtype) * scales.float().exp().to(q.dtype)
    k_proj = k_roped_fp32.to(k.dtype)
    v_proj = v
    
    return q_proj, k_proj, v_proj


@torch.library.custom_op("flash_prologue::qknorm_rope_scale", mutates_args=())
def qknorm_rope_scale_op(
    qkv: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # We call CuTe implementation on CUDA, otherwise eager fallback
    if qkv.device.type == "cuda":
        B, N, total_dim = qkv.shape
        num_heads = scales.shape[0]
        head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
        
        qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim)
        q = qkv_reshaped[..., 0, :, :].permute(0, 2, 1, 3).contiguous()
        k = qkv_reshaped[..., 1, :, :].permute(0, 2, 1, 3).contiguous()
        v_proj = qkv_reshaped[..., 2, :, :].permute(0, 2, 1, 3).contiguous()
        
        # We call the CuTe forward op
        from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op
        q_proj, k_proj = qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
        return q_proj, k_proj, v_proj
    else:
        return qknorm_rope_scale_eager(qkv, rmsnorm_weight, scales, sin, cos, eps)


@qknorm_rope_scale_op.register_fake
def _qknorm_rope_scale_fake(qkv, rmsnorm_weight, scales, sin, cos, eps=1e-5):
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=qkv.dtype, device=qkv.device)
    v_proj = torch.empty((B, num_heads, N, head_dim), dtype=qkv.dtype, device=qkv.device)
    
    return q_proj, k_proj, v_proj


def setup_qknorm_rope_scale_context(ctx, inputs, output):
    qkv, rmsnorm_weight, scales, sin, cos, eps = inputs
    ctx.save_for_backward(qkv, rmsnorm_weight, scales, sin, cos)
    ctx.eps = eps


def qknorm_rope_scale_backward_hook(ctx, grad_q, grad_k, grad_v):
    qkv, rmsnorm_weight, scales, sin, cos = ctx.saved_tensors
    eps = ctx.eps
    
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    
    if qkv.device.type == "cuda":
        # Extract q, k from qkv
        qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim)
        q = qkv_reshaped[..., 0, :, :].permute(0, 2, 1, 3).contiguous()
        k = qkv_reshaped[..., 1, :, :].permute(0, 2, 1, 3).contiguous()
        
        # We call the CuTe backward op
        from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_backward_op
        dq, dk, dw, ds = qknorm_rope_scale_cute_backward_op(
            q, k, rmsnorm_weight, scales, sin, cos, grad_q, grad_k, eps
        )
        
        # Squeeze grad_rmsnorm_weight to match shape of rmsnorm_weight if necessary
        if rmsnorm_weight.dim() == 2:
            dw = dw.squeeze(1)
            
        # Reconstruct grad_qkv
        grad_qkv = torch.stack((dq, dk, grad_v), dim=2)  # [B, H, 3, N, D]
        grad_qkv = grad_qkv.permute(0, 3, 2, 1, 4).reshape(B, N, 3 * num_heads * head_dim)
        
        return grad_qkv, dw, ds, None, None, None
    else:
        half_dim = head_dim // 2
        
        # 1. Recompute forward normalized states (using eager ops for simple compilation)
        qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
        q, k, v = qkv_reshaped[:, :, 0], qkv_reshaped[:, :, 1], qkv_reshaped[:, :, 2]
        
        q_fp32 = q.float()
        k_fp32 = k.float()
        
        q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        
        w_q = rmsnorm_weight[0] if rmsnorm_weight.dim() == 3 else rmsnorm_weight[0:1]
        w_k = rmsnorm_weight[1] if rmsnorm_weight.dim() == 3 else rmsnorm_weight[1:2]
        
        q_normed = (q_fp32 * q_rsqrt).to(q.dtype) * w_q
        k_normed = (k_fp32 * k_rsqrt).to(k.dtype) * w_k
        
        # 2. Gradient of Scale and RoPE backward
        scale_factor_bf16 = scales.float().exp().to(qkv.dtype)
        grad_q_scaled = grad_q * scale_factor_bf16
        grad_q_normed = (grad_q_scaled.float() * cos) + rotate_half_backward(grad_q_scaled.float() * sin)
        grad_k_normed = (grad_k.float() * cos) + rotate_half_backward(grad_k.float() * sin)
        
        grad_q_normed = grad_q_normed.to(qkv.dtype)
        grad_k_normed = grad_k_normed.to(qkv.dtype)
        
        # 3. Gradient of RMSNorm weights
        grad_rmsnorm_weight_q = (grad_q_normed * (q_fp32 * q_rsqrt).to(qkv.dtype)).sum(dim=(0, 1, 2)).view(1, 1, head_dim)
        grad_rmsnorm_weight_k = (grad_k_normed * (k_fp32 * k_rsqrt).to(qkv.dtype)).sum(dim=(0, 1, 2)).view(1, 1, head_dim)
        grad_rmsnorm_weight = torch.cat((grad_rmsnorm_weight_q, grad_rmsnorm_weight_k), dim=0)
        
        if rmsnorm_weight.dim() == 2:
            grad_rmsnorm_weight = grad_rmsnorm_weight.squeeze(1)
            
        # 4. Gradient of input Q / K
        grad_q_prenorm = grad_q_normed * w_q
        grad_k_prenorm = grad_k_normed * w_k
        
        q_sum = (grad_q_prenorm.float() * q_fp32).sum(dim=-1, keepdim=True)
        grad_q_in = q_rsqrt * (grad_q_prenorm.float() - q_fp32 * q_sum * (q_rsqrt.pow(2) / head_dim))
        
        k_sum = (grad_k_prenorm.float() * k_fp32).sum(dim=-1, keepdim=True)
        grad_k_in = k_rsqrt * (grad_k_prenorm.float() - k_fp32 * k_sum * (k_rsqrt.pow(2) / head_dim))
        
        grad_q_in = grad_q_in.to(qkv.dtype)
        grad_k_in = grad_k_in.to(qkv.dtype)
        grad_v_in = grad_v
        
        # 5. Gradient of scales parameter
        q_normed_fp32 = q_normed.float()
        q_roped_fp32 = (q_normed_fp32 * cos) + (rotate_half(q_normed_fp32) * sin)
        q_roped = q_roped_fp32.to(qkv.dtype)
        
        grad_scales = (grad_q * q_roped).float().sum(dim=(0, 2, 3)).view(num_heads, 1, 1).to(scales.dtype)
        grad_scales = (grad_scales.float() * scale_factor_bf16.float()).to(scales.dtype)
        
        # Reconstruct grad_qkv
        grad_qkv = torch.stack((grad_q_in, grad_k_in, grad_v_in), dim=2)  # [B, H, 3, N, D]
        grad_qkv = grad_qkv.permute(0, 3, 2, 1, 4).reshape(B, N, 3 * num_heads * head_dim)
        
        return grad_qkv, grad_rmsnorm_weight, grad_scales, None, None, None


torch.library.register_autograd(
    "flash_prologue::qknorm_rope_scale",
    qknorm_rope_scale_backward_hook,
    setup_context=setup_qknorm_rope_scale_context
)
