import torch
import triton
import triton.language as tl
from typing import Tuple

# Set architecture targets
import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_N': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=5),
    ],
    key=['B', 'N', 'NUM_HEADS'],
)
@triton.jit
def qknorm_rope_scale_triton_kernel(
    qkv_ptr,
    rmsnorm_weight_ptr,
    scales_ptr,
    sin_ptr,
    cos_ptr,
    q_proj_ptr,
    k_proj_ptr,
    stride_qkv_b, stride_qkv_n,
    stride_q_b, stride_q_h, stride_q_n,
    stride_sin_b, stride_sin_n,
    B, N, NUM_HEADS, HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Sequence offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n[:, None] < N

    cols_first = tl.arange(0, HALF_DIM)
    cols_second = tl.arange(HALF_DIM, HEAD_DIM)

    # Base pointers
    qkv_base = qkv_ptr + pid_b * stride_qkv_b
    
    # 2D Load offsets
    # qkv has shape [B, N, 3 * NUM_HEADS * HEAD_DIM]
    qkv_n_offset = offs_n[:, None] * stride_qkv_n
    
    q_offset_1 = qkv_n_offset + (0 * NUM_HEADS * HEAD_DIM + pid_h * HEAD_DIM + cols_first)[None, :]
    q_offset_2 = qkv_n_offset + (0 * NUM_HEADS * HEAD_DIM + pid_h * HEAD_DIM + cols_second)[None, :]
    
    k_offset_1 = qkv_n_offset + (1 * NUM_HEADS * HEAD_DIM + pid_h * HEAD_DIM + cols_first)[None, :]
    k_offset_2 = qkv_n_offset + (1 * NUM_HEADS * HEAD_DIM + pid_h * HEAD_DIM + cols_second)[None, :]

    # Load Q, K (shape: BLOCK_N, HALF_DIM)
    q_1 = tl.load(qkv_base + q_offset_1, mask=mask_n, other=0.0).to(tl.float32)
    q_2 = tl.load(qkv_base + q_offset_2, mask=mask_n, other=0.0).to(tl.float32)
    
    k_1 = tl.load(qkv_base + k_offset_1, mask=mask_n, other=0.0).to(tl.float32)
    k_2 = tl.load(qkv_base + k_offset_2, mask=mask_n, other=0.0).to(tl.float32)

    # 1. Compute RMSNorm for Q and K along the last dimension (HEAD_DIM)
    q_var = (tl.sum(q_1 * q_1, axis=1) + tl.sum(q_2 * q_2, axis=1)) / HEAD_DIM
    q_rsqrt = tl.math.rsqrt(q_var + eps)[:, None] # Shape: (BLOCK_N, 1)
    
    k_var = (tl.sum(k_1 * k_1, axis=1) + tl.sum(k_2 * k_2, axis=1)) / HEAD_DIM
    k_rsqrt = tl.math.rsqrt(k_var + eps)[:, None] # Shape: (BLOCK_N, 1)

    # Load RMSNorm weights
    w_q1 = tl.load(rmsnorm_weight_ptr + cols_first).to(tl.float32)[None, :]
    w_q2 = tl.load(rmsnorm_weight_ptr + cols_second).to(tl.float32)[None, :]
    w_k1 = tl.load(rmsnorm_weight_ptr + HEAD_DIM + cols_first).to(tl.float32)[None, :]
    w_k2 = tl.load(rmsnorm_weight_ptr + HEAD_DIM + cols_second).to(tl.float32)[None, :]

    # Apply normalization
    q_norm1 = q_1 * q_rsqrt * w_q1
    q_norm2 = q_2 * q_rsqrt * w_q2
    k_norm1 = k_1 * k_rsqrt * w_k1
    k_norm2 = k_2 * k_rsqrt * w_k2

    # 2. RoPE
    sin_base = sin_ptr + pid_b * stride_sin_b
    cos_base = cos_ptr + pid_b * stride_sin_b
    
    sin_n_offset = offs_n[:, None] * stride_sin_n
    
    sin_1 = tl.load(sin_base + sin_n_offset + cols_first[None, :], mask=mask_n, other=0.0).to(tl.float32)
    sin_2 = tl.load(sin_base + sin_n_offset + cols_second[None, :], mask=mask_n, other=0.0).to(tl.float32)
    cos_1 = tl.load(cos_base + sin_n_offset + cols_first[None, :], mask=mask_n, other=0.0).to(tl.float32)
    cos_2 = tl.load(cos_base + sin_n_offset + cols_second[None, :], mask=mask_n, other=0.0).to(tl.float32)

    q_roped1 = q_norm1 * cos_1 - q_norm2 * sin_1
    q_roped2 = q_norm2 * cos_2 + q_norm1 * sin_2
    
    k_roped1 = k_norm1 * cos_1 - k_norm2 * sin_1
    k_roped2 = k_norm2 * cos_2 + k_norm1 * sin_2

    # 3. Scales for Q
    scale_val = tl.load(scales_ptr + pid_h).to(tl.float32)
    q_scale_factor = tl.math.exp(scale_val)
    
    q_final1 = q_roped1 * q_scale_factor
    q_final2 = q_roped2 * q_scale_factor

    # Out offsets
    out_base_offset = pid_b * stride_q_b + pid_h * stride_q_h
    out_n_offset = offs_n[:, None] * stride_q_n
    
    out_offset_first = out_base_offset + out_n_offset + cols_first[None, :]
    out_offset_second = out_base_offset + out_n_offset + cols_second[None, :]

    # Store outputs
    tl.store(q_proj_ptr + out_offset_first, q_final1, mask=mask_n)
    tl.store(q_proj_ptr + out_offset_second, q_final2, mask=mask_n)
    
    tl.store(k_proj_ptr + out_offset_first, k_roped1, mask=mask_n)
    tl.store(k_proj_ptr + out_offset_second, k_roped2, mask=mask_n)


def qknorm_rope_scale_triton(qkv, rmsnorm_weight, scales, sin, cos, eps=1e-5):
    # Get shape info
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]

    # Stride info
    stride_qkv_b, stride_qkv_n, _ = qkv.stride()
    
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=qkv.dtype, device=qkv.device)
    
    # Extract v_proj directly from qkv using a view + permute + contiguous in PyTorch
    v_proj = qkv.view(B, N, 3, num_heads, head_dim)[..., 2, :, :].permute(0, 2, 1, 3).contiguous()
    
    stride_q_b, stride_q_h, stride_q_n, _ = q_proj.stride()
    
    stride_sin_b, _, stride_sin_n, _ = sin.stride()

    # Launch Triton kernel
    grid = lambda meta: (B, num_heads, triton.cdiv(N, meta['BLOCK_N']))
    qknorm_rope_scale_triton_kernel[grid](
        qkv,
        rmsnorm_weight,
        scales,
        sin,
        cos,
        q_proj,
        k_proj,
        stride_qkv_b, stride_qkv_n,
        stride_q_b, stride_q_h, stride_q_n,
        stride_sin_b, stride_sin_n,
        B, N, num_heads, head_dim, head_dim // 2,
        eps
    )
    
    return q_proj, k_proj, v_proj

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


def qknorm_rope_scale_cute(
    qkv: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[-1]
    
    qkv_reshaped = qkv.view(B, N, 3, num_heads, head_dim)
    q_in = qkv_reshaped[..., 0, :, :].permute(0, 2, 1, 3).contiguous()
    k_in = qkv_reshaped[..., 1, :, :].permute(0, 2, 1, 3).contiguous()
    v_proj = qkv_reshaped[..., 2, :, :].permute(0, 2, 1, 3).contiguous()
    
    import flash_attn_prologue.fused_epilogue
    q_proj, k_proj = torch.ops.flash_prologue.qknorm_rope_scale_cute(
        q_in, k_in, rmsnorm_weight, scales, sin, cos, eps
    )
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
    # We call CuTe implementation in non-tracing modes on CUDA
    if qkv.device.type == "cuda":
        return qknorm_rope_scale_cute(qkv, rmsnorm_weight, scales, sin, cos, eps)
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
    q_proj, k_proj, v_proj = output
    ctx.save_for_backward(qkv, rmsnorm_weight, scales, sin, cos)
    ctx.eps = eps


def qknorm_rope_scale_backward_hook(ctx, grad_q, grad_k, grad_v):
    qkv, rmsnorm_weight, scales, sin, cos = ctx.saved_tensors
    eps = ctx.eps
    
    B, N, total_dim = qkv.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
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
    # Note: rmsnorm_weight shape is [2, 1, head_dim] or [2, head_dim]
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
    grad_qkv = torch.stack((grad_q_in, grad_k_in, grad_v_in), dim=2) # [B, H, 3, N, D]
    grad_qkv = grad_qkv.permute(0, 3, 2, 1, 4).reshape(B, N, 3 * num_heads * head_dim)
    
    return grad_qkv, grad_rmsnorm_weight, grad_scales, None, None, None


torch.library.register_autograd(
    "flash_prologue::qknorm_rope_scale",
    qknorm_rope_scale_backward_hook,
    setup_context=setup_qknorm_rope_scale_context
)
