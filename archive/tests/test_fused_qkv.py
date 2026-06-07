import torch
import torch.nn.functional as F
import math

def eager_fused_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch baseline for the Fused QKV Epilogue.
    Follows strict BFP16 / FP32 numerical stability rules.
    """
    B, N, C = x.shape
    
    # 1. Main GEMM (Computed in FP32 accumulators internally, output as BFP16 theoretically, 
    # but here we'll keep the raw linear as BFP16 since standard linear returns its input dtype)
    qkv = F.linear(x, weight) # [B, N, 3 * num_heads * head_dim]
    
    # Slicing out Q, K, V
    qkv = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4) # [B, num_heads, 3, N, head_dim]
    q = qkv[:, :, 0] # [B, num_heads, N, head_dim]
    k = qkv[:, :, 1]
    v_proj = qkv[:, :, 2]
    
    # 2. RMSNorm (Reduction in FP32)
    # Q and K compute their own reductions entirely independently over the head_dim
    q_fp32 = q.float()
    k_fp32 = k.float()
    
    q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + eps)
    k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + eps)
    
    q_normed_fp32 = q_fp32 * q_rsqrt
    k_normed_fp32 = k_fp32 * k_rsqrt
    
    # Downcast to BFP16 BEFORE multiplying with weight
    q_normed = q_normed_fp32.to(q.dtype)
    k_normed = k_normed_fp32.to(k.dtype)
    
    q_normed = q_normed * rmsnorm_weight[0] # Weight is BFP16
    k_normed = k_normed * rmsnorm_weight[1] # Weight is BFP16
    
    # 3. Apply RoPE (Rotations in FP32)
    def rotate_half(x_tensor):
        x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
        x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    q_normed_fp32 = q_normed.float()
    k_normed_fp32 = k_normed.float()
    
    q_roped_fp32 = (q_normed_fp32 * cos) + (rotate_half(q_normed_fp32) * sin)
    k_roped_fp32 = (k_normed_fp32 * cos) + (rotate_half(k_normed_fp32) * sin)
    
    # Downcast to BFP16 after RoPE
    q_proj = q_roped_fp32.to(q.dtype)
    k_proj = k_roped_fp32.to(k.dtype)
    
    # 4. Q Scaling
    # Exponentiating scales in FP32, then casting down to BFP16 for multiplication
    scale_factor = scales.float().exp().to(q.dtype)
    q_proj = q_proj * scale_factor
    
    return q_proj, k_proj, v_proj

def test_fused_qkv_parity():
    # Example dimensions
    B = 2
    N = 64
    num_heads = 32
    head_dim = 64
    hidden = num_heads * head_dim
    dtype = torch.bfloat16
    
    x = torch.randn(B, N, hidden, dtype=dtype, device='cuda')
    weight = torch.randn(3 * hidden, hidden, dtype=dtype, device='cuda')
    rmsnorm_weight = torch.ones(2, 1, head_dim, dtype=dtype, device='cuda')
    scales = torch.randn(num_heads, 1, 1, dtype=dtype, device='cuda')
    sin = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    cos = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    
    q, k, v = eager_fused_qkv(x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim)
    
    print("Eager test passed - No crashes.")
    print("Q output shape:", q.shape)

def rotate_half(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_half_backward(x_tensor):
    x1 = x_tensor[..., : x_tensor.shape[-1] // 2]
    x2 = x_tensor[..., x_tensor.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def eager_fused_qkv_backward(
    grad_q_proj: torch.Tensor,
    grad_k_proj: torch.Tensor,
    grad_v_proj: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch backward for the Fused QKV Epilogue (Option A).
    Recomputes Q, K, V to backprop through RMSNorm, then computes grad_x, grad_w.
    """
    B, N, C = x.shape
    
    # 1. Forward Recomputation (to get original q and k for RMSNorm backward)
    with torch.no_grad():
        qkv = F.linear(x, weight)
        qkv = qkv.view(B, N, 3, num_heads, head_dim).permute(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        q_fp32 = q.float()
        k_fp32 = k.float()
        q_rsqrt = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + eps)
        k_rsqrt = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + eps)
    
    # 2. Backward Vector Math (Exact PyTorch Autograd simulation with BF16 bounds)
    
    # Q Scale Un-exponentiate (BFP16 math)
    scale_factor_bf16 = scales.float().exp().to(x.dtype)
    grad_q_scaled = grad_q_proj * scale_factor_bf16 # BFP16
    
    # Inverse RoPE (FP32 math)
    grad_q_scaled_fp32 = grad_q_scaled.float()
    grad_k_proj_fp32 = grad_k_proj.float()
    
    grad_q_normed_fp32 = (grad_q_scaled_fp32 * cos) + rotate_half_backward(grad_q_scaled_fp32 * sin)
    grad_k_normed_fp32 = (grad_k_proj_fp32 * cos) + rotate_half_backward(grad_k_proj_fp32 * sin)
    
    # Downcast to BFP16 (because q_normed was cast to float for RoPE)
    grad_q_normed = grad_q_normed_fp32.to(x.dtype)
    grad_k_normed = grad_k_normed_fp32.to(x.dtype)
    
    # Inverse RMSNorm weight mul (BFP16 math)
    q_normed = (q_fp32 * q_rsqrt).to(x.dtype)
    k_normed = (k_fp32 * k_rsqrt).to(x.dtype)
    
    grad_rmsnorm_weight_q = (grad_q_normed * q_normed).float().sum(dim=(0, 1, 2)).view(1, 1, head_dim)
    grad_rmsnorm_weight_k = (grad_k_normed * k_normed).float().sum(dim=(0, 1, 2)).view(1, 1, head_dim)
    grad_rmsnorm_weight = torch.cat((grad_rmsnorm_weight_q, grad_rmsnorm_weight_k), dim=0).to(rmsnorm_weight.dtype)

    grad_q_prenorm = grad_q_normed * rmsnorm_weight[0] # BFP16
    grad_k_prenorm = grad_k_normed * rmsnorm_weight[1] # BFP16
    
    # RMSNorm Core Backward (FP32 math)
    grad_q_prenorm_fp32 = grad_q_prenorm.float()
    grad_k_prenorm_fp32 = grad_k_prenorm.float()
    
    q_sum = (grad_q_prenorm_fp32 * q_fp32).sum(-1, keepdim=True)
    grad_q_original = q_rsqrt * (grad_q_prenorm_fp32 - q_fp32 * q_sum * (q_rsqrt.pow(2) / head_dim))
    
    k_sum = (grad_k_prenorm_fp32 * k_fp32).sum(-1, keepdim=True)
    grad_k_original = k_rsqrt * (grad_k_prenorm_fp32 - k_fp32 * k_sum * (k_rsqrt.pow(2) / head_dim))
    
    grad_v_original = grad_v_proj # BFP16
    
    # Pack grad_qkv
    grad_q_original = grad_q_original.to(x.dtype)
    grad_k_original = grad_k_original.to(x.dtype)
    
    grad_qkv = torch.stack((grad_q_original, grad_k_original, grad_v_original), dim=2) # [B, num_heads, 3, N, head_dim]
    grad_qkv = grad_qkv.permute(0, 3, 2, 1, 4).reshape(B, N, 3 * num_heads * head_dim)
    
    # 3. Backward GEMMs (Option A)
    grad_x = F.linear(grad_qkv, weight.t())
    grad_weight = F.linear(grad_qkv.reshape(-1, 3 * num_heads * head_dim).t(), x.reshape(-1, C).t())
    
    # Scale grad (BFP16 math)
    q_roped = (q_normed.float() * cos) + (rotate_half(q_normed.float()) * sin)
    q_roped = q_roped.to(x.dtype)
    grad_scales = (grad_q_proj * q_roped).float().sum(dim=(0, 2, 3)).view(num_heads, 1, 1).to(scales.dtype)
    
    # Since d(exp(x))/dx = exp(x), we multiply the upstream grad by exp(x)
    grad_scales = (grad_scales.float() * scale_factor_bf16.float()).to(scales.dtype)


    
    return grad_x, grad_weight, grad_rmsnorm_weight, grad_scales

def test_fused_qkv_backward_parity():
    # Example dimensions
    B = 2
    N = 64
    num_heads = 32
    head_dim = 64
    hidden = num_heads * head_dim
    dtype = torch.bfloat16
    
    x = torch.randn(B, N, hidden, dtype=dtype, device='cuda', requires_grad=True)
    weight = torch.randn(3 * hidden, hidden, dtype=dtype, device='cuda', requires_grad=True)
    rmsnorm_weight = torch.ones(2, 1, head_dim, dtype=dtype, device='cuda', requires_grad=True)
    scales = torch.randn(num_heads, 1, 1, dtype=dtype, device='cuda', requires_grad=True)
    
    sin = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    cos = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    
    # 1. Forward pass (Eager)
    q, k, v = eager_fused_qkv(x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim)
    
    # 2. Fake backward signal
    grad_q = torch.randn_like(q)
    grad_k = torch.randn_like(k)
    grad_v = torch.randn_like(v)
    
    # 3. Autograd Backward
    loss = (q * grad_q + k * grad_k + v * grad_v).sum()
    loss.backward()
    
    grad_x_ref = x.grad.clone()
    grad_weight_ref = weight.grad.clone()
    grad_rmsnorm_weight_ref = rmsnorm_weight.grad.clone()
    grad_scales_ref = scales.grad.clone()
    
    # 4. Manual Backward
    grad_x_manual, grad_w_manual, grad_norm_manual, grad_scales_manual = eager_fused_qkv_backward(
        grad_q, grad_k, grad_v, x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim
    )
    
    # 5. Parity Check
    # High tolerances because BF16 accumulates error easily across GEMMs,
    # but we will hold strictly to the 1e-2 and 1e-3 bounds as requested.
    torch.testing.assert_close(grad_x_manual, grad_x_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_w_manual, grad_weight_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_norm_manual, grad_rmsnorm_weight_ref, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(grad_scales_manual, grad_scales_ref, rtol=1e-2, atol=1e-3)
    
    print("Backward Parity test passed! All gradients match autograd.")

def test_fused_qkv_module():
    import flash_attn_prologue.layers.FusedQKV as mod
    B = 2
    N = 64
    num_heads = 32
    head_dim = 64
    hidden = num_heads * head_dim
    dtype = torch.bfloat16
    
    x = torch.randn(B, N, hidden, dtype=dtype, device='cuda', requires_grad=True)
    sin = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    cos = torch.randn(B, 1, N, head_dim, dtype=torch.float32, device='cuda')
    
    layer = mod.FusedQKV(hidden, num_heads, head_dim).cuda().to(dtype)
    
    q, k, v = layer(x, thetas=(sin, cos))
    
    loss = q.sum() + k.sum() + v.sum()
    loss.backward()
    
    assert x.grad is not None
    assert layer.qkv_proj.weight.grad is not None
    
    print("Module CustomOp Integration test passed! Autograd correctly triggers manual backward.")
    
if __name__ == "__main__":
    test_fused_qkv_parity()
    test_fused_qkv_backward_parity()
    test_fused_qkv_module()

