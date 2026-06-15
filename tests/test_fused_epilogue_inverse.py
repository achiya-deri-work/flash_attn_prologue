import torch
import torch.nn as nn
from flash_attn_prologue.layers.RoPE import apply_rope
import pytest

def test_fused_epilogue_inverse_correctness():
    B = 4
    N = 128
    num_heads = 4
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    # 1. Instantiate/Define parameters
    rmsnorm_weight = nn.Parameter(torch.ones(2, 1, head_dim, device=device, dtype=dtype))
    scales = nn.Parameter(torch.full((num_heads, 1, 1), -2.0, device=device, dtype=dtype))

    # 2. Inputs
    q = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Precompute sin/cos frequencies
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)

    # 3. Clones for different paths
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    rmsnorm_weight_ref = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_ref = scales.clone().detach().requires_grad_(True)

    q_std = q.clone().detach().requires_grad_(True)
    k_std = k.clone().detach().requires_grad_(True)
    rmsnorm_weight_std = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_std = scales.clone().detach().requires_grad_(True)

    # --- Path A: Eager Baseline ---
    q_normed = q_ref.float() * torch.rsqrt(q_ref.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight_ref[0, 0].float()
    k_normed = k_ref.float() * torch.rsqrt(k_ref.float().pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight_ref[1, 0].float()
    q_rotated = apply_rope(q_normed.to(dtype), sin, cos)
    k_rotated = apply_rope(k_normed.to(dtype), sin, cos)
    q_out_ref = q_rotated * scales_ref.float().exp().to(dtype)
    k_out_ref = k_rotated

    # --- Path B: Standard Operator ---
    q_out_std, k_out_std = torch.ops.flash_prologue.qknorm_rope_scale_cute(
        q_std, k_std, rmsnorm_weight_std, scales_std, sin, cos, eps
    )

    # --- Path C: Inverse Operator ---
    q_out_inv, k_out_inv, rms_q, rms_k = torch.ops.flash_prologue.qknorm_rope_scale_cute_inverse(
        q, k, rmsnorm_weight, scales, sin, cos, eps
    )

    # 4. Verify Forward Parity
    torch.testing.assert_close(q_out_std, q_out_ref, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_out_std, k_out_ref, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(q_out_inv, q_out_ref, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_out_inv, k_out_ref, rtol=1e-2, atol=1e-1)
    print("✓ Forward outputs parity matched!")

    # 5. Backward Pass
    # Grads of Path A
    loss_ref = q_out_ref.sum() + k_out_ref.sum()
    loss_ref.backward()

    # Grads of Path B
    loss_std = q_out_std.sum() + k_out_std.sum()
    loss_std.backward()

    # Grads of Path C
    loss_inv = q_out_inv.sum() + k_out_inv.sum()
    loss_inv.backward()

    # 6. Verify Gradient Parity
    # Check inputs (q, k)
    torch.testing.assert_close(q.grad, q_std.grad, rtol=1e-2, atol=2e-1)
    torch.testing.assert_close(k.grad, k_std.grad, rtol=1e-2, atol=2e-1)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=2e-1)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=2e-1)

    # Check weights (rmsnorm_weight)
    torch.testing.assert_close(rmsnorm_weight.grad, rmsnorm_weight_std.grad, rtol=1e-2, atol=2.5e-1)
    torch.testing.assert_close(rmsnorm_weight.grad, rmsnorm_weight_ref.grad, rtol=1e-2, atol=5e-1)

    # Check scales
    torch.testing.assert_close(scales.grad, scales_std.grad, rtol=1e-2, atol=2e-1)
    torch.testing.assert_close(scales.grad, scales_ref.grad, rtol=1e-2, atol=3e-1)
    print("✓ Backward gradients parity matched!")

if __name__ == "__main__":
    test_fused_epilogue_inverse_correctness()
