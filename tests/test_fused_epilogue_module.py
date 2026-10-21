import math
import pytest
import torch
import torch.nn as nn
from flash_attn_prologue.layers.RMSNorm import RMSNormRoPEScale
from flash_attn_prologue.layers.RoPE import apply_rope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@pytest.mark.parametrize("N", [64, 100, 128, 512])
def test_fused_epilogue_module_parity(N):
    B = 32
    num_heads = 16
    head_dim = 64
    device = "cuda"

    # 1. Instantiate the module
    module = RMSNormRoPEScale(num_heads=num_heads, head_dim=head_dim).to(device=device, dtype=torch.bfloat16)

    # Verify that parameters retain FP32 master precision
    assert module.rmsnorm_weight.dtype == torch.float32
    assert module.scales.dtype == torch.float32

    # 2. Inputs
    q = torch.randn(B, num_heads, N, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, num_heads, N, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Precompute sin/cos frequencies in Float32
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)

    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)

    # 3. Eager Baseline clones
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)

    # 4. Custom Op execution (in BFloat16)
    q_out, k_out = module(q, k, sin, cos)

    # 5. Eager baseline execution (in Float32 internally for stability, then back to BFloat16)
    eps = module.eps
    q_normed = q_ref.float() * torch.rsqrt(q_ref.float().pow(2).mean(-1, keepdim=True) + eps) * module.rmsnorm_weight[0, 0].float()
    k_normed = k_ref.float() * torch.rsqrt(k_ref.float().pow(2).mean(-1, keepdim=True) + eps) * module.rmsnorm_weight[1, 0].float()

    # Apply RoPE (requires float32 sin/cos)
    q_rotated = apply_rope(q_normed, sin, cos)
    k_rotated = apply_rope(k_normed, sin, cos)

    # Scale Q by scales
    q_ref_final = (q_rotated * module.scales.float().exp()).to(torch.bfloat16)
    k_ref_final = k_rotated.to(torch.bfloat16)

    # 6. Verify Forward Parity
    torch.testing.assert_close(q_out, q_ref_final, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_out, k_ref_final, rtol=1e-2, atol=1e-1)
    print("✓ Forward parity matches eagerness baseline!")

    # 7. Backward pass and verify gradient parity
    loss_c = q_out.sum() + k_out.sum()
    loss_c.backward()

    loss_r = q_ref_final.sum() + k_ref_final.sum()
    loss_r.backward()

    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-1)
    print("✓ Backward inputs gradient parity matches eager!")
    print("SUCCESS: RMSNormRoPEScale module correctness tests passed!")

@pytest.mark.parametrize("N", [64, 100, 128, 512])
def test_fused_epilogue_module_compiled(N):
    B = 32
    num_heads = 16
    head_dim = 64
    device = "cuda"

    # 1. Instantiate the module
    module = RMSNormRoPEScale(num_heads=num_heads, head_dim=head_dim).to(device=device, dtype=torch.bfloat16)

    # 2. Compile module using torch.compile
    compiled_module = torch.compile(module, fullgraph=True)

    # 3. Inputs
    q = torch.randn(B, num_heads, N, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, num_heads, N, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Precompute sin/cos frequencies in Float32
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)

    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)

    # Clones for eager reference
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)

    # 4. Compiled execution (in BFloat16)
    q_out, k_out = compiled_module(q, k, sin, cos)

    # 5. Eager reference execution
    eps = module.eps
    q_normed = q_ref.float() * torch.rsqrt(q_ref.float().pow(2).mean(-1, keepdim=True) + eps) * module.rmsnorm_weight[0, 0].float()
    k_normed = k_ref.float() * torch.rsqrt(k_ref.float().pow(2).mean(-1, keepdim=True) + eps) * module.rmsnorm_weight[1, 0].float()

    q_rotated = apply_rope(q_normed, sin, cos)
    k_rotated = apply_rope(k_normed, sin, cos)

    q_ref_final = (q_rotated * module.scales.float().exp()).to(torch.bfloat16)
    k_ref_final = k_rotated.to(torch.bfloat16)

    # Verify compiled forward output matches reference
    torch.testing.assert_close(q_out, q_ref_final, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k_out, k_ref_final, rtol=1e-2, atol=1e-1)
    print("✓ Compiled Forward parity matches reference!")

    # 6. Backward pass on compiled outputs
    loss_c = q_out.sum() + k_out.sum()
    loss_c.backward()

    loss_r = q_ref_final.sum() + k_ref_final.sum()
    loss_r.backward()

    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-1)
    print("✓ Compiled Backward gradient parity matches reference!")
    print("SUCCESS: RMSNormRoPEScale torch.compile test passed!")

if __name__ == "__main__":
    test_fused_epilogue_module_parity()
    test_fused_epilogue_module_compiled()
