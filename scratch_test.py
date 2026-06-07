import torch
from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op
from flash_attn_prologue.epilogue_kernel import qknorm_rope_scale_eager
import os

B = 2
N = 16
num_heads = 4
head_dim = 64
device = "cuda"
dtype = torch.bfloat16

q = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype)
k = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype)
rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32)
scales = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32)

print("PyTorch actual scales:")
for h in range(num_heads):
    print(f"  head {h}: {scales[h, 0, 0].item()}")

# Generate simple cos/sin
sin = torch.zeros(B, 1, N, head_dim, device=device, dtype=torch.float32)
cos = torch.ones(B, 1, N, head_dim, device=device, dtype=torch.float32)

# Setup inputs for Eager
qkv_e = torch.stack((q, k, torch.zeros_like(q)), dim=2).permute(0, 3, 2, 1, 4).reshape(B, N, -1).requires_grad_(True)
rms_e = rmsnorm_weight.clone().detach().requires_grad_(True)
s_e = scales.clone().detach().requires_grad_(True)

q_e, k_e, _ = qknorm_rope_scale_eager(qkv_e, rms_e, s_e, sin, cos)
loss_e = q_e.sum() + k_e.sum()
loss_e.backward()

# Setup inputs for CuTe
q_c_in = q.clone().detach().requires_grad_(True)
k_c_in = k.clone().detach().requires_grad_(True)
rms_c = rmsnorm_weight.clone().detach().requires_grad_(True)
s_c = scales.clone().detach().requires_grad_(True)

q_c, k_c = torch.ops.flash_prologue.qknorm_rope_scale_cute(q_c_in, k_c_in, rms_c, s_c, sin, cos)
loss_c = q_c.sum() + k_c.sum()
loss_c.backward()

print("\n--- Compare scales gradient ---")
print("Eager s.grad:")
print(s_e.grad.squeeze())
print("CuTe s.grad:")
print(s_c.grad.squeeze())
