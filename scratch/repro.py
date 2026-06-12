import torch
import sys
from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op

device = "cuda"

q_base = torch.randn(933376, device=device, dtype=torch.bfloat16)
q_strided = torch.as_strided(q_base, (16, 4, 76, 64), (58368, 64, 768, 1))

k_base = torch.randn(933376, device=device, dtype=torch.bfloat16)
k_strided = torch.as_strided(k_base, (16, 4, 76, 64), (58368, 64, 768, 1))

rmsnorm_weight = torch.randn(2, 1, 64, device=device, dtype=torch.float32)
scales = torch.randn(4, 1, 1, device=device, dtype=torch.float32)

sin_base = torch.randn(2432, device=device, dtype=torch.float32)
sin_strided = torch.as_strided(sin_base, (16, 1, 76, 32), (0, 0, 32, 1))

cos_base = torch.randn(2432, device=device, dtype=torch.float32)
cos_strided = torch.as_strided(cos_base, (16, 1, 76, 32), (0, 0, 32, 1))

@torch.compile(fullgraph=True, dynamic=False)
def compiled_fn(q, k, rmsnorm_weight, scales, sin, cos):
    return qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos)

print("Calling compiled_fn...")
try:
    q_out, k_out = compiled_fn(q_strided, k_strided, rmsnorm_weight, scales, sin_strided, cos_strided)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
