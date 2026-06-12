import torch
import traceback
import sys
from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op

@torch.compile(fullgraph=True, dynamic=False)
def compiled_fn(q, k, rmsnorm_weight, scales, sin, cos):
    return qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos)

def test_slicing_issue():
    print("================ Running Slicing Issue Test (Non-Contiguous) ================")
    
    # 1. Warmup with N=704
    q_704 = torch.randn(16, 4, 704, 64, dtype=torch.bfloat16, device='cuda')
    k_704 = torch.randn(16, 4, 704, 64, dtype=torch.bfloat16, device='cuda')
    rmsnorm_weight = torch.ones(2, 1, 64, dtype=torch.bfloat16, device='cuda')
    scales = torch.ones(4, 1, 1, dtype=torch.bfloat16, device='cuda')
    
    sin_full = torch.randn(16, 1, 704, 32, dtype=torch.float32, device='cuda')
    cos_full = torch.randn(16, 1, 704, 32, dtype=torch.float32, device='cuda')
    
    print("Warmup with N=704...")
    compiled_fn(q_704, k_704, rmsnorm_weight, scales, sin_full, cos_full)
    
    # 2. Run with N=576 using non-contiguous slice (simulate old slice_seq behavior)
    q_576 = torch.randn(16, 4, 576, 64, dtype=torch.bfloat16, device='cuda')
    k_576 = torch.randn(16, 4, 576, 64, dtype=torch.bfloat16, device='cuda')
    
    sin_sliced_nc = sin_full[:, :, :576]
    cos_sliced_nc = cos_full[:, :, :576]
    
    print(f"sin_sliced_nc shape: {sin_sliced_nc.shape}, stride: {sin_sliced_nc.stride()}")
    
    print("Calling compiled_fn for N=576 with non-contiguous thetas...")
    try:
        q_out, k_out = compiled_fn(q_576, k_576, rmsnorm_weight, scales, sin_sliced_nc, cos_sliced_nc)
        print("SUCCESS! Output shape:", q_out.shape)
    except Exception as e:
        print("EXPECTED FAILURE OCCURRED:")
        traceback.print_exc()

if __name__ == '__main__':
    test_slicing_issue()
