import os
# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import torch
import cutlass.cute as cute
from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dx_kernel

def main():
    print("Compiling get_matmul_bwd_dx_kernel...")
    kernel = get_matmul_bwd_dx_kernel(
        num_warps=4,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K_STAGE=32,
        ab_stage=3,
        K_dim=3072,
        N_dim=512,
        maxrregcount=255
    )
    print("Compilation successful!")
    
    print("Creating inputs...")
    grad_output = torch.randn(16384, 3072, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3072, 512, device="cuda", dtype=torch.bfloat16)
    grad_x = torch.empty(16384, 512, device="cuda", dtype=torch.bfloat16)
    
    print("Running kernel...")
    kernel(grad_output, weight, grad_x)
    torch.cuda.synchronize()
    print("Execution completed successfully!")

if __name__ == "__main__":
    main()
