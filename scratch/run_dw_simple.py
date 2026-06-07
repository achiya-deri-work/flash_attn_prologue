import os
# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import torch
import cutlass.cute as cute
from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dw_kernel

def main():
    print("Compiling get_matmul_bwd_dw_kernel...")
    kernel = get_matmul_bwd_dw_kernel(
        num_warps=4,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K_STAGE=32,
        ab_stage=2,
        M_dim=3072,
        K_dim=16384,
        N_dim=512,
        maxrregcount=255,
        SPLIT_K=16
    )
    print("Compilation successful!")
    
    print("Creating inputs...")
    grad_output_t = torch.randn(16384, 3072, device="cuda", dtype=torch.bfloat16).t()
    x = torch.randn(16384, 512, device="cuda", dtype=torch.bfloat16)
    grad_weight = torch.zeros(3072, 512, device="cuda", dtype=torch.float32)
    
    print("Running kernel...")
    kernel(grad_output_t, x, grad_weight)
    torch.cuda.synchronize()
    print("Execution completed successfully!")

if __name__ == "__main__":
    main()
