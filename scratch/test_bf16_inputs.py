import torch
import torch.nn as nn
from flash_attn_prologue.fused_epilogue import get_fused_epilogue_kernel

B = 4
H = 4
N = 128
D = 64
device = "cuda"
dtype = torch.bfloat16

def test_bf16_inputs():
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    rmsnorm_weight = torch.ones(2, 1, D, device=device, dtype=dtype)
    scales = torch.full((H, 1, 1), -2.0, device=device, dtype=dtype)
    
    # Precompute sin/cos in BF16
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, D, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / D))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=dtype)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=dtype)
    
    q_strides = tuple(q.stride())
    k_strides = tuple(k.stride())
    sin_strides = tuple(sin.stride())
    cos_strides = tuple(cos.stride())
    
    print("Compiling JIT kernel with BFloat16 inputs...")
    kernel = get_fused_epilogue_kernel(
        num_warps=4,
        BLOCK_M=64,
        NUM_HEADS=H,
        HEAD_DIM=D,
        HEAD_GROUP=1,
        ab_stage=1,
        q_strides=q_strides,
        k_strides=k_strides,
        sin_strides=sin_strides,
        cos_strides=cos_strides,
        maxrregcount=255,
        B=B,
        N=N,
        sin_shape=tuple(sin.shape),
        cos_shape=tuple(cos.shape)
    )
    
    q_out = q.clone()
    k_out = k.clone()
    
    print("Launching JIT kernel...")
    # Pass BFloat16 inputs directly!
    kernel(q_out, k_out, rmsnorm_weight, scales, sin, cos)
    torch.cuda.synchronize()
    print("✓ Success! Kernel ran and synchronized.")

if __name__ == "__main__":
    test_bf16_inputs()
