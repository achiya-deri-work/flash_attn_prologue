import torch
import torch.nn as nn
from flash_attn_prologue.layers.RoPE import apply_rope
from flash_attn_prologue.fused_epilogue import qknorm_rope_scale_cute_op, qknorm_rope_scale_cute_inverse_op
import gc

B = 32
H = 16
N = 512
D = 64
device = "cuda"
dtype = torch.bfloat16
eps = 1e-5

def setup_inputs():
    x = torch.randn(B, N, H * D, device=device, dtype=dtype)
    w_q = torch.randn(H * D, H * D, device=device, dtype=dtype, requires_grad=True)
    w_k = torch.randn(H * D, H * D, device=device, dtype=dtype, requires_grad=True)
    
    rmsnorm_weight = nn.Parameter(torch.ones(2, 1, D, device=device, dtype=dtype))
    scales = nn.Parameter(torch.full((H, 1, 1), -2.0, device=device, dtype=dtype))
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, D, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / D))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    cos = cos_val.unsqueeze(1).contiguous().to(device=device, dtype=torch.float32)
    return x, w_q, w_k, rmsnorm_weight, scales, sin, cos

def simulate_forward_standard():
    print("\n--- Standard Epilogue Memory Step-by-Step ---")
    x, w_q, w_k, rmsnorm_weight, scales, sin, cos = setup_inputs()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"1. Setup complete: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q = (x @ w_q).view(B, N, H, D).transpose(1, 2)
    k = (x @ w_k).view(B, N, H, D).transpose(1, 2)
    print(f"2. Matmuls complete (q, k created): {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q_out, k_out = qknorm_rope_scale_cute_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
    print(f"3. Epilogue complete (q_out, k_out created): {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q = None
    k = None
    x = None
    gc.collect()
    print(f"4. Inputs deleted & GC run: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    return torch.cuda.memory_allocated()

def simulate_forward_inverse():
    print("\n--- Inverse Epilogue Memory Step-by-Step ---")
    x, w_q, w_k, rmsnorm_weight, scales, sin, cos = setup_inputs()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"1. Setup complete: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q = (x @ w_q).view(B, N, H, D).transpose(1, 2)
    k = (x @ w_k).view(B, N, H, D).transpose(1, 2)
    print(f"2. Matmuls complete (q, k created): {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q_out, k_out, rms_q, rms_k = qknorm_rope_scale_cute_inverse_op(q, k, rmsnorm_weight, scales, sin, cos, eps)
    print(f"3. Epilogue complete (q_out, k_out, rms created): {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    
    q = None
    k = None
    x = None
    gc.collect()
    print(f"4. Inputs deleted & GC run: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")
    return torch.cuda.memory_allocated()

if __name__ == "__main__":
    simulate_forward_standard()
    simulate_forward_inverse()
