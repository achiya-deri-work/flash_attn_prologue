import torch
import flash_attn_prologue.gemm_bwd
import math

def diagnose_dw():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    def truncated_normal(shape, std=0.3):
        u = torch.rand(shape, device=device, dtype=torch.float32)
        u = 0.0227501 + u * (0.9772499 - 0.0227501)
        x_val = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0) * std
        return x_val.to(dtype)
        
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    grad_output = truncated_normal((B * N, 3 * num_heads * head_dim), std=0.3)
    x = truncated_normal((B * N, hidden), std=0.3)
    
    grad_weight_eager = grad_output.t() @ x
    print("Running dw...")
    grad_weight_cute = torch.ops.flash_prologue.matmul_bwd_dw_cute(grad_output.t(), x)
    
    diff = (grad_weight_cute - grad_weight_eager).abs()
    mismatches = diff > 0.3
    mismatch_indices = torch.nonzero(mismatches)
    
    print("Total mismatching elements for dw:", mismatch_indices.shape[0])
    print("NaNs in cute for dw:", torch.isnan(grad_weight_cute).sum().item())
    
    if mismatch_indices.shape[0] > 0:
        print("First 20 mismatch indices for dw:")
        for i in range(min(20, mismatch_indices.shape[0])):
            r, c = mismatch_indices[i].tolist()
            print(f"  Coord ({r}, {c}): cute={grad_weight_cute[r, c].item()}, eager={grad_weight_eager[r, c].item()}")

if __name__ == "__main__":
    diagnose_dw()
