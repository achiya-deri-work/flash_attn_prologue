import torch
import flash_attn_prologue.gemm_bwd
import math

def diagnose_dx():
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
    weight = truncated_normal((3 * num_heads * head_dim, hidden), std=0.3)
    
    grad_x_eager = grad_output @ weight
    grad_x_cute = torch.ops.flash_prologue.matmul_bwd_dx_cute(grad_output, weight)
    
    diff = (grad_x_cute - grad_x_eager).abs()
    mismatches = diff > 1e-1
    mismatch_indices = torch.nonzero(mismatches)
    
    print("Total mismatching elements:", mismatch_indices.shape[0])
    print("NaNs in cute:", torch.isnan(grad_x_cute).sum().item())
    
    if mismatch_indices.shape[0] > 0:
        # Find which CTA blocks have mismatches
        block_counts = {}
        for idx in mismatch_indices:
            r, c = idx.tolist()
            block_r = r // 64
            block_c = c // 64
            block_counts[(block_r, block_c)] = block_counts.get((block_r, block_c), 0) + 1
            
        print("Mismatches per 64x64 CTA block:")
        for block, count in sorted(block_counts.items())[:10]:
            print(f"  Block {block}: {count} mismatches")
            
        # Inspect first mismatching block details
        first_block = sorted(block_counts.keys())[0]
        br, bc = first_block
        print(f"\nDetails for Block ({br}, {bc}):")
        cute_block = grad_x_cute[br*64:(br+1)*64, bc*64:(bc+1)*64]
        eager_block = grad_x_eager[br*64:(br+1)*64, bc*64:(bc+1)*64]
        diff_block = diff[br*64:(br+1)*64, bc*64:(bc+1)*64]
        
        print("  Max diff in block:", diff_block.max().item())
        print("  NaNs in block:", torch.isnan(cute_block).sum().item())
        
        mismatch_in_block = torch.nonzero(diff_block > 1e-1)
        print(f"  First 20 mismatches in block ({br}, {bc}):")
        for i in range(min(20, mismatch_in_block.shape[0])):
            r_idx, c_idx = mismatch_in_block[i].tolist()
            print(f"    Coord ({r_idx}, {c_idx}): cute={cute_block[r_idx, c_idx].item()}, eager={eager_block[r_idx, c_idx].item()}")

if __name__ == "__main__":
    diagnose_dx()
