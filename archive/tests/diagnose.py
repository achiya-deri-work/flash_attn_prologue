import torch
import flash_attn_prologue.gemm_bwd

def diagnose():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Helper for truncated normal on [-2*std, 2*std]
    def truncated_normal(shape, std=0.3):
        import math
        u = torch.rand(shape, device=device, dtype=torch.float32)
        u = 0.0227501 + u * (0.9772499 - 0.0227501)
        x_val = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0) * std
        return x_val.to(dtype)
        
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    grad_output = truncated_normal((B * N, 3 * num_heads * head_dim), std=0.3)
    weight = truncated_normal((3 * num_heads * head_dim, hidden), std=0.3)
    x = truncated_normal((B * N, hidden), std=0.3)
    
    grad_x_eager = grad_output @ weight
    grad_weight_eager = grad_output.t() @ x
    
    print("Running dx...")
    grad_x_cute = torch.ops.flash_prologue.matmul_bwd_dx_cute(grad_output, weight)
    print("dx diff:", (grad_x_cute - grad_x_eager).abs().max().item())
    
    print("Running dw...")
    grad_weight_cute = torch.ops.flash_prologue.matmul_bwd_dw_cute(grad_output.t(), x)
    print("dw diff:", (grad_weight_cute - grad_weight_eager).abs().max().item())
    
    mismatches = (grad_weight_cute - grad_weight_eager).abs() > 1e-1
    mismatch_indices = torch.nonzero(mismatches)
    print("Total mismatching elements:", mismatch_indices.shape[0])
    if mismatch_indices.shape[0] > 0:
        print("Mismatches per block:")
        block_counts = {}
        for idx in mismatch_indices:
            r, c = idx.tolist()
            block_r = r // 64
            block_c = c // 64
            block_counts[(block_r, block_c)] = block_counts.get((block_r, block_c), 0) + 1
        for block, count in sorted(block_counts.items()):
            print(f"  Block {block}: {count} mismatches")
            
        print("Details of first mismatching block:")
        first_block = sorted(block_counts.keys())[0]
        br, bc = first_block
        print(f"  Block ({br}, {bc}):")
        # Print a subgrid of the differences in this block
        diff_grid = (grad_weight_cute - grad_weight_eager).abs()[br*64:br*64+64, bc*64:bc*64+64]
        print("  Max diff in this block:", diff_grid.max().item())
        print("  Indices of mismatches in this block:")
        block_mismatches = torch.nonzero(diff_grid > 1e-1)
        print("  Number of mismatches in this block:", block_mismatches.shape[0])
        print("  First 20 block-relative indices of mismatches:")
        print(block_mismatches[:20])

if __name__ == "__main__":
    diagnose()
