import torch
import time
from flash_attn_prologue.autotune import get_autotuned_config, is_config_valid
from flash_attn_prologue.gemm_bwd import get_matmul_bwd_dx_kernel, get_matmul_bwd_dw_kernel, get_reduction_kernel, get_workspace_tensor

device = "cuda"
dtype = torch.bfloat16

B = 32
N = 512
hidden = 512
num_heads = 16
head_dim = 64

M_dim_dx = B * N
K_dim_dx = 3 * num_heads * head_dim
N_dim = hidden

# Generate test inputs
grad_q = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype) * 0.1
grad_k = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype) * 0.1
grad_v = torch.randn(B, num_heads, N, head_dim, device=device, dtype=dtype) * 0.1
weight = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=dtype) * 0.1
x = torch.randn(B * N, hidden, device=device, dtype=dtype) * 0.1

grad_output = torch.empty((B * N, 3 * num_heads * head_dim), device=device, dtype=dtype)
grad_output[:, 0 : num_heads * head_dim] = grad_q.permute(0, 2, 1, 3).reshape(B * N, num_heads * head_dim)
grad_output[:, num_heads * head_dim : 2 * num_heads * head_dim] = grad_k.permute(0, 2, 1, 3).reshape(B * N, num_heads * head_dim)
grad_output[:, 2 * num_heads * head_dim : 3 * num_heads * head_dim] = grad_v.permute(0, 2, 1, 3).reshape(B * N, num_heads * head_dim)

# Ref outputs
ref_grad_x = grad_output @ weight
ref_grad_weight = grad_output.t() @ x

static_stride_h = N * head_dim
static_stride_n = head_dim
static_stride_d = 1
static_strides = (None, static_stride_h, static_stride_n, static_stride_d)

def profile_dx(config):
    try:
        dx_kernel = get_matmul_bwd_dx_kernel(
            num_warps=config.get("num_warps", 4),
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K_STAGE=config["BLOCK_K"],
            ab_stage=config.get("ab_stage", 3),
            B=B,
            N=N,
            K_dim=K_dim_dx,
            N_dim=N_dim,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=config.get("maxrregcount", 255),
            swizzle_x=config.get("swizzle_x", 8),
            atom_layout_mnk=config.get("atom_layout_mnk", None)
        )
        grad_x_out = torch.empty((B * N, hidden), dtype=dtype, device=device)
        
        # Warmup
        dx_kernel(grad_q, grad_k, grad_v, weight, grad_x_out)
        
        diff = (grad_x_out - ref_grad_x).abs().max().item()
        if diff > 1e-1 or torch.isnan(grad_x_out).any() or torch.isinf(grad_x_out).any():
            return float("inf")
            
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            dx_kernel(grad_q, grad_k, grad_v, weight, grad_x_out)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / 50
    except Exception:
        return float("inf")

def profile_dw(config):
    try:
        split_k = config.get("SPLIT_K", 1)
        dw_kernel = get_matmul_bwd_dw_kernel(
            num_warps=config.get("num_warps", 4),
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K_STAGE=config["BLOCK_K"],
            ab_stage=config.get("ab_stage", 2),
            B=B,
            N=N,
            M_dim=3 * num_heads * head_dim,
            K_dim=B * N,
            N_dim=hidden,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=config.get("maxrregcount", 255),
            SPLIT_K=split_k,
            swizzle_x=config.get("swizzle_x", 8),
            atom_layout_mnk=config.get("atom_layout_mnk", None)
        )
        
        grad_weight = torch.zeros((split_k * 3 * num_heads * head_dim, hidden), dtype=dtype, device=device)
        dw_kernel(grad_q, grad_k, grad_v, x, grad_weight)
        
        if split_k > 1:
            grad_weight_summed = grad_weight.view(split_k, 3 * num_heads * head_dim, hidden).float().sum(dim=0).to(dtype)
        else:
            grad_weight_summed = grad_weight
            
        diff = (grad_weight_summed - ref_grad_weight).abs().max().item()
        if diff > 0.3 or torch.isnan(grad_weight_summed).any() or torch.isinf(grad_weight_summed).any():
            return float("inf")
            
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            dw_kernel(grad_q, grad_k, grad_v, x, grad_weight)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / 50
    except Exception:
        return float("inf")

# 1. Sweep dx configs
print("Sweep dx:")
dx_results = []
for bm in [64, 128]:
    for bn in [64, 128]:
        for bk in [16, 32]:
            for stages in [2, 3, 4]:
                for warps in [4, 8]:
                    for m_layout in [(1, warps, 1), (2, warps // 2, 1), (warps, 1, 1)]:
                        c = {
                            "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                            "ab_stage": stages, "num_warps": warps, "atom_layout_mnk": m_layout,
                            "maxrregcount": 255, "swizzle_x": 8
                        }
                        if is_config_valid(c, "bwd_dx"):
                            t = profile_dx(c)
                            if t < float("inf"):
                                dx_results.append((t, c))
                                print(f"  {c} -> {t:.4f} ms")

dx_results.sort(key=lambda x: x[0])
print(f"Top 3 dx configs:")
for t, c in dx_results[:3]:
    print(f"  Time: {t:.4f} ms -> {c}")

# 2. Sweep dw configs
print("\nSweep dw:")
dw_results = []
for bm in [32, 64]:
    for bn in [64]:
        for bk in [16, 32, 64]:
            for stages in [2, 3, 4]:
                for split_k in [1, 2, 4, 8, 12, 16, 32]:
                    for warps in [4, 8]:
                        for m_layout in [(1, warps, 1), (2, warps // 2, 1), (warps, 1, 1)]:
                            c = {
                                "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                "ab_stage": stages, "num_warps": warps, "atom_layout_mnk": m_layout,
                                "maxrregcount": 255, "swizzle_x": 8, "SPLIT_K": split_k
                            }
                            if is_config_valid(c, "bwd_dw"):
                                t = profile_dw(c)
                                if t < float("inf"):
                                    dw_results.append((t, c))
                                    print(f"  {c} -> {t:.4f} ms")

dw_results.sort(key=lambda x: x[0])
print(f"Top 3 dw configs:")
for t, c in dw_results[:3]:
    print(f"  Time: {t:.4f} ms -> {c}")
