import torch
import flash_attn_prologue.fused_epilogue

def reproduce():
    B = 16
    num_heads = 4
    N = 76
    head_dim = 64

    # Create storage for Q and K with the exact stride (58368, 64, 768, 1)
    # Total storage needed: 16 * 58368 = 933888
    storage_q = torch.randn(933888, dtype=torch.bfloat16, device="cuda")
    q = storage_q.as_strided((B, num_heads, N, head_dim), (58368, 64, 768, 1))

    storage_k = torch.randn(933888, dtype=torch.bfloat16, device="cuda")
    k = storage_k.as_strided((B, num_heads, N, head_dim), (58368, 64, 768, 1))

    rmsnorm_weight = torch.randn(2, 1, head_dim, dtype=torch.float32, device="cuda")
    scales = torch.randn(num_heads, 1, 1, dtype=torch.float32, device="cuda")

    # What are the strides of sin/cos? Let's check both contiguous and strided cases.
    sin = torch.randn(B, 1, N, head_dim // 2, dtype=torch.float32, device="cuda")
    cos = torch.randn(B, 1, N, head_dim // 2, dtype=torch.float32, device="cuda")

    print("q shape:", q.shape, "stride:", q.stride())
    print("rmsnorm_weight shape:", rmsnorm_weight.shape)
    print("scales shape:", scales.shape)
    print("sin shape:", sin.shape, "stride:", sin.stride())

    print("\nCalling qknorm_rope_scale_cute...")
    q_out, k_out = torch.ops.flash_prologue.qknorm_rope_scale_cute(
        q, k, rmsnorm_weight, scales, sin, cos
    )
    print("Success!")

if __name__ == "__main__":
    reproduce()
