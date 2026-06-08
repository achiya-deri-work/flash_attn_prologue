import torch
import flash_attn_prologue.gemm_bwd
import flash_attn_prologue.fused_qkv

def test_empty_inputs():
    device = "cuda"
    dtype = torch.bfloat16
    B = 32
    N = 0  # empty sequence
    hidden = 512
    num_heads = 16
    head_dim = 64
    
    print("\n--- Testing Empty Inputs (N=0) ---")
    x = torch.zeros((B, N, hidden), device=device, dtype=dtype)
    weight = torch.zeros((3 * num_heads * head_dim, hidden), device=device, dtype=dtype)
    
    # 1. Forward pass
    print("Running forward pass...")
    q, k, v = torch.ops.flash_prologue.fused_qkv(x, weight)
    assert q.shape == (B, num_heads, N, head_dim)
    assert k.shape == (B, num_heads, N, head_dim)
    assert v.shape == (B, num_heads, N, head_dim)
    assert q.numel() == 0
    
    # 2. Backward pass custom op
    print("Running backward pass custom op...")
    grad_q = torch.zeros((B, num_heads, N, head_dim), device=device, dtype=dtype)
    grad_k = torch.zeros((B, num_heads, N, head_dim), device=device, dtype=dtype)
    grad_v = torch.zeros((B, num_heads, N, head_dim), device=device, dtype=dtype)
    
    grad_x, grad_weight = torch.ops.flash_prologue.fused_qkv_backward(grad_q, grad_k, grad_v, weight, x)
    assert grad_x.shape == (B, N, hidden)
    assert grad_weight.shape == (3 * num_heads * head_dim, hidden)
    assert grad_x.sum() == 0
    assert grad_weight.sum() == 0
    
    # 3. matmul_bwd_dx and matmul_bwd_dw custom ops
    print("Running component backward custom ops...")
    grad_x_comp = torch.ops.flash_prologue.matmul_bwd_dx_cute(grad_q, grad_k, grad_v, weight)
    assert grad_x_comp.shape == (B * N, hidden)
    assert grad_x_comp.sum() == 0
    
    grad_weight_comp = flash_attn_prologue.gemm_bwd.matmul_bwd_dw(grad_q, grad_k, grad_v, x)
    assert grad_weight_comp.shape == (3 * num_heads * head_dim, hidden)
    assert grad_weight_comp.sum() == 0
    
    # 4. Under torch.compile
    print("Running under torch.compile...")
    @torch.compile(fullgraph=True)
    def compiled_fwd_bwd(x_val, w_val):
        q_v, k_v, v_v = torch.ops.flash_prologue.fused_qkv(x_val, w_val)
        grad_q_val = torch.zeros_like(q_v)
        grad_k_val = torch.zeros_like(k_v)
        grad_v_val = torch.zeros_like(v_v)
        dx_val, dw_val = torch.ops.flash_prologue.fused_qkv_backward(grad_q_val, grad_k_val, grad_v_val, w_val, x_val)
        return q_v, dx_val, dw_val
        
    q_c, dx_c, dw_c = compiled_fwd_bwd(x, weight)
    assert q_c.numel() == 0
    assert dx_c.numel() == 0
    assert dw_c.sum() == 0
    print("Empty inputs test passed successfully!")

def test_inplace_accumulation():
    device = "cuda"
    dtype = torch.bfloat16
    B = 4
    N = 128
    hidden = 128
    num_heads = 4
    head_dim = 64
    
    print("\n--- Testing In-place Gradient Accumulation ---")
    torch.manual_seed(42)
    # Generate random inputs
    grad_q = torch.randn((B, num_heads, N, head_dim), device=device, dtype=dtype)
    grad_k = torch.randn((B, num_heads, N, head_dim), device=device, dtype=dtype)
    grad_v = torch.randn((B, num_heads, N, head_dim), device=device, dtype=dtype)
    x = torch.randn((B, N, hidden), device=device, dtype=dtype)
    
    # 1. Out-of-place reference
    print("Computing out-of-place reference...")
    ref_dw = flash_attn_prologue.gemm_bwd.matmul_bwd_dw(grad_q, grad_k, grad_v, x)
    
    # 2. In-place into zero-initialized tensor
    print("Testing in-place zero-initialized accumulation...")
    out_zero = torch.zeros_like(ref_dw)
    res_zero = flash_attn_prologue.gemm_bwd.matmul_bwd_dw(grad_q, grad_k, grad_v, x, out=out_zero)
    # Check that output matches and is in-place
    torch.testing.assert_close(res_zero, ref_dw, atol=1e-1, rtol=2e-2)
    torch.testing.assert_close(out_zero, ref_dw, atol=1e-1, rtol=2e-2)
    
    # 3. In-place into pre-initialized tensor (accumulate = True)
    print("Testing in-place with existing values...")
    out_init = torch.randn_like(ref_dw)
    out_init_clone = out_init.clone()
    res_init = flash_attn_prologue.gemm_bwd.matmul_bwd_dw(grad_q, grad_k, grad_v, x, out=out_init)
    
    # Expected is out_init_clone + ref_dw
    expected = out_init_clone + ref_dw
    torch.testing.assert_close(res_init, expected, atol=1e-1, rtol=2e-2)
    torch.testing.assert_close(out_init, expected, atol=1e-1, rtol=2e-2)
    
    # 4. Under torch.compile (with out argument)
    print("Testing in-place under torch.compile...")
    @torch.compile(fullgraph=True)
    def compiled_dw_inplace(gq, gk, gv, x_val, out_val):
        return flash_attn_prologue.gemm_bwd.matmul_bwd_dw(gq, gk, gv, x_val, out=out_val)
        
    out_compile = torch.randn_like(ref_dw)
    out_compile_clone = out_compile.clone()
    res_compile = compiled_dw_inplace(grad_q, grad_k, grad_v, x, out_compile)
    
    expected_compile = out_compile_clone + ref_dw
    torch.testing.assert_close(res_compile, expected_compile, atol=1e-1, rtol=2e-2)
    torch.testing.assert_close(out_compile, expected_compile, atol=1e-1, rtol=2e-2)
    print("In-place gradient accumulation test passed successfully!")

if __name__ == "__main__":
    test_empty_inputs()
    test_inplace_accumulation()
