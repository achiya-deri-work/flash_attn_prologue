import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import time
from flash_attn_prologue.epilogue_kernel import qknorm_rope_scale_eager
import flash_attn_prologue.epilogue_kernel # force registration of custom op

class EagerEpilogueModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = nn.Parameter(torch.randn(num_heads, 1, 1))

    def forward(self, x, sin, cos):
        qkv = self.qkv_proj(x)
        # Call the custom op for epilogue
        q, k, v = torch.ops.flash_prologue.qknorm_rope_scale(
            qkv, self.qk_norm_weight, self.scales, sin, cos
        )
        return q, k, v

class EagerPureModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.qk_norm_weight = nn.Parameter(torch.ones(2, 1, head_dim))
        self.scales = nn.Parameter(torch.randn(num_heads, 1, 1))

    def forward(self, x, sin, cos):
        qkv = self.qkv_proj(x)
        q_proj, k_proj, v_proj = qknorm_rope_scale_eager(
            qkv, self.qk_norm_weight, self.scales, sin, cos
        )
        return q_proj, k_proj, v_proj

def test_epilogue_forward_and_backward():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn(B, N, hidden, device=device, dtype=dtype, requires_grad=True)
    rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=dtype, requires_grad=True)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=dtype, requires_grad=True)
    
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)
    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(-1).expand(*sin_val.shape, 2).reshape(B, 1, N, head_dim).to(device=device, dtype=dtype)
    cos = cos_val.unsqueeze(-1).expand(*cos_val.shape, 2).reshape(B, 1, N, head_dim).to(device=device, dtype=dtype)

    # 1. Forward check
    qkv = torch.randn(B, N, 3 * num_heads * head_dim, device=device, dtype=dtype, requires_grad=True)
    qkv_custom = qkv.detach().clone().requires_grad_(True)
    
    q_e, k_e, v_e = qknorm_rope_scale_eager(qkv, rmsnorm_weight, scales, sin, cos, eps)
    q_t, k_t, v_t = torch.ops.flash_prologue.qknorm_rope_scale(qkv_custom, rmsnorm_weight, scales, sin, cos, eps)

    # Validate output parity
    torch.testing.assert_close(q_t, q_e, atol=1.0, rtol=5e-2)
    torch.testing.assert_close(k_t, k_e, atol=1.0, rtol=5e-2)
    torch.testing.assert_close(v_t, v_e, atol=1e-2, rtol=2e-2)
    print("Forward parity passed successfully!")

    # 2. Backward check
    loss_e = q_e.sum() + k_e.sum() + v_e.sum()
    loss_e.backward()
    
    loss_t = q_t.sum() + k_t.sum() + v_t.sum()
    loss_t.backward()

    # Validate gradient parity
    torch.testing.assert_close(qkv_custom.grad, qkv.grad, atol=5e-2, rtol=5e-2)
    print("Backward gradient check passed successfully!")

def test_epilogue_compilation():
    B = 4
    N = 64
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    sin = torch.randn(B, 1, N, head_dim, device=device, dtype=dtype)
    cos = torch.randn(B, 1, N, head_dim, device=device, dtype=dtype)

    eager_model = EagerEpilogueModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    compiled_model = torch.compile(eager_model, backend="inductor")

    # Warmup / compilation
    compiled_model(x, sin, cos)
    
    # Run and check parity
    q_ref, k_ref, v_ref = eager_model(x, sin, cos)
    q_comp, k_comp, v_comp = compiled_model(x, sin, cos)

    torch.testing.assert_close(q_comp, q_ref, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(k_comp, k_ref, atol=2e-3, rtol=1e-2)
    print("Triton epilogue compiled correctly with torch.compile!")

def benchmark_linear_plus_triton():
    B = 32
    N = 512
    hidden = 512
    num_heads = 16
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(B, N, hidden, device=device, dtype=dtype)
    sin = torch.randn(B, 1, N, head_dim, device=device, dtype=dtype)
    cos = torch.randn(B, 1, N, head_dim, device=device, dtype=dtype)

    pure_eager = EagerPureModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    pure_compiled = torch.compile(pure_eager, backend="inductor")
    
    triton_model = EagerEpilogueModel(hidden, num_heads, head_dim).to(device=device, dtype=dtype)
    triton_model.load_state_dict(pure_eager.state_dict())
    triton_compiled = torch.compile(triton_model, backend="inductor")

    # Warmups
    for _ in range(20):
        _ = pure_compiled(x, sin, cos)
        _ = triton_compiled(x, sin, cos)
    torch.cuda.synchronize()

    # 1. Pure compiled benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _ = pure_compiled(x, sin, cos)
    end_event.record()
    torch.cuda.synchronize()
    pure_time = start_event.elapsed_time(end_event) / 100.0

    # 2. Triton compiled benchmark
    start_event.record()
    for _ in range(100):
        _ = triton_compiled(x, sin, cos)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / 100.0

    print(f"\n[Epilogue Benchmarks]")
    print(f"Inductor Pure Compiled Baseline: {pure_time:.4f} ms")
    print(f"F.linear + Triton Custom Epilogue: {triton_time:.4f} ms")
    print(f"Ratio: {triton_time / pure_time:.2f}x")

if __name__ == "__main__":
    test_epilogue_forward_and_backward()
    test_epilogue_compilation()
    benchmark_linear_plus_triton()
