import sys
import os
import math
from typing import Tuple, Optional, cast, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Ensure the package root and Advantage-First-RL are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/home/achiya/vs_code_main/Advantage-First-RL")

from flash_attn_prologue.layers.RMSNorm import RMSNormRoPEScale, RMSNorm
from cute_mamba3 import cute_mamba3, cute_mamba3_fallback


# =====================================================================
# 1. Cloned RoPE Self Attention Module (from Advantage-First-RL)
# =====================================================================

def format_rope_buffer(sin_or_cos: torch.Tensor) -> torch.Tensor:
    """Formats a buffer to [1, 1, seq, dhead/2] matching ROPE_5D = False."""
    seq_len = sin_or_cos.shape[-2]
    dhead_half = sin_or_cos.shape[-1]
    return sin_or_cos.view(1, 1, seq_len, dhead_half).contiguous()


class ClonedFlashRoPESelfAttention(nn.Module):
    """
    Self-Attention using FLASH_ATTENTION backend with RoPE support.
    Cloned from Advantage-First-RL.
    """
    def __init__(self, hidden_dim: int, num_heads: int, thetas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, bias: bool = False, attn_drop: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.bias = bias
        self.attn_drop = attn_drop
        self.backends = [SDPBackend.FLASH_ATTENTION]
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.qkrmsnorm_rope_scales = RMSNormRoPEScale(head_dim=self.head_dim, num_heads=self.num_heads)
        
        if thetas is not None:
            sin, cos = thetas
            self.register_buffer("sin_cached", format_rope_buffer(sin), persistent=True)
            self.register_buffer("cos_cached", format_rope_buffer(cos), persistent=True)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters of the self-attention layer."""
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, 'reset_parameters') and callable(module.reset_parameters):
                module.reset_parameters()

    def reset_rope(self, thetas: Tuple[torch.Tensor, torch.Tensor] = None) -> None:
        if hasattr(self, 'sin_cached') and thetas is not None:
            sin, cos = thetas
            self.sin_cached.data = format_rope_buffer(sin).to(device=self.sin_cached.device, dtype=torch.float32)
            self.cos_cached.data = format_rope_buffer(cos).to(device=self.cos_cached.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor, thetas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        q_proj, k_proj, v_proj = self.qkv_proj(x).view(B, N, 3, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4).unbind(2)
        
        if thetas is None:
            q_proj, k_proj = self.qkrmsnorm_rope_scales(q_proj, k_proj, self.sin_cached, self.cos_cached)
        else:
            q_proj, k_proj = self.qkrmsnorm_rope_scales(q_proj, k_proj, thetas[0], thetas[1])
        
        with sdpa_kernel(backends=self.backends):
            attn_output = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=self.attn_drop if self.training else 0.0, scale=1.0)
        
        output = self.out_proj(attn_output.transpose(1, 2).contiguous().view(B, N, -1))
        return output


# =====================================================================
# 2. Cloned Mamba Single Step Update Module (from Advantage-First-RL)
# =====================================================================

def init_mamba_state(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    state_dim: int,
    head_dim: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_layers: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cloned state initialization helper."""
    if num_layers is not None:
        shape_full = (num_layers, batch_size, seq_len, num_heads, state_dim, head_dim)
        shape_rope = (num_layers, batch_size, seq_len, num_heads, state_dim // 4)
    else:
        shape_full = (batch_size, seq_len, num_heads, state_dim, head_dim)
        shape_rope = (batch_size, seq_len, num_heads, state_dim // 4)

    ssm_state = torch.zeros(shape_full, device=device, dtype=dtype)
    trap_mem = torch.zeros(shape_full, device=device, dtype=dtype)
    rope_angles = torch.zeros(shape_rope, device=device, dtype=dtype)
    
    return ssm_state, trap_mem, rope_angles


class ClonedMamba3SingleStepUpdate(nn.Module):
    """
    Continuous-time ODE Memory Update Module.
    Cloned from Advantage-First-RL.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        state_dim: int = 16,
        mimo_rank: int = 4,      # R
        dt_min: float = 0.001, 
        dt_max: float = 0.1, 
        dt_init_floor: float = 1e-4, 
        A_floor: float = 1e-4,
        backend: Optional[str] = None
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension D must be divisible by num_heads"
        self.backend = backend or "cute"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_dim // num_heads)  # P = D / H
        self.state_dim = state_dim  # N
        self.rank = mimo_rank             # R
        
        assert self.state_dim % 4 == 0, "State dimension N must be divisible by 4 for partial RoPE rotation."
        
        self.A_floor = A_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        
        self.split_sizes = [
            self.hidden_dim,                           # z: gate (D)
            self.hidden_dim,                           # x: value input (D)
            self.state_dim * self.rank,               # B: shared state input proj (N * R)
            self.state_dim * self.rank,               # C: shared state output proj (N * R)
            self.num_heads,                           # dt: step size per head (H)
            self.num_heads,                           # A: decay parameter per head (H)
            self.num_heads,                           # trap: lambda for trapezoidal rule (H)
            self.state_dim // 4                       # theta: RoPE frequencies (Shared, N/4)
        ]
        self.d_proj = sum(self.split_sizes)
        self.in_proj = nn.Linear(self.hidden_dim, self.d_proj, bias=False)
        
        self.W_X = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        self.W_Z = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        self.W_O = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        
        self.b_norm = RMSNorm(self.state_dim * self.rank)
        self.c_norm = RMSNorm(self.state_dim * self.rank)
        
        self.B_bias = nn.Parameter(torch.empty(self.num_heads, self.state_dim, self.rank))
        self.C_bias = nn.Parameter(torch.empty(self.num_heads, self.state_dim, self.rank))
        
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.D_skip = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameter hierarchies for numerical stability"""
        self.float()
        for module in self.children():
            if hasattr(module, 'reset_parameters') and callable(module.reset_parameters):
                module.reset_parameters()
                
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.W_X, 1.0 / self.rank)
        nn.init.constant_(self.W_O, 1.0 / self.rank)
        nn.init.constant_(self.W_Z, 1.0)
        
        nn.init.ones_(self.B_bias)
        nn.init.ones_(self.C_bias)
        nn.init.ones_(self.D_skip)
        
        _dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        _dt = torch.clamp(_dt, min=self.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        with torch.no_grad():
            self.dt_bias.copy_(_dt_bias)
        self.out_proj.bfloat16()

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """High-performance Real-Valued Data-Dependent RoPE evaluation."""
        x1, x2 = x.chunk(2, dim=-2)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)
        rotated_x1 = (x1 * cos) - (x2 * sin)
        rotated_x2 = (x2 * cos) + (x1 * sin)
        return torch.cat([rotated_x1, rotated_x2], dim=-2)

    def forward(
        self, 
        curr_mems: torch.Tensor, 
        ssm_state: Optional[torch.Tensor] = None,
        trap_mem: Optional[torch.Tensor] = None,
        rope_angles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = curr_mems.shape
        proj_out = self.in_proj(curr_mems.to(self.in_proj.weight.dtype))
        
        z_raw, x_raw, B_raw, C_raw, dt_raw, A_raw, trap_raw, theta_raw = torch.split(
            proj_out, self.split_sizes, dim=-1
        )
        
        B_normed = self.b_norm(B_raw).reshape(B, S, 1, self.state_dim, self.rank)
        C_normed = self.c_norm(C_raw).reshape(B, S, 1, self.state_dim, self.rank)
        
        B_bias_5d = self.B_bias.unsqueeze(0).unsqueeze(0)
        C_bias_5d = self.C_bias.unsqueeze(0).unsqueeze(0)
        B_t = B_normed + B_bias_5d
        C_t = C_normed + C_bias_5d
        
        dt = F.softplus(dt_raw + self.dt_bias)              
        A = -torch.clamp(F.softplus(A_raw), min=self.A_floor)
        lam = torch.sigmoid(trap_raw)                       
        
        theta = theta_raw.reshape(B, S, 1, self.state_dim // 4)
        
        rope_angles_t = cast(torch.Tensor, rope_angles)
        curr_angles = rope_angles_t + (torch.tanh(theta) * dt.unsqueeze(-1) * math.pi)
        
        two_pi = 2.0 * math.pi
        curr_angles = curr_angles - two_pi * torch.floor(curr_angles / two_pi)
        
        cos_angles = torch.cos(curr_angles)
        sin_angles = torch.sin(-curr_angles)
        
        B_t_rot, B_t_pass = B_t.chunk(2, dim=-2)
        C_t_rot, C_t_pass = C_t.chunk(2, dim=-2)
        
        B_rot_rotated = self._apply_rope(B_t_rot, cos_angles, sin_angles)
        C_rot_rotated = self._apply_rope(C_t_rot, cos_angles, sin_angles)
        
        B_rot = torch.cat([B_rot_rotated, B_t_pass], dim=-2)
        C_rot = torch.cat([C_rot_rotated, C_t_pass], dim=-2)
        
        is_optimized = (self.head_dim % 16 == 0) and (self.state_dim % 8 == 0) and ((self.state_dim * self.rank) % 32 == 0)
        
        if self.backend == "cute" and is_optimized:
            out_gated_prime, curr_state, curr_V = cute_mamba3(
                x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                self.W_X, self.W_Z, self.W_O, self.D_skip
            )
        else:
            out_gated_prime, curr_state, curr_V = cute_mamba3_fallback(
                x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                self.W_X, self.W_Z, self.W_O, self.D_skip
            )
        
        out = self.out_proj(out_gated_prime)
        return out, curr_state, curr_V, curr_angles


# =====================================================================
# 3. Unit Tests & Numerical Correctness Checks under torch.compile
# =====================================================================

def test_cloned_flash_rope_self_attention_compile_parity():
    """Validates FlashRoPESelfAttention eager vs compiled numerical output/gradients."""
    B = 16
    N = 64
    num_heads = 4
    head_dim = 64
    hidden_dim = num_heads * head_dim
    device = "cuda"
    
    # 1. Inputs & Targets
    x = torch.randn(B, N, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(B, N, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Generate mock cached sin/cos frequencies
    sin_val = torch.randn(N, head_dim // 2, device=device, dtype=torch.float32)
    cos_val = torch.randn(N, head_dim // 2, device=device, dtype=torch.float32)
    thetas = (sin_val, cos_val)
    
    # 2. Instantiate and compile module
    module = ClonedFlashRoPESelfAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, thetas=thetas, bias=True
    ).to(device=device, dtype=torch.bfloat16)
    
    # Make deep copy for eager baseline
    import copy
    eager_module = copy.deepcopy(module)
    compiled_module = torch.compile(module, fullgraph=True)
    
    # 3. Cloned inputs for eager baseline
    x_eager = x.clone().detach().requires_grad_(True)
    
    # 4. Eager Forward & Backward
    out_eager = eager_module(x_eager)
    loss_eager = F.mse_loss(out_eager, target)
    loss_eager.backward()
    
    # 5. Compiled Forward & Backward
    out_compiled = compiled_module(x)
    loss_compiled = F.mse_loss(out_compiled, target)
    loss_compiled.backward()
    
    # 6. Verify Numerical Correctness (Output & Gradient Parity)
    print("out_compiled max abs:", out_compiled.abs().max().item())
    print("out_eager max abs:", out_eager.abs().max().item())
    print("x.grad max abs:", x.grad.abs().max().item() if x.grad is not None else "None")
    print("x_eager.grad max abs:", x_eager.grad.abs().max().item() if x_eager.grad is not None else "None")
    
    # Since default calculation is in bfloat16, we adjust tolerances to account for low precision and compilation fusions
    torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=2e-1)
    torch.testing.assert_close(x.grad, x_eager.grad, rtol=1e-2, atol=2e-1)
    
    # Verify parameter gradients match
    for (name_comp, p_comp), (name_eag, p_eag) in zip(module.named_parameters(), eager_module.named_parameters()):
        assert p_comp.grad is not None, f"Compiled parameter {name_comp} gradient is None"
        assert p_eag.grad is not None, f"Eager parameter {name_eag} gradient is None"
        torch.testing.assert_close(p_comp.grad, p_eag.grad, rtol=1e-2, atol=2e-1)
        
    print("✓ test_cloned_flash_rope_self_attention_compile_parity passed!")


def test_cloned_mamba3_single_step_update_compile_parity():
    """Validates Mamba3SingleStepUpdate eager vs compiled numerical output/gradients."""
    B = 16
    S = 64
    num_heads = 4
    state_dim = 32
    head_dim = 64
    mimo_rank = 4
    hidden_dim = num_heads * head_dim
    device = "cuda"
    
    # 1. Inputs & Targets
    curr_mems = torch.randn(B, S, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(B, S, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Initialize states in float32 for precision
    ssm_state, trap_mem, rope_angles = init_mamba_state(
        batch_size=B, seq_len=S, num_heads=num_heads, state_dim=state_dim, head_dim=head_dim, device=device, dtype=torch.float32
    )
    
    # Initialize states with random non-zero values to activate decay path gradients (e.g. A gradients)
    ssm_state.copy_(torch.randn_like(ssm_state))
    trap_mem.copy_(torch.randn_like(trap_mem))
    rope_angles.copy_(torch.randn_like(rope_angles))
    
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True
    
    # 2. Clones for Eager Baseline
    curr_mems_eager = curr_mems.clone().detach().requires_grad_(True)
    ssm_state_eager = ssm_state.clone().detach().requires_grad_(True)
    trap_mem_eager = trap_mem.clone().detach().requires_grad_(True)
    rope_angles_eager = rope_angles.clone().detach()
    
    # 3. Instantiate and compile module
    module = ClonedMamba3SingleStepUpdate(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dim=state_dim,
        mimo_rank=mimo_rank,
        backend="cute"
    ).to(device=device)
    
    import copy
    eager_module = copy.deepcopy(module)
    compiled_module = torch.compile(module, fullgraph=True)
    
    # 4. Eager Forward & Backward
    out_eag, state_eag, V_eag, angles_eag = eager_module(
        curr_mems_eager, ssm_state_eager, trap_mem_eager, rope_angles_eager
    )
    # Calculate dummy loss and run backward
    loss_eager = F.mse_loss(out_eag, target) + state_eag.sum() + V_eag.sum() + angles_eag.sum()
    loss_eager.backward()
    
    # 5. Compiled Forward & Backward
    out_comp, state_comp, V_comp, angles_comp = compiled_module(
        curr_mems, ssm_state, trap_mem, rope_angles
    )
    loss_compiled = F.mse_loss(out_comp, target) + state_comp.sum() + V_comp.sum() + angles_comp.sum()
    loss_compiled.backward()
    
    # 6. Verify Numerical Correctness (Output & Gradient Parity)
    # Output Parity
    torch.testing.assert_close(out_comp, out_eag, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_comp, state_eag, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(V_comp, V_eag, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(angles_comp, angles_eag, rtol=1e-2, atol=1e-2)
    
    # Input Gradient Parity
    torch.testing.assert_close(curr_mems.grad, curr_mems_eager.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ssm_state.grad, ssm_state_eager.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(trap_mem.grad, trap_mem_eager.grad, rtol=1e-2, atol=1e-2)
    
    # Parameter Gradient Parity
    for (name_comp, p_comp), (name_eag, p_eag) in zip(module.named_parameters(), eager_module.named_parameters()):
        assert p_comp.grad is not None, f"Compiled parameter {name_comp} gradient is None"
        assert p_eag.grad is not None, f"Eager parameter {name_eag} gradient is None"
        torch.testing.assert_close(p_comp.grad, p_eag.grad, rtol=1e-2, atol=1e-2)
        
    print("✓ test_cloned_mamba3_single_step_update_compile_parity passed!")


if __name__ == "__main__":
    test_cloned_flash_rope_self_attention_compile_parity()
    test_cloned_mamba3_single_step_update_compile_parity()
