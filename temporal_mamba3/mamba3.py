import math
from typing import Tuple, NamedTuple, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import os
from advantage_first.layers.cute.cute_mamba3 import cute_mamba3, cute_mamba3_fallback

from advantage_first.layers.common import LayerScale, DropPath

# Global default backend toggle ("cute" or "native")
DEFAULT_BACKEND = os.environ.get("MAMBA3_BACKEND", "cute")
from advantage_first.layers.norms import RMSNorm

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
    """
    Initializes the continuous-time ODE memory states for the MIMO Mamba-3 vault.
    
    Natively supports [Batch, Seq, ...] dimensions without requiring manual flattening,
    treating the 'Seq' dimension as independent parallel spatial/latent tokens.
    
    Precision Safety:
    `dtype` should strictly remain FP32 to prevent the swamping problem 
    (where microscopic ODE delta updates are lost to low-precision rounding).
    
    Args:
        batch_size: Number of independent environments in the batch (B)
        seq_len: Number of independent memory tokens (M)
        num_heads: Number of attention heads (H)
        state_dim: State dimension (N)
        head_dim: Head dimension (P)
        device: Target compute device
        dtype: Target precision for the ODE vault (Default: float32)
        num_layers: Optional depth dimension to stack states for multi-layer models
        
    Returns:
        ssm_state: [num_layers, B, S, H, N, P] if num_layers is provided, else [B, S, H, N, P]
        trap_mem: [num_layers, B, S, H, N, P] if num_layers is provided, else [B, S, H, N, P]
        rope_angles: [num_layers, B, S, H, N/2] if num_layers is provided, else [B, S, H, N/2]
    """
    if num_layers is not None:
        shape_full = (num_layers, batch_size, seq_len, num_heads, state_dim, head_dim)
        shape_rope = (num_layers, batch_size, seq_len, num_heads, state_dim // 4)
    else:
        shape_full = (batch_size, seq_len, num_heads, state_dim, head_dim)
        shape_rope = (batch_size, seq_len, num_heads, state_dim // 4)

    # 1. Full-width MIMO states
    ssm_state = torch.zeros(shape_full, device=device, dtype=dtype)
    
    # 2. Previous State Input (Trapezoidal Memory) V_{t-1}
    trap_mem = torch.zeros(shape_full, device=device, dtype=dtype)
    
    # 3. RoPE accumulation angles (\Theta_t)
    rope_angles = torch.zeros(shape_rope, device=device, dtype=dtype)
    
    return ssm_state, trap_mem, rope_angles


class Mamba3SingleStepUpdate(nn.Module):
    """
    Pure PyTorch implementation of a Single-Step Autoregressive Working Memory Update.
    Natively supports [Batch, Seq, ...] dimensions without requiring manual flattening,
    treating the 'Seq' dimension as independent parallel spatial/latent tokens.
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
        self.backend = backend or DEFAULT_BACKEND
        
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
        
        # ----------------------------------------------------------------------
        # 1. Hardware-Efficient Packed Projection
        # ----------------------------------------------------------------------
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
        
        # ----------------------------------------------------------------------
        # 2. MIMO Scale Parameters (Data-Independent Rank Projections)
        # ----------------------------------------------------------------------
        self.W_X = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        self.W_Z = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        self.W_O = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.rank))
        
        # ----------------------------------------------------------------------
        # 3. Value Normalization & Head-Specific Biases
        # ----------------------------------------------------------------------
        # Replace with your repo's RMSNorm
        self.b_norm = RMSNorm(self.state_dim * self.rank)
        self.c_norm = RMSNorm(self.state_dim * self.rank)
        
        self.B_bias = nn.Parameter(torch.empty(self.num_heads, self.state_dim, self.rank))
        self.C_bias = nn.Parameter(torch.empty(self.num_heads, self.state_dim, self.rank))
        
        # ----------------------------------------------------------------------
        # 4. Temporal Step (dt) and Final D Skip Connection
        # ----------------------------------------------------------------------
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.D_skip = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        
        # Final Output Demotion Projection
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
        
        # Mamba-3 initialized constant-bias rules
        nn.init.ones_(self.B_bias)
        nn.init.ones_(self.C_bias)
        nn.init.ones_(self.D_skip)
        
        # dt bias Softplus-inverse initialization per head
        _dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        _dt = torch.clamp(_dt, min=self.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        with torch.no_grad():
            self.dt_bias.copy_(_dt_bias)
        self.out_proj.bfloat16()


    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        High-performance Real-Valued Data-Dependent RoPE evaluation.
        Native support for arbitrary leading dimensions.
        """
        # x shape: [B, S, H, N, R] 
        # Llama-style contiguous RoPE blocks split identically across state dimension (N)
        x1, x2 = x.chunk(2, dim=-2)
        
        # Broadcast trigonometric arrays over the MIMO Rank (R) dimension natively
        # cos/sin shape: [B, S, H, N/2] -> [B, S, H, N/2, 1]
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
        """
        Assumes curr_mems is bfp16 and ssm_state, trap_mem, rope_angles are fp32.
        """
        orig_dtype = curr_mems.dtype
        B, S, _ = curr_mems.shape
            
        # =====================================================================
        # 1. BFloat16 -> FP32 Precision Safety Cutoff
        # =====================================================================
        proj_out = self.in_proj(curr_mems.to(self.in_proj.weight.dtype))
        
        z_raw, x_raw, B_raw, C_raw, dt_raw, A_raw, trap_raw, theta_raw = torch.split(
            proj_out, self.split_sizes, dim=-1
        )
        
        # =====================================================================
        # 2. Value Normalization & MIMO Shaping
        # =====================================================================
        # B and C are initially shared across heads: [B, S, N, R]
        B_normed = self.b_norm(B_raw).reshape(B, S, 1, self.state_dim, self.rank)
        C_normed = self.c_norm(C_raw).reshape(B, S, 1, self.state_dim, self.rank)
        
        # Broadcast to H heads, THEN apply head-specific independent biases -> [B, S, H, N, R]
        B_bias_5d = self.B_bias.unsqueeze(0).unsqueeze(0)
        C_bias_5d = self.C_bias.unsqueeze(0).unsqueeze(0)
        B_t = B_normed + B_bias_5d
        C_t = C_normed + C_bias_5d
        
        # =====================================================================
        # 3. Parameterize Selective Decays Per-Head (H)
        # =====================================================================
        dt = F.softplus(dt_raw + self.dt_bias)              
        A = -torch.clamp(F.softplus(A_raw), min=self.A_floor)
        lam = torch.sigmoid(trap_raw)                       
        
        # =====================================================================
        # 4. Complex-Valued State Tracking (Data-Dependent RoPE)
        # =====================================================================
        # Project state_dim // 4 frequencies directly
        theta = theta_raw.reshape(B, S, 1, self.state_dim // 4)
        
        # Cumulative Angle tracking with tanh bounds and pi scaling: \Theta_t = \Theta_{t-1} + \tanh(\theta_t) * dt_t * \pi
        rope_angles_t = cast(torch.Tensor, rope_angles)
        curr_angles = rope_angles_t + (torch.tanh(theta) * dt.unsqueeze(-1) * math.pi)
        
        # Modulo 2*pi normalization to prevent numerical swamping
        two_pi = 2.0 * math.pi
        curr_angles = curr_angles - two_pi * torch.floor(curr_angles / two_pi)
        
        # Mamba-3 RoPE Trick: Rotation by negative inverse-angles 
        cos_angles = torch.cos(curr_angles)
        sin_angles = torch.sin(-curr_angles)
        
        # Split B_t and C_t along the state dimension (N) into rotated and non-rotated halves
        B_t_rot, B_t_pass = B_t.chunk(2, dim=-2)
        C_t_rot, C_t_pass = C_t.chunk(2, dim=-2)
        
        B_rot_rotated = self._apply_rope(B_t_rot, cos_angles, sin_angles)
        C_rot_rotated = self._apply_rope(C_t_rot, cos_angles, sin_angles)
        
        B_rot = torch.cat([B_rot_rotated, B_t_pass], dim=-2)
        C_rot = torch.cat([C_rot_rotated, C_t_pass], dim=-2)
        
        # Check if the shape is optimized for the custom JIT CuTe kernel (P % 16 == 0, N % 8 == 0, (N * R) % 32 == 0)
        is_optimized = (self.head_dim % 16 == 0) and (self.state_dim % 8 == 0) and ((self.state_dim * self.rank) % 32 == 0)
        
        # Call custom operator or fallback depending on self.backend and shape optimization
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
        
        out = self.out_proj(out_gated_prime.to(orig_dtype))

        
        return out, curr_state, curr_V, curr_angles






class MambaTokenODEUpdater(nn.Module):
    """
    Continuous-time ODE Integrator for Recurrent Memory Tokens.
    Replaces standard residual connections with Mamba's exponential forget 
    gating and data-dependent plasticity.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int = 16, 
        dt_min: float = 0.001, 
        dt_max: float = 0.1, 
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        ls_init_value: float = 1e-5,
        state_dtype: torch.dtype = torch.float32,
        working_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by num_heads."
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.A_floor = A_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.ls_init_value = ls_init_value
        self.state_dtype = state_dtype
        self.working_dtype = working_dtype
        
        # ----------------------------------------------------------------------
        # 1. Parameter Projections
        # We project exactly 2 parameters per head: dt (plasticity) and A (decay)
        # ----------------------------------------------------------------------
        self.param_proj = nn.Linear(hidden_dim, num_heads * 2, bias=False, dtype=self.state_dtype)
        
        # ----------------------------------------------------------------------
        # 2. Step Size (dt) Bias Initialization
        # Initialized identically to Mamba to ensure softplus(dt) stays within bounds
        # ----------------------------------------------------------------------
        self.dt_bias = nn.Parameter(torch.empty(num_heads, dtype=self.state_dtype))
        
        # ----------------------------------------------------------------------
        # 3. LayerScale for the Update Vector
        # Replaces LayerNorm to ensure the update magnitude starts near zero
        # ----------------------------------------------------------------------
        self.gamma = nn.Parameter(torch.empty(1, 1, self.num_heads, self.head_dim, dtype=self.state_dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights for numerical stability."""
        self.to(dtype=self.state_dtype)
        nn.init.trunc_normal_(self.param_proj.weight, std=0.02)
        nn.init.constant_(self.gamma, self.ls_init_value)
        
        # dt bias Softplus-inverse initialization
        _dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        _dt = torch.clamp(_dt, min=self.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        with torch.no_grad():
            self.dt_bias.copy_(_dt_bias)
        

    def forward(self, mem: torch.Tensor, candidate_update: torch.Tensor) -> torch.Tensor:
        """
        Executes the continuous ODE integration step.
        
        Args:
            mem: Current memory tokens [Batch, Seq, Hidden] torch.float32
            candidate_update: The output from the Transformer model [Batch, Seq, Hidden] torch.bfloat16
            
        Returns:
            next_mem: The ODE-integrated memory for t+1 [Batch, Seq, Hidden] torch.float32
        """
        orig_dtype = mem.dtype
        B, S, D = mem.shape
        
        # =====================================================================
        # 1. BFloat16 -> FP32 Precision Safety Cutoff
        # ODE integrations will explode if executed in BFloat16 over long horizons
        # =====================================================================
        assert mem.dtype == self.state_dtype, f"Memory must be {self.state_dtype}, got {mem.dtype}"
        assert candidate_update.dtype in (self.working_dtype, self.state_dtype), f"Candidate update must be {self.working_dtype} or {self.state_dtype}, got {candidate_update.dtype}"

        update_f32 = candidate_update.to(self.state_dtype)
        
        # =====================================================================
        # 2. Project ODE Parameters (Data-Dependent)
        # We project the parameters from the candidate update so the model 
        # dynamically chooses how to integrate the new information.
        # =====================================================================
        # Optimized version using torch.chunk() to split the last dimension
        dt_raw, A_raw = self.param_proj(update_f32).chunk(2, dim=-1)  # Both [B, S, H]

        # =====================================================================
        # 3. Constrain Parameters
        # dt > 0 (Time flows forward)
        # A < 0  (Memory decays, preventing explosion)
        # =====================================================================
        dt = F.softplus(dt_raw + self.dt_bias)
        A = -torch.clamp(F.softplus(A_raw), min=self.A_floor)
        
        # =====================================================================
        # 4. Multi-Head Geometry Formatting
        # Expand [B, S, H] -> [B, S, H, 1] for broadcasting across the head dimension
        # =====================================================================
        # dt_expanded = dt.unsqueeze(-1)
        # A_expanded = A.unsqueeze(-1)
        
        # Reshape physical memory and updates to [B, S, H, P]
        mem_heads = mem.view(B, S, self.num_heads, self.head_dim)
        update_heads = update_f32.view(B, S, self.num_heads, self.head_dim)
        
        # Apply LayerScale to the candidate update
        scaled_update = update_heads * self.gamma
        
        # =====================================================================
        # 5. The Euler ODE Integration (Mamba's Decay Mechanism)
        # mem_{t+1} = exp(dt * A) * mem_t + dt * update_t
        # =====================================================================
        # alpha_t is strictly between 0 and 1.
        alpha_t = torch.exp(dt * A).unsqueeze(-1) # expand just in time letting torch compile kernel fuse it
        
        next_mem_heads = (alpha_t * mem_heads) + (dt.unsqueeze(-1) * scaled_update)
        
        # =====================================================================
        # 6. Repack Geometry and Demote Precision
        # =====================================================================
        next_mem = next_mem_heads.view(B, S, D)
        
        return next_mem



class Mamba3Block(nn.Module):
    """
    A full Mamba-3 Block consisting of a Pre-Norm, Mamba3SingleStepUpdate, LayerScale, 
    DropPath, and a Residual Connection.
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
        ls_init_value: float = 1e-5,
        drop_path: float = 0.0,
        backend: Optional[str] = None
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.mamba = Mamba3SingleStepUpdate(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dim=state_dim,
            mimo_rank=mimo_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            A_floor=A_floor,
            backend=backend
        )
        self.ls = LayerScale(hidden_dim, init_values=ls_init_value)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters') and callable(module.reset_parameters):
                module.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        past: Optional[torch.Tensor] = None,
        prev_V: Optional[torch.Tensor] = None,
        prev_angles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mamba Core Update
        out, n_past, n_prev_V, n_prev_angles = self.mamba(self.norm(x), past, prev_V, prev_angles)
        
        # LayerScale -> DropPath -> Residual Connection
        x = x + self.drop_path(self.ls(out))
        
        return x, n_past, n_prev_V, n_prev_angles


class Mamba3Encoder(nn.Module):
    """
    A stack of Mamba-3 Blocks. Iterates through the depth dimension maintaining state.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        state_dim: int = 16,
        mimo_rank: int = 4,      # R
        dt_min: float = 0.001, 
        dt_max: float = 0.1, 
        dt_init_floor: float = 1e-4, 
        A_floor: float = 1e-4,
        ls_init_value: float = 1e-5,
        dpr_start: float = 0.0,
        dpr_end: float = 0.0,
        backend: Optional[str] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_dim // num_heads)
        self.state_dim = state_dim
        self.mimo_rank = mimo_rank

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(dpr_start, dpr_end, num_layers)]
        
        self.layers = nn.ModuleList([
            Mamba3Block(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                state_dim=state_dim,
                mimo_rank=mimo_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                A_floor=A_floor,
                ls_init_value=ls_init_value,
                drop_path=dpr[i],
                backend=backend
            ) for i in range(num_layers)
        ])
        
    def reset_parameters(self):
        for layer in self.layers:
            cast(Mamba3Block, layer).reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        pasts: Optional[torch.Tensor] = None,
        prev_Vs: Optional[torch.Tensor] = None,
        prev_angles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_pasts, next_prev_Vs, next_prev_angles = [], [], []
        
        pasts_t = cast(torch.Tensor, pasts)
        prev_Vs_t = cast(torch.Tensor, prev_Vs)
        prev_angles_t = cast(torch.Tensor, prev_angles)
        
        for i, layer in enumerate(self.layers):
            x, n_past, n_prev_V, n_prev_angle = layer(x, pasts_t[i], prev_Vs_t[i], prev_angles_t[i])
            
            next_pasts.append(n_past)
            next_prev_Vs.append(n_prev_V)
            next_prev_angles.append(n_prev_angle)
            
        # Stack the states back into [num_layers, batch, seq, dim] format
        next_pasts = torch.stack(next_pasts, dim=0)
        next_prev_Vs = torch.stack(next_prev_Vs, dim=0)
        next_prev_angles = torch.stack(next_prev_angles, dim=0)
        
        return x, next_pasts, next_prev_Vs, next_prev_angles