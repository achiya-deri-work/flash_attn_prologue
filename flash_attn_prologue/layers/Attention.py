# Copyright (c) 2025 Achiya Deri.
# All rights reserved.

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import python built-in dependencies
from typing import Optional, Tuple, List, Union
import math

# Import third-party dependencies

# Import the internal dependencies
from .RoPE import apply_rope
from .RMSNorm import QKRMSNorm


class FlashRoPESelfAttention(nn.Module):
    """
    Self-Attention using FLASH_ATTENTION backend with RoPE support.
    """
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool = False, attn_drop: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.bias = bias
        self.attn_drop = attn_drop
        self.backends = [SDPBackend.FLASH_ATTENTION]
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.qk_norm = QKRMSNorm(self.head_dim)

        self.scales = nn.Parameter(torch.randn(self.num_heads, 1, 1))
        nn.init.constant_(self.scales, math.log(self.head_dim ** -0.5))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters of the self-attention layer."""
        import math
        nn.init.constant_(self.scales, math.log(self.head_dim ** -0.5))
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, 'reset_parameters') and callable(module.reset_parameters):
                module.reset_parameters()


    def forward(self, x: torch.Tensor, thetas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        
        qk, v_proj = qkv[:, :, :2], qkv[:, :, 2]
        
        # QK Norm
        qk = self.qk_norm(qk)
        
        if thetas is not None:
            sin, cos = thetas
            qk = apply_rope(qk, sin, cos)
            
        q_proj, k_proj = qk.unbind(2)
        q_proj = q_proj * self.scales.float().exp().to(q_proj.dtype)
        
        with sdpa_kernel(backends=self.backends):
            attn_output = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=self.attn_drop if self.training else 0.0, scale=1.0)
        
        output = self.out_proj(attn_output.transpose(1, 2).contiguous().view(B, N, -1))
        return output

