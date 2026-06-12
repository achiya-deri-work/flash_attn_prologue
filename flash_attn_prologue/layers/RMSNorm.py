# Copyright (c) 2025 Achiya Deri.
# All rights reserved.

import torch
from torch import Tensor, nn
from typing import Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).

    A simpler and more efficient alternative to LayerNorm.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-5, const=False):
        """Initialize RMSNorm."""
        super().__init__()
        if not const:
            self.weight = nn.Parameter(torch.ones(hidden_dim))
        else:
            # pyrefly: ignore [bad-assignment]
            self.weight = None
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters to default."""
        if self.weight is not None:
            nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor: 
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        return output


class QKRMSNorm(nn.Module):
    """
    RMSNorm specialized for Query/Key normalization in Attention.
    
    Normalizes the last dimension of Q and K.
    """
    def __init__(self, head_dim: int, eps: float = 1e-5):
        """Initialize QKRMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 1, head_dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
           x: Input tensor containing Q and K.
        Returns:
           Normalized tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SplitQKRMSNorm(nn.Module):
    """
    RMSNorm specialized for Cross-Attention Query/Key normalization.
    
    Maintains separate weights for Query and Key normalization.
    """
    def __init__(self, head_dim: int, eps: float = 1e-5):
        """Initialize SplitQKRMSNorm."""
        super().__init__()
        self.q_weight = nn.Parameter(torch.ones(1, head_dim))
        self.k_weight = nn.Parameter(torch.ones(1, head_dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.constant_(self.q_weight, 1)
        nn.init.constant_(self.k_weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for split Q and K.
        
        Args:
           q: Query tensor.
           k: Key tensor.
        Returns:
           Tuple of (normalized_q, normalized_k).
        """
        q_out = (self._norm(q.float()).type_as(q)) * self.q_weight
        k_out = (self._norm(k.float()).type_as(k)) * self.k_weight
        return q_out, k_out

class RMSNorm2d(nn.Module):
    """
    RMSNorm for 2D spatial inputs [B, C, H, W].
    Normalizes over the channel dimension (dim=1).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, const=False):
        super().__init__()
        if not const:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        else:
            # pyrefly: ignore [bad-assignment]
            self.weight = None
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters to default."""
        if self.weight is not None:
            nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        return output


class RMSNormRoPEScale(nn.Module):
    """
    Fused epilogue layer implementing QK normalization, RoPE, and scaling in a single JIT operator.
    Stores weights and scales in BFloat16 and takes split Q and K.
    """
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        
        # Store weights and scales as BFloat16 parameters
        self.rmsnorm_weight = nn.Parameter(torch.ones(2, 1, head_dim, dtype=torch.bfloat16))
        self.scales = nn.Parameter(torch.empty(num_heads, 1, 1, dtype=torch.bfloat16))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        import math
        nn.init.constant_(self.rmsnorm_weight, 1.0)
        nn.init.constant_(self.scales, math.log(self.head_dim ** -0.5))
        
    def forward(self, q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.flash_prologue.qknorm_rope_scale_cute(
            q, k, self.rmsnorm_weight, self.scales, sin, cos, self.eps
        )

