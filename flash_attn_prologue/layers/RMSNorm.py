# Copyright (c) 2025 Achiya Deri.
# All rights reserved.

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Tuple
from ..inductor_fused import qknorm_rope_scale


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
    """Packed/dense prologue with FP32 master affine and log-scale parameters.

    initial_scale multiplies Q only; use 1.0 when attention applies 1/sqrt(D).
    The legacy default is 1/sqrt(D), requiring attention softmax_scale=1.
    native is traceable PyTorch (Inductor-fused when compiled); cute selects
    registered CUDA kernels explicitly, without an implicit length switch.
    State-dict names/shapes remain rmsnorm_weight [2,1,D], scales [H,1,1].
    """
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5,
                 *, initial_scale=None, backend: str = "native"):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0 or head_dim % 2:
            raise ValueError("num_heads must be positive and head_dim positive/even")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        initial_scale = head_dim ** -0.5 if initial_scale is None else initial_scale
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")
        from ..functional import prepare_backend
        prepare_backend(backend)
        self.num_heads, self.head_dim, self.eps = num_heads, head_dim, eps
        self.initial_scale, self.backend = initial_scale, backend
        self.rmsnorm_weight = nn.Parameter(torch.empty(2, 1, head_dim, dtype=torch.float32))
        self.scales = nn.Parameter(torch.empty(num_heads, 1, 1, dtype=torch.float32))
        self.reset_parameters()

    def _apply(self, fn, recurse=True):
        # Let PyTorch own Parameter/device/gradient conversion, while preserving
        # FP32 values rather than rounding them through a requested BF16 cast.
        def preserve_master(tensor):
            target = fn(tensor)
            if tensor.is_meta or target.is_meta:
                return target.float()
            return tensor.to(device=target.device, dtype=torch.float32)
        return super()._apply(preserve_master, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # assign=True must not replace FP32 masters with BF16 checkpoint tensors.
        for name in ("rmsnorm_weight", "scales"):
            key = prefix + name
            if key in state_dict:
                state_dict[key] = state_dict[key].float()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def bfloat(self):
        """Project convention alias; FP32 norm/scale masters remain FP32."""
        return self.bfloat16()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.rmsnorm_weight)
        nn.init.constant_(self.scales, math.log(self.initial_scale))

    def forward(self, q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> Tuple[Tensor, Tensor]:
        from ..functional import qknorm_rope_scale
        return qknorm_rope_scale(q, k, self.rmsnorm_weight, self.scales,
                                 sin, cos, self.eps, backend=self.backend)
