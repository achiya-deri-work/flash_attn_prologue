# Copyright (c) 2025 Achiya Deri.
# All rights reserved.

import torch

def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    x might be q or k or qk, of shape [batch, num_heads, (1/2), seq_len, d_head]
    sin/cos of shape [(batch/1), (num_heads/1), (seq_len), (d_head)]
    """
    x_dtype = x.dtype
    rope_dtype = cos.dtype

    assert x.shape[-1] == cos.shape[-1], "cos/sin dim must be equal to x dim"
    assert rope_dtype == torch.float32, "cos/sin must be float32 for numerical stability"

    # FIX: Capture the original shape to prevent the 5D leak
    original_shape = x.shape

    # Standardize to 5D for unified math
    assert x.ndim in (4, 5), f"Invalid input shape: {x.shape}"
    if x.ndim == 4:
        batch, num_heads, seq_len, d_head = x.shape
        x = x.view(batch, num_heads, 1, seq_len, d_head)
    else:  # x.ndim == 5
        batch, num_heads, _, seq_len, d_head = x.shape

    # Broadcast cos/sin to 5D using unsqueeze for native broadcasting
    assert cos.ndim in (2, 3, 4, 5), f"Invalid cos/sin shape: {cos.shape}"
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1).unsqueeze(1)
        sin = sin.unsqueeze(1).unsqueeze(1)
    elif cos.ndim == 4:
        # Insert QK dim before seq_len
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    x_fp32 = x.to(torch.float32)

    # EXPLICIT CHUNKING: Maximum compiler fusion, zero hidden allocations
    x1, x2 = x_fp32.chunk(2, dim=-1)
    c1, c2 = cos.chunk(2, dim=-1)
    s1, s2 = sin.chunk(2, dim=-1)

    # Forward rotation
    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2

    out = torch.cat((out1, out2), dim=-1).to(x_dtype)

    # FIX: Restore the original 4D or 5D shape seamlessly
    return out.view(original_shape)
