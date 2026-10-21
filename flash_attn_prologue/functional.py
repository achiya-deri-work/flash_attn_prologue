"""Stable, traceable Q/K RMSNorm -> split-half RoPE -> query scale API."""
import math
import torch


def prepare_backend(backend: str = "native") -> None:
    """Register optional CuTe operators before capturing a compiled graph."""
    if backend not in ("native", "cute"):
        raise ValueError("backend must be 'native' or 'cute'")
    if backend == "cute":
        from . import fused_epilogue  # noqa: F401


def qknorm_rope_scale(q, k, rmsnorm_weight, scales, sin, cos, eps=1e-5, *, backend="native"):
    """Normalize Q/K, apply half-width RoPE, and multiply Q by exp(scales).

    Q/K: matching packed [T,H,D] or dense [B,H,N,D], BF16/FP16/FP32.
    Weight: [2,1,D]. Log-scales: [H,1,1]. Both must be FP32.
    Packed sin/cos: [T,D/2] or [T,1,D/2]. Dense: [N,D/2],
    [B,N,D/2], or [B,1,N,D/2]; singleton B/N broadcasting is supported.
    Frequencies must be FP32; positions/resets are entirely caller-owned.
    All arithmetic through the final cast is FP32. Inputs are never mutated.
    Native supports all input gradients and CPU/CUDA; torch.compile may fuse it.
    CuTe is explicit, CUDA BF16 only, eps=1e-5, D=32/64, fixed frequencies.
    Call prepare_backend('cute') outside torch.compile when using it directly.
    No sequence padding or cu_seqlens is needed: this operation is per row.
    Attention must use softmax_scale=1 if this prologue applies 1/sqrt(D).
    """
    if backend not in ("native", "cute"):
        raise ValueError("backend must be 'native' or 'cute'")
    if q.ndim not in (3, 4) or q.shape != k.shape:
        raise ValueError("q/k must have matching packed [T,H,D] or dense [B,H,N,D] shapes")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("q/k must share BF16, FP16, or FP32 dtype")
    d, h = q.shape[-1], q.shape[-2] if q.ndim == 3 else q.shape[1]
    if d <= 0 or d % 2 or h <= 0:
        raise ValueError("head dimension must be positive/even and head count positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if rmsnorm_weight.shape != (2, 1, d) or scales.shape != (h, 1, 1):
        raise ValueError("weight must be [2,1,D], log-scales must be [H,1,1]")
    for value in (rmsnorm_weight, scales, sin, cos):
        if value.dtype != torch.float32:
            raise ValueError("weights, log-scales, and sin/cos must be FP32")
    if any(t.device != q.device for t in (k, rmsnorm_weight, scales, sin, cos)):
        raise ValueError("all tensors must share a device")
    if sin.shape != cos.shape or sin.shape[-1:] != (d // 2,):
        raise ValueError("sin/cos must have identical shapes and half-width D/2")
    if q.ndim == 3:
        if sin.ndim == 2:
            sin, cos = sin[:, None, :], cos[:, None, :]
        if sin.ndim != 3 or sin.shape[0] not in (1, q.shape[0]) or sin.shape[1] != 1:
            raise ValueError("packed sin/cos must be [T,D/2] or [T,1,D/2]")
        scale = scales.reshape(1, h, 1)
    else:
        if sin.ndim == 2:
            sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        elif sin.ndim == 3:
            sin, cos = sin[:, None, :, :], cos[:, None, :, :]
        if (sin.ndim != 4 or sin.shape[0] not in (1, q.shape[0])
                or sin.shape[1] != 1 or sin.shape[2] not in (1, q.shape[2])):
            raise ValueError("dense sin/cos must broadcast over [B,1,N,D/2]")
        scale = scales[None, :, :, :]
    if backend == "cute":
        if q.device.type != "cuda" or q.dtype != torch.bfloat16 or d not in (32, 64):
            raise ValueError("CuTe requires CUDA BF16 q/k and D=32 or 64")
        if eps != 1e-5:
            raise ValueError("CuTe currently supports only eps=1e-5")
        if sin.requires_grad or cos.requires_grad:
            raise ValueError("CuTe does not implement frequency gradients; use native")
        if q.numel() == 0:
            return qknorm_rope_scale(q, k, rmsnorm_weight, scales, sin, cos, eps)
        if q.ndim == 3:
            q4, k4 = q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None]
            s4, c4 = sin.permute(1, 0, 2)[None], cos.permute(1, 0, 2)[None]
        else:
            q4, k4, s4, c4 = q, k, sin, cos
        s4 = s4.expand(q4.shape[0], 1, q4.shape[2], d // 2)
        c4 = c4.expand_as(s4)
        if q4.stride(-1) != 1 or any(n > 1 and st == 0 for n, st in zip(q4.shape, q4.stride())):
            q4 = q4.contiguous()
        if k4.stride(-1) != 1 or any(n > 1 and st == 0 for n, st in zip(k4.shape, k4.stride())):
            k4 = k4.contiguous()
        out_q, out_k = torch.ops.flash_prologue.qknorm_rope_scale_cute(
            q4, k4, rmsnorm_weight, scales, s4, c4, eps)
        if q.ndim == 3:
            return out_q[0].permute(1, 0, 2), out_k[0].permute(1, 0, 2)
        return out_q, out_k
    def transform(x, weight):
        x = x.float()
        normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
        first, second = normed.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return (transform(q, rmsnorm_weight[0]) * scale.exp()).to(q.dtype), transform(k, rmsnorm_weight[1]).to(k.dtype)
