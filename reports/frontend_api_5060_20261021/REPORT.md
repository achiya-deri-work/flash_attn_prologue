# Frontend qualification — RTX 5060 Laptop, 2026-10-21

This report qualifies the public `qknorm_rope_scale` function and `RMSNormRoPEScale`
module for packed/dense norm, half-width split-half RoPE, and query scaling.
It does not qualify the legacy fused GEMM, Mamba, or inverse kernels.

## Public contract

- Matching packed `[T,H,D]` or dense `[B,H,N,D]` Q/K. No padding or sequence
  reductions: caller-supplied sin/cos encode positions and prompt/image resets.
- FP32 norm weights `[2,1,D]`, log-scales `[H,1,1]`, and half-width sin/cos.
  Normalize, affine, rotate, exponentiate scale, and scale Q in FP32 before the
  final activation cast. No bias. `initial_scale=1` avoids duplicating attention's
  own `1/sqrt(D)`; legacy default remains `1/sqrt(D)`.
- `native`: traceable PyTorch, CPU/CUDA, BF16/FP16/FP32, gradients for all six
  tensor inputs. `torch.compile` can fuse it; this is not the CuTe kernel.
- `cute`: explicit registered CUDA BF16 kernel, D=32/64, epsilon=1e-5,
  non-differentiable frequency tables. `prepare_backend('cute')` registers the
  optional operators outside compilation; module construction does this for you.
  Unsupported epsilon and frequency gradients are rejected, not silently ignored.
- Strided packed Q/K use view adaptation. Unsupported channel-strided or expanded
  activation layouts are materialized by the frontend. Runtime alignment repair
  may copy misaligned storage-offset inputs/gradients. No per-image padding.
- Masters retain FP32 values through `.bfloat16()` / `.bfloat()`, ordinary and
  `assign=True` checkpoint loads, and device conversion. Fresh `.bfloat()` then
  `reset_parameters()` is supported. Meta construction/materialization is tested.

## Installation and provenance

The wheel declares `torch>=2.8` and `quack-kernels>=0.6.5,<0.7`. Qualification used
published **quack-kernels 0.6.5**, not the dirty nested checkout. The local Quack
checkout remains on disk, ignored and no longer a Git dependency.

- Python 3.13.16, Torch 2.14.1+cu132, CUTLASS DSL 4.7.0.
- RTX 5060 Laptop, SM120, 8 GiB, WSL.
- Wheel `flash_attn_prologue-0.2.0-py3-none-any.whl` SHA256:
  `ea78de58fdf7b08daf3fa7240d5adc2b5d03e158570a17c3778f90a83f4f702f`.
- Isolated install: `/home/achiya/prologue-api-20261021/deps`.
- Imported Quack: `deps/quack/__init__.py`; imported prologue:
  `deps/flash_attn_prologue/__init__.py`, verified from `/tmp`.
- Wheels were extracted into this isolated import root; existing CUDA/runtime
  dependencies came from `/home/achiya/flashdetr-5060-20261020/.venv`.
- Wheel inventory contains the previously omitted `layers/RMSNorm.py` and no
  nested Quack, archive, tests, or unrelated `temporal_mamba3` package.
- Basic public import neither imports Quack/CUTLASS nor mutates environment.
  Legacy CUDA entrypoints remain lazy imports; legacy callable aliases were
  smoke-tested independently of GEMM numerical qualification.

## Correctness and framework evidence

GPU: **61 passed** (53 focused frontend tests and 8 updated existing module tests).
CPU with CUDA hidden: **23 passed, 38 skipped**. CUDA skips are explicit.

The independent oracle uses FP64 vector norms and explicit 2x2 rotation matrices;
it does not invoke production norm/rotation helpers. Identical upstream gradients
are used. Compare Q/K outputs and Q/K, norm-weight, log-scale gradients; native
also compares sin/cos gradients. Finite checks precede error comparisons.

Gates: FP32 `atol=5e-4, rtol=1e-4`; BF16/FP16 `atol=0.01, rtol=0.01`; normalized
RMS error `<0.001` when reference RMS exceeds `1e-5`. Normalized RMS denominator
is reference RMS floored at `1e-8`. Per-case measurements are in
`numerical_metrics.jsonl`.

Coverage includes:

- T=0/1/63/64/65/95/96/127/128/129/6400, D=32/64, four heads, packed and dense.
- Both backends BF16; native FP16/FP32; nondefault native epsilon.
- Tail lengths, noncontiguous and storage-offset inputs, noncontiguous and
  misaligned upstream gradients, channel-transposed and expanded activations.
- Explicit rejection at the raw CuTe boundary for unsupported channel stride
  or expanded activations; frontend handles them without aliased output writes.
- `torch.library.opcheck`: schema, autograd registration, fake tensor, AOT dynamic
  dispatch, including misaligned packed views. Outputs preserve the fake/runtime
  stride agreement after alignment repair.
- `torch.compile(fullgraph=True, dynamic=False)` forward/backward, one graph per
  backend across changed frequency values. Ten repeated training calls after
  warmup show exactly unchanged live CUDA allocation.
- Single-output losses, three AdamW updates, activation checkpointing, BF16
  autocast, state-dict reload, exact FP32-master preservation.

CuTe ordinary forward/backward use a conservative fixed configuration
(BLOCK_M=64, HEAD_GROUP=1, four warps). First use no longer triggers an unbounded
shape autotuning sweep or filesystem cache mutation. Kernel compilation remains
cached inside the opaque registered operator. Higher-order derivatives and
export were not part of this qualification.

## FlashDETR composition

`prologue_window_integration.py` composes each prologue backend with the packed,
shifted-window attention kernel and fused-projection strided Q/K/V. It tests two
images, terminal padding, shift=4, and static fullgraph compilation while runtime
geometry changes from `(8x8,13x17)` to `(9x9,12x16)`. It compares outputs and all
QKV, sink-vector, norm-weight and log-scale gradients. `initial_scale=1` ensures
attention scales logits exactly once. Results are in `window_integration.json`.
This proves this composition, not the complete detector training system.

## Timing scope

`timing.json` contains 30 CUDA-event samples after five warmups for compiled
native/CuTe, T=64 and T=6400, H=4,D=32, forward and forward+backward. These timings
include CPU dispatch gaps, allocation/zero-grad bookkeeping, and no CUDA-graph
replay. Variability is material; they do not establish kernel throughput or a
reliable speedup. Native remains the portable default; backend choice is explicit.

## Failures found and fixed

The preserved `repaired_channel_layout_failure.txt` shows an adversarial
channel-transposed CuTe case failing before the frontend materialization fix.
Other review fixes prevent ignored nondefault epsilon, missing frequency gradients,
misaligned pointer assumptions, fake/runtime stride disagreement, expanded-output
aliasing, BF16 master rounding, and surprise backend switching at 96 tokens.
No failures were hidden by weakening the numerical gates. One test-only assertion
was corrected because native autograd may legitimately return None for an unused K
branch rather than an explicit zero tensor.

## Reproduction

```bash
# On personal_laptop, using the isolated wheel import root:
cd /home/achiya/prologue-api-20261021
PYTHONPATH=/home/achiya/prologue-api-20261021/deps \
PROLOGUE_TEST_DEVICE=cuda \
PROLOGUE_METRICS=metrics.jsonl \
/home/achiya/flashdetr-5060-20261020/.venv/bin/python -m pytest \
  prologue-api-tests.py test_fused_epilogue_module.py -q

# Source-checkout CPU frontend tests, without using local GPUs:
CUDA_VISIBLE_DEVICES='' python -m pytest \
  tests/test_frontend_api.py tests/test_fused_epilogue_module.py -q
```

The remote `prologue-api-tests.py` is an exact copy of `tests/test_frontend_api.py`.
The integration script needs the separately qualified FlashDETR window-attention
checkout on PYTHONPATH; it is included as evidence rather than a package dependency.
