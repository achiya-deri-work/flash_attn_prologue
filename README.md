# Flash Attention Prologue

The supported frontend normalizes Q/K, applies split-half RoPE, and scales Q.
It accepts packed `[tokens, heads, head_dim]` or dense `[batch, heads, sequence,
head_dim]` tensors. The operation is per token: positions and per-image/prompt
RoPE resets belong to the caller, and no `cu_seqlens` or per-image padding is
needed here.

## Installation

```bash
pip install -e .
```

Quack is a normal `quack-kernels` dependency from PyPI. No cloned or patched
Quack checkout is required. The local `quack/` developer directory is ignored
and is not shipped in wheels. Package discovery includes the frontend layers
but excludes archived experiments, diagnostic scripts, and the separate Mamba
project. CUDA tests must use a compatible CUDA-enabled PyTorch installation.

## Public API

```python
import torch
from flash_attn_prologue import RMSNormRoPEScale, qknorm_rope_scale

# q/k: BF16 [tokens, heads, head_dim]
# sin/cos: FP32 [tokens, head_dim // 2]
prologue = RMSNormRoPEScale(
    num_heads=4, head_dim=32, initial_scale=1.0, backend="native"
).cuda().bfloat()
prologue.reset_parameters()  # Fresh initialization only; never reset a checkpoint.
compiled = torch.compile(prologue, fullgraph=True, dynamic=False)
q_out, k_out = compiled(q, k, sin, cos)
```

`initial_scale=1.0` is appropriate when the attention operator already applies
`1/sqrt(head_dim)`, including FlashDETR's packed window attention. The legacy
default is `1/sqrt(head_dim)`; use attention scale **1** with that default.
The learned `scales` parameter stores the logarithm of the multiplier applied
to Q, not K. Do not apply the attention scale twice.

The functional API is:

```python
q_out, k_out = qknorm_rope_scale(
    q, k, rmsnorm_weight, scales, sin, cos, eps=1e-5, backend="native"
)
```

- `rmsnorm_weight`: FP32 `[2, 1, head_dim]`.
- `scales`: FP32 `[heads, 1, 1]`.
- Packed frequencies: FP32 `[tokens, head_dim/2]` or `[tokens, 1, head_dim/2]`.
- Dense frequencies: FP32 `[sequence, head_dim/2]`, `[batch, sequence,
  head_dim/2]`, or `[batch, 1, sequence, head_dim/2]`, with supported singleton
  batch/sequence broadcasting. Frequencies are shared across heads.
- Norm reduction, affine multiplication, RoPE, and query scaling run in FP32;
  outputs are cast once to the Q/K input dtype. Inputs are not mutated.
- The module keeps FP32 norm and scale parameters through dtype conversion.
  State-dict keys remain `rmsnorm_weight` and `scales`.

## Backends and support boundaries

`native` is traceable PyTorch and supports CPU/CUDA, BF16/FP16/FP32 inputs,
arbitrary positive finite epsilon, and gradients through Q/K, affine weights,
log-scales, sine and cosine. It can be fused by Inductor; eager execution is
not a custom CuTe kernel.

`cute` explicitly selects the existing CUDA kernels. Its frontend currently
requires CUDA BF16 Q/K, head dimension 32 or 64, epsilon `1e-5`, and fixed
sine/cosine tensors without gradients. Unsupported requests raise errors.
There is no implicit short-sequence backend switch. The module prepares this
backend at construction; functional callers must call
`prepare_backend("cute")` before compilation. The backend is opaque to
Inductor, with registered fake and first-order autograd implementations.
Second-order CuTe gradients are not part of this support contract. The wrapper
may copy unusual channel-strided or misaligned views to satisfy kernel memory
requirements; ordinary packed QKV views do not require image padding.

Importing the public frontend does not eagerly load GPU kernels or override
CUDA architecture environment variables. Legacy exports remain lazy and are
separate from the qualified frontend surface. See the [qualification report](reports/frontend_api_5060_20261021/REPORT.md)
for executed shapes, gradients, compiler checks, and remaining limitations;
frontend stabilization does not qualify every archived GEMM/Mamba experiment.

## Historical implementation notes

The following notes describe earlier experiments and historical performance
claims. They are not the current frontend contract or a fresh qualification.
In particular, the current standalone CuTe norm/RoPE/scale operator allocates
separate outputs; it does not mutate Q/K in place.

## Features

- **Fused Execution:** Combines GEMM (QKV projection) with epilogue operations (RMSNorm, RoPE, and scaling) to eliminate global memory roundtrips.
- **CuTe Python DSL Implementation:** Written using standard CuTe DSL coordinate structures, tiling, layouts, and copy/MMA primitives.
- **cp.async Memory Transfer:** Leverages asynchronous global-to-shared memory copy operations (`cp.async`) targeting modern NVIDIA GPUs.
- **Tiled GEMM Loop:** Features a tiled K-loop (`BLOCK_K_TILE = 32`) to manage shared memory usage (within 64 KB limits) and optimize throughput.
- **Split-N Optimization:** A specialized cooperatively partitioned threadblock design (`fused_qkv_cute_splitn`) targeting consumer Blackwell platforms (`sm_120a`) to avoid TMEM hardware requirements.
- Triton Standalone Epilogue: A standalone Triton kernel (`qknorm_rope_scale`) fusing epilogue operations after a standard PyTorch linear layer.
- **CuTe Standalone Epilogue:** High-performance, layout-agnostic standalone forward and backward CuTe epilogue operators (`qknorm_rope_scale_cute_op` / `qknorm_rope_scale_cute_backward_op`) that mutate separate Q/K tensors in-place, specialize on strides at JIT compile time to eliminate `.contiguous()` conversions, and run up to 2.33x faster than Triton.
- **Blackwell TMA-based Backward GEMM Kernels:** Features high-performance JIT-compiled backward GEMM kernels (`matmul_bwd_dx` and `matmul_bwd_dw`) utilizing hardware TMA loads, asynchronous `mbarrier` transaction barriers, and a custom on-chip JIT Split-K reduction kernel on SM120.
- **Nested Incremental Counter Loop Optimization:** Eliminates dynamic integer division and modulo calculations from the hot path of dynamic loops in backward kernels, replacing them with fast nested incremental counter registers.
- **Dynamic Register Fine-Tuning & Register-Level Epilogue Cast:** Minimizes register pressure via dynamic nvvm setmaxregister controls and completely eliminates dynamic shared memory epilogue writeback buffer allocations by casting accumulator fragments directly in registers and storing via BFloat16 MMA partition layout maps.
- **Autotuning Grid Swizzling:** Autotunes the grid swizzle factors (`swizzle_x` values `[1, 2, 4, 8, 16]`) to optimize hardware L2 cache saturation on SM120.
- **Bandwidth-Optimized RoPE Buffer Size & On-Chip Reuse:** Accepts `sin`/`cos` buffers of shape `[..., dhead // 2]` directly (halving memory footprint and loading bandwidth), and processes the positional rotation twice on-chip inside the FWD/BWD JIT kernels.
- **Inverse Epilogue Operator (Activation Recomputation):** Standalone memory-efficient inverse epilogue operator (`qknorm_rope_scale_cute_inverse_op`) that reconstructs the forward inputs $Q$ and $K$ on-chip in registers during the backward pass using the forward outputs and low-overhead scale factors (`rms_q`, `rms_k`), eliminating input VRAM storage and saving 64x activation memory.
- **torch.compile Integration:** Fully compatible with `torch.compile` via custom PyTorch operator registration and custom schema definitions.
- **Temporal Mamba 3 Scan Kernels:** Coordinate-descent autotuned forward and backward scan kernels using the CuTe Python DSL. Features independent FWD/BWD profiling and caching, a hybrid search algorithm (wide grid search followed by deep coordinate descent), warp-shuffle elimination via shared memory broadcasts in the backward pass, and advanced performance optimizations (two-pass parameter gradient workspace reduction to eliminate global atomic collisions, compile-time AST rank loop unrolling, and direct-to-global epilogue store for states to bypass 64KB SMEM writeback buffers), achieving 100% correctness parity and up to **2.08x speedup** (FWD+BWD) vs PyTorch Inductor.
- **Cloned RoPE Self-Attention & Mamba-3 Single Step Update Compilation Parity:** Tests the cloned modules from `Advantage-First-RL` under `torch.compile(fullgraph=True)` with fixes for stride-based positional broadcasting (zero-stride batching) and protection against Inductor-reused gradient buffer garbage in the padded region.

## Installation

Ensure you are inside the virtual environment:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -e .
```

## Running Tests and Benchmarks

Verify correctness of the package with active tests (no extra environment variables required):

- **Verify the compilation parity and numerical correctness of Cloned RoPE Attention and Mamba-3 Single Step Update modules**:
  ```bash
  pytest tests/test_rope_mamba_compile.py
  ```

- **Verify the cloned and adapted TMA-based backward GEMM kernels (`matmul_bwd_dx` and `matmul_bwd_dw`)**:
  ```bash
  pytest tests/test_gemm_bwd_clone.py
  ```

- **Verify and benchmark the fully integrated Fused QKV Forward + Fused TMA-based Backward pass**:
  ```bash
  pytest -s tests/test_fused_qkv_bwd.py
  ```

- **Verify the main Fused QKV Forward operator correctness**:
  ```bash
  pytest tests/test_fused_qkv.py
  ```

- **Verify the Fused QKV Epilogue chained operator**:
  ```bash
  pytest tests/test_fused_qkv_epilogue.py
  ```

- **Verify the Temporal Mamba 3 FWD/BWD CuTe kernels parity against eager PyTorch**:
  ```bash
  python temporal_mamba3/test_mamba3_parity.py
  ```

- **Benchmark and compare the Temporal Mamba 3 CuTe kernel against eager PyTorch and PyTorch Inductor Compiled**:
  ```bash
  python temporal_mamba3/benchmark_mamba3.py
  ```
