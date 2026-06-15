# Flash Attention Prologue with CuTe Python DSL

This project contains a high-performance custom CUDA kernel implementation for a fused QKV projection and epilogue using the NVIDIA CuTe Python DSL (`cutlass.cute`). It fuses linear QKV projection, RMSNorm, Rotary Position Embedding (RoPE), and query scaling into a single fused GPU kernel.

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
