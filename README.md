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
- **Pipelined cp.async GEMM Kernel:** An optimized 3-stage pipelined GEMM kernel (`pipelined_gemm_cute_op`) utilizing `cp.async` copy groups for memory latency overlap instead of TMA.
- **torch.compile Integration:** Fully compatible with `torch.compile` via custom PyTorch operator registration and custom schema definitions.

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

Verify the base CuTe forward pass compilation and correctness:
```bash
PYTHONPATH=. python tests/test_fused_qkv_cute.py
```

Verify the Split-N CuTe kernel forward/backward correctness and benchmark performance:
```bash
PYTHONPATH=. python tests/test_fused_qkv_cute_splitn.py
```

Verify the Triton standalone epilogue kernel:
```bash
PYTHONPATH=. pytest tests/test_epilogue_kernel.py
```

Run the detailed component latency breakdown benchmark (GEMM vs Epilogue):
```bash
PYTHONPATH=. python tests/test_pure_gemm.py
```

Verify the Pipelined cp.async GEMM kernel forward correctness and benchmark performance:
```bash
PYTHONPATH=. python tests/test_pipelined_gemm.py
```

Verify the Milestone 1 Fused QKV (GEMM + RMSNorm) kernel correctness and benchmark performance:
```bash
PYTHONPATH=. python tests/test_fused_pipelined.py
```

Verify the resource-optimized Fused QKV Overlap kernel:
```bash
PYTHONPATH=. CUTE_DSL_LIBS=/home/achiya/vs_code_main/flash_attn_prologue/.venv/lib/python3.13/site-packages/nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so .venv/bin/python tests/test_fused_qkv_overlap.py
```

Verify the cloned and adapted backward GEMM kernels (matmul_bwd_dx and matmul_bwd_dw):
```bash
PYTHONPATH=./quack:. CUTE_DSL_LIBS=/home/achiya/vs_code_main/flash_attn_prologue/.venv/lib/python3.13/site-packages/nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so .venv/bin/python tests/test_gemm_bwd_clone.py
```
