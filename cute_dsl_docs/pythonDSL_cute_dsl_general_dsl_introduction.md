

Introduction — NVIDIA CUTLASS Documentation



[Skip to main content](#main-content)

Back to top




`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../../index.html)

Choose version

Search
`Ctrl`+`K`

Search
`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../../index.html)

Choose version

Table of Contents

* [Changelog](../../../../CHANGELOG.html)

CuTe DSL

* [Overview](../overview.html)
* [Functionality](../functionality.html)
* [Quick Start Guide](../quick_start.html)
* [CuTe DSL](../cute_dsl.html)
  + [Introduction](#)
  + [Code Generation](dsl_code_generation.html)
  + [Control Flow](dsl_control_flow.html)
  + [JIT Argument Generation](dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](dsl_dynamic_layout.html)
  + [JIT Caching](dsl_jit_caching.html)
  + [JIT Compilation Options](dsl_jit_compilation_options.html)
  + [JIT Types](types.html)
  + [Integration with Frameworks](framework_integration.html)
  + [Debugging with the DSL](debugging.html)
  + [Autotuning with the DSL](autotuning_gemm.html)
  + [Educational Notebooks](notebooks.html)
  + [Deprecation Policy](../deprecation.html)
  + [Compile with TVM FFI](compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](resources.html)
* [CuTe DSL API](../cute_dsl_api.html)
  + [changelog](../cute_dsl_api/changelog.html)
  + [cute](../cute_dsl_api/cute.html)
    - [arch](../cute_dsl_api/cute_arch.html)
    - [Runtime](../cute_dsl_api/cute_runtime.html)
  + [cute\_nvgpu](../cute_dsl_api/cute_nvgpu.html)
    - [Common](../cute_dsl_api/cute_nvgpu_common.html)
    - [warp submodule](../cute_dsl_api/cute_nvgpu_warp.html)
    - [warpgroup submodule](../cute_dsl_api/cute_nvgpu_warpgroup.html)
    - [cpasync submodule](../cute_dsl_api/cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](../cute_dsl_api/cute_nvgpu_tcgen05.html)
  + [pipeline](../cute_dsl_api/pipeline.html)
  + [utils](../cute_dsl_api/utils.html)
    - [Utilities for SM90](../cute_dsl_api/utils_sm90.html)
    - [Utilities for SM100](../cute_dsl_api/utils_sm100.html)
* [Limitations](../limitations.html)
* [FAQs](../faqs.html)

CUTLASS C++

* [Overview](../../../../overview.html)
* [Getting Started](../../cpp/getting_started.html)
  + [Quickstart](../../cpp/quickstart.html)
  + [IDE Setup](../../cpp/ide_setup.html)
  + [Build](../../cpp/build/index.html)
    - [Building on Windows with Visual Studio](../../cpp/build/building_in_windows_with_visual_studio.html)
    - [Building with Clang as host compiler](../../cpp/build/building_with_clang_as_host_compiler.html)
  + [Functionality](../../cpp/functionality.html)
  + [Terminology](../../cpp/terminology.html)
  + [Fundamental Types](../../cpp/fundamental_types.html)
  + [Programming Guidelines](../../cpp/programming_guidelines.html)
  + [GEMM Heuristics](../../cpp/heuristics.html)
* [Efficient GEMM in CUDA](../../cpp/efficient_gemm.html)
* [Synchronization primitives](../../cpp/pipeline.html)
* [CUTLASS Profiler](../../cpp/profiler.html)
* [Dependent Kernel Launch](../../cpp/dependent_kernel_launch.html)
* [Blackwell Specific](../../cpp/blackwell.html)
  + [Blackwell SM100 GEMMs](../../cpp/blackwell_functionality.html)
  + [Blackwell Cluster Launch Control](../../cpp/blackwell_cluster_launch_control.html)
* [CuTe](../../cpp/cute/index.html)
  + [00\_quickstart](../../cpp/cute/00_quickstart.html)
  + [01\_layout](../../cpp/cute/01_layout.html)
  + [02\_layout\_algebra](../../cpp/cute/02_layout_algebra.html)
  + [03\_tensor](../../cpp/cute/03_tensor.html)
  + [04\_algorithms](../../cpp/cute/04_algorithms.html)
  + [0t\_mma\_atom](../../cpp/cute/0t_mma_atom.html)
  + [0x\_gemm\_tutorial](../../cpp/cute/0x_gemm_tutorial.html)
  + [0y\_predication](../../cpp/cute/0y_predication.html)
  + [0z\_tma\_tensors](../../cpp/cute/0z_tma_tensors.html)
* [CUTLASS 3.x](../../cpp/cutlass_3x.html)
  + [Design](../../cpp/cutlass_3x_design.html)
  + [GEMM Backwards Compatibility](../../cpp/cutlass_3x_backwards_compatibility.html)
  + [GEMM API](../../cpp/gemm_api_3x.html)
* [CUTLASS 2.x](../../cpp/cutlass_2x.html)
  + [Layouts and Tensors](../../cpp/layout.html)
  + [GEMM API](../../cpp/gemm_api.html)
  + [Tile Iterator Concepts](../../cpp/tile_iterator_concept.html)
  + [Utilities](../../cpp/utilities.html)
* [Code Organization](../../cpp/code_organization.html)
* [Grouped Kernel Schedulers](../../cpp/grouped_scheduler.html)
* [CUTLASS Convolution](../../cpp/implicit_gemm_convolution.html)

Reference

* [Software License Agreement](../license.html)

* [CuTe DSL](../cute_dsl.html)
* Introduction

# Introduction[#](#introduction "Link to this heading")

## Overview[#](#overview "Link to this heading")

CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of
high-performance GPU kernels. It evolved from the C++ CUTLASS library and is
now available as a decorator-based DSL.

Its primary goals are:

* **Zero-cost abstraction**, DSL is a zero-cost abstraction thanks to Hybrid DSL approach.
* **Consistent with CuTe C++**, allowing users to express GPU kernels with full
  control of the hardware.
* **JIT compilation** for both host and GPU execution.
* [DLPack](https://github.com/dmlc/dlpack) **integration**, enabling seamless
  interop with frameworks (e.g., PyTorch, JAX).
* **JIT caching**, so that repeated calls to the same function benefit from
  cached IR modules.
* **Native types and type inference** to reduce boilerplate and improve
  performance.
* **Optional lower-level control**, offering direct access to GPU backends or
  specialized IR dialects.

## Decorators[#](#decorators "Link to this heading")

CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1. `@jit` â Host-side JIT-compiled functions
2. `@kernel` â GPU kernel functions

Both decorators can optionally use a **preprocessor** that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying IR.

### `@jit`[#](#jit "Link to this heading")

Declares JIT-compiled functions that can be invoked from Python or from other CuTe DSL functions.

**Decorator Parameters**:

* `preprocessor`:

  + `True` (default) â Automatically translate Python flow control (e.g., loops, if-statements) into IR operations.
  + `False` â No automatic expansion; Python flow control must be handled manually or avoided.

**Call-site Parameters**:

* `no_cache`:

  + `True` â Disables JIT caching, forcing a fresh compilation each call.
  + `False` (default) â Enables caching for faster subsequent calls.

### `@kernel`[#](#kernel "Link to this heading")

Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

**Decorator Parameters**:

* `preprocessor`:

  + `True` (default) â Automatically expands Python loops/ifs into GPU-compatible IR operations.
  + `False` â Expects manual or simplified kernel implementations.

**Kernel Launch Parameters**:

* `grid`
  Specifies the grid size as a list of integers.
* `block`
  Specifies the block size as a list of integers.
* `cluster`
  Specifies the cluster size as a list of integers.
* `smem`
  Specifies the size of shared memory in bytes (integer).

  + `None` (default) â Automatically calculates the kernelâs shared memory usage via **utils.SmemAllocator**. Recommended unless manual control is required.
  + `int` â Manually specifies the size of shared memory in bytes.

**Additional Kernel Launch Parameters**:

* `fallback_cluster`
  Specifies the minimum-guaranteed cluster size. When set, `cluster` becomes the **preferred** size, enabling graceful degradation when hardware cannot satisfy the preferred dimensions.

  + `None` (default) â No fallback; `cluster` is used directly.
  + `list[int]` â Three-element list [x, y, z].
* `max_number_threads`
  Specifies the maximum thread count per block (**maxntid**).

  + `[0, 0, 0]` (default) â Auto-generate **reqntid** from `block`.
  + `list[int]` â Three-element list [x, y, z].
* `min_blocks_per_mp`
  Specifies the minimum blocks per multiprocessor (**minctasm**).

  + `0` (default) â No minimum occupancy hint.
  + `int` â Minimum number of blocks per multiprocessor.
* `use_pdl`
  Enables Programmatic Dependent Launch (PDL) to overlap dependent kernel launches in the same stream.

  + `False` (default) â PDL disabled.
  + `True` â PDL enabled.
* `cooperative`
  Enables cooperative kernel launch; all thread blocks launch cooperatively with grid-wide synchronization support.

  + `False` (default) â Standard kernel launch.
  + `True` â Cooperative kernel launch.

## Calling Conventions[#](#calling-conventions "Link to this heading")

| **Caller** | **Callee** | **Allowed** | **Compilation/Runtime** |
| --- | --- | --- | --- |
| Python function | `@jit` | â | DSL runtime |
| Python function | `@kernel` | â | N/A (error raised) |
| `@jit` | `@jit` | â | Compile-time call, inlined |
| `@jit` | Python function | â | Compile-time call, inlined |
| `@jit` | `@kernel` | â | Dynamic call via GPU driver or runtime |
| `@kernel` | `@jit` | â | Compile-time call, inlined |
| `@kernel` | Python function | â | Compile-time call, inlined |
| `@kernel` | `@kernel` | â | N/A (error raised) |

[previous

CuTe DSL](../cute_dsl.html "previous page")
[next

End-to-End Code Generation](dsl_code_generation.html "next page")

On this page

* [Overview](#overview)
* [Decorators](#decorators)
  + [`@jit`](#jit)
  + [`@kernel`](#kernel)
* [Calling Conventions](#calling-conventions)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.