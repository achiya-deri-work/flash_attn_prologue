

Changelog for CuTe DSL API changes — NVIDIA CUTLASS Documentation



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
  + [Introduction](../cute_dsl_general/dsl_introduction.html)
  + [Code Generation](../cute_dsl_general/dsl_code_generation.html)
  + [Control Flow](../cute_dsl_general/dsl_control_flow.html)
  + [JIT Argument Generation](../cute_dsl_general/dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](../cute_dsl_general/dsl_dynamic_layout.html)
  + [JIT Caching](../cute_dsl_general/dsl_jit_caching.html)
  + [JIT Compilation Options](../cute_dsl_general/dsl_jit_compilation_options.html)
  + [JIT Types](../cute_dsl_general/types.html)
  + [Integration with Frameworks](../cute_dsl_general/framework_integration.html)
  + [Debugging with the DSL](../cute_dsl_general/debugging.html)
  + [Autotuning with the DSL](../cute_dsl_general/autotuning_gemm.html)
  + [Educational Notebooks](../cute_dsl_general/notebooks.html)
  + [Deprecation Policy](../deprecation.html)
  + [Compile with TVM FFI](../cute_dsl_general/compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](../cute_dsl_general/dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](../cute_dsl_general/resources.html)
* [CuTe DSL API](../cute_dsl_api.html)
  + [changelog](#)
  + [cute](cute.html)
    - [arch](cute_arch.html)
    - [Runtime](cute_runtime.html)
  + [cute\_nvgpu](cute_nvgpu.html)
    - [Common](cute_nvgpu_common.html)
    - [warp submodule](cute_nvgpu_warp.html)
    - [warpgroup submodule](cute_nvgpu_warpgroup.html)
    - [cpasync submodule](cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](cute_nvgpu_tcgen05.html)
  + [pipeline](pipeline.html)
  + [utils](utils.html)
    - [Utilities for SM90](utils_sm90.html)
    - [Utilities for SM100](utils_sm100.html)
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

* [CuTe DSL API](../cute_dsl_api.html)
* Changelog for CuTe DSL API changes

# Changelog for CuTe DSL API changes[#](#changelog-for-cute-dsl-api-changes "Link to this heading")

## [4.3.0](https://github.com/NVIDIA/cutlass/releases/tree/main) (2025-10-20)[#](#id2 "Link to this heading")

* Debuggability improvements:

  + Supported source location tracking for DSL APIs
  + Supported dumping PTX and CUBIN
* Removed deprecated `cutlass.<arch>_utils.SMEM_CAPACITY["<arch_str>"]` and `cutlass.utils.ampere_helpers`
* Supported calling nested functions without capturing variables inside dynamic control flow
* Replaced usage of `cute.arch.barrier` in examples with corresponding APIs in `pipeline`

  + Use `pipeline.sync` for simple cases like synchronizing the whole CTA
  + Use `pipeline.NamedBarrier` to customize barriers with different participating threads and barrier id
* Added new APIs `repeat` and `repeat_as_tuple`
* Added new APIs `make_rmem_tensor` to create tensor in register memory (replace `make_fragment` with better naming)
* Added new APIs `make_rmem_tensor_like` which create rmem tensor from a tensor using the same shape with compact col-major strides
* Added `TmemAllocator` for allocating tensor memory
* Updated `SmemAllocator.allocate` to support allocation of a single scalar value
* Fixed `TensorSSA.reduce` to support static value as initial value
* Updated docstring for following APIs to be more concise and easier to understand:

  + `make_layout_tv`
  + `is_static`
  + `PipelineAsync`
  + `SmemAllocator`
* Fixed documentation for `pipeline`, `utils` and `cute.math` (`cute.math` is part of top level documentation)

## [4.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.0) (2025-09-10)[#](#id4 "Link to this heading")

* Added back `cute.make_tiled_copy` per the request from community
* Added support for explicit and implicit broadcast in `TensorSSA`

  + `cutlass.cute.TensorSSA`: support `broadcast_to` and implicit broadcasting for binary operations.
* Supported printing `TensorSSA` value in `cutlass.cute.print_tensor`
* Updated `cute.gemm` to support all dispatch patterns and improved checks for illegal inputs
* Introduced automatic kernel smem usage calculation for launch config.
* Introduced per op fast-math control for math ops(e.g. `exp`, `exp2`, `log2`, `log`)
* Introduced `CopyReduceBulkTensorTileS2GOp` in [tcgen05/copy.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py) to support TMA Reduce.

## [4.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.1.0) (2025-07-16)[#](#id6 "Link to this heading")

* for loop

  + Python built-in `range` now always generates codes and executes at runtime
  + `cutlass.range` is advanced `range` with kernel code level unrolling and pipelining control
  + Deprecated `cutlass.range_dynamic`, please replace with `range` or `cutlass.range`
  + **Experimental** Added `pipelining` control for compiler generated software pipeline code
* while/if

  + `while`/`if` now by default generates codes and executes at runtime unless `cutlass.const_expr` is specified for the predicate
  + Deprecated `cutlass.dynamic_expr`, please remove it
* Rename mbarrier functions to reduce ambiguity
* Modify SyncObject API (`MbarrierArray`, `NamedBarrier`, `TmaStoreFence`) to match `std::barrier`
* Change pipeline `create` function to take only keyword arguments, and make `barrier_storage` optional.
* Introduce `cutlass.cute.arch.get_dyn_smem_size` api to get runtime dynamic shared memory size.
* Various API Support for SM100 BlockScaled Gemm

  + Introduce BlockScaled MmaOps in [tcgen05/mma.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py), and provide a `make_blockscaled_trivial_tiled_mma` function in [blackwell\_helpers.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py) to help construct a BlockScaled TiledMma.
  + Introduce S2T CopyOps in [tcgen05/copy.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py).
  + Introduce BlockScaled layout utilities in [blockscaled\_layout.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blockscaled_layout.py) for creating the required scale factor layouts in global memory, shared memory and tensor memory.
* `cutlass.cute.compile` now supports compilation options. Refer to [JIT compilation options](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html) for more details.
* `cutlass.cute.testing.assert_` now works for device JIT function. Specify `--enable-assertions` as compilation option to enable.
* `cutlass.cute.make_tiled_copy` is now deprecated. Please use `cutlass.cute.make_tiled_copy_tv` instead.
* Shared memory capacity query

  + Introduce `cutlass.utils.get_smem_capacity_in_bytes` for querying the shared memory capacity.
  + `<arch>_utils.SMEM_CAPACITY["<arch_str>"]` is now deprecated.

## [4.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.0.0) (2025-06-03)[#](#id9 "Link to this heading")

* Fixed API mismatch in class `cute.runtime.Pointer`: change `element_type` to `dtype` to match `typing.Pointer`

[previous

CuTe DSL API](../cute_dsl_api.html "previous page")
[next

cutlass.cute](cute.html "next page")

On this page

* [4.3.0 (2025-10-20)](#id2)
* [4.2.0 (2025-09-10)](#id4)
* [4.1.0 (2025-07-16)](#id6)
* [4.0.0 (2025-06-03)](#id9)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.