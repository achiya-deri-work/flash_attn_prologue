

Functionality — NVIDIA CUTLASS Documentation



[Skip to main content](#main-content)

Back to top




`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Search
`Ctrl`+`K`

Search
`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Table of Contents

* [Changelog](../../../CHANGELOG.html)

CuTe DSL

* [Overview](overview.html)
* [Functionality](#)
* [Quick Start Guide](quick_start.html)
* [CuTe DSL](cute_dsl.html)
  + [Introduction](cute_dsl_general/dsl_introduction.html)
  + [Code Generation](cute_dsl_general/dsl_code_generation.html)
  + [Control Flow](cute_dsl_general/dsl_control_flow.html)
  + [JIT Argument Generation](cute_dsl_general/dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](cute_dsl_general/dsl_dynamic_layout.html)
  + [JIT Caching](cute_dsl_general/dsl_jit_caching.html)
  + [JIT Compilation Options](cute_dsl_general/dsl_jit_compilation_options.html)
  + [JIT Types](cute_dsl_general/types.html)
  + [Integration with Frameworks](cute_dsl_general/framework_integration.html)
  + [Debugging with the DSL](cute_dsl_general/debugging.html)
  + [Autotuning with the DSL](cute_dsl_general/autotuning_gemm.html)
  + [Educational Notebooks](cute_dsl_general/notebooks.html)
  + [Deprecation Policy](deprecation.html)
  + [Compile with TVM FFI](cute_dsl_general/compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](cute_dsl_general/dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](cute_dsl_general/resources.html)
* [CuTe DSL API](cute_dsl_api.html)
  + [changelog](cute_dsl_api/changelog.html)
  + [cute](cute_dsl_api/cute.html)
    - [arch](cute_dsl_api/cute_arch.html)
    - [Runtime](cute_dsl_api/cute_runtime.html)
  + [cute\_nvgpu](cute_dsl_api/cute_nvgpu.html)
    - [Common](cute_dsl_api/cute_nvgpu_common.html)
    - [warp submodule](cute_dsl_api/cute_nvgpu_warp.html)
    - [warpgroup submodule](cute_dsl_api/cute_nvgpu_warpgroup.html)
    - [cpasync submodule](cute_dsl_api/cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](cute_dsl_api/cute_nvgpu_tcgen05.html)
  + [pipeline](cute_dsl_api/pipeline.html)
  + [utils](cute_dsl_api/utils.html)
    - [Utilities for SM90](cute_dsl_api/utils_sm90.html)
    - [Utilities for SM100](cute_dsl_api/utils_sm100.html)
* [Limitations](limitations.html)
* [FAQs](faqs.html)

CUTLASS C++

* [Overview](../../../overview.html)
* [Getting Started](../cpp/getting_started.html)
  + [Quickstart](../cpp/quickstart.html)
  + [IDE Setup](../cpp/ide_setup.html)
  + [Build](../cpp/build/index.html)
    - [Building on Windows with Visual Studio](../cpp/build/building_in_windows_with_visual_studio.html)
    - [Building with Clang as host compiler](../cpp/build/building_with_clang_as_host_compiler.html)
  + [Functionality](../cpp/functionality.html)
  + [Terminology](../cpp/terminology.html)
  + [Fundamental Types](../cpp/fundamental_types.html)
  + [Programming Guidelines](../cpp/programming_guidelines.html)
  + [GEMM Heuristics](../cpp/heuristics.html)
* [Efficient GEMM in CUDA](../cpp/efficient_gemm.html)
* [Synchronization primitives](../cpp/pipeline.html)
* [CUTLASS Profiler](../cpp/profiler.html)
* [Dependent Kernel Launch](../cpp/dependent_kernel_launch.html)
* [Blackwell Specific](../cpp/blackwell.html)
  + [Blackwell SM100 GEMMs](../cpp/blackwell_functionality.html)
  + [Blackwell Cluster Launch Control](../cpp/blackwell_cluster_launch_control.html)
* [CuTe](../cpp/cute/index.html)
  + [00\_quickstart](../cpp/cute/00_quickstart.html)
  + [01\_layout](../cpp/cute/01_layout.html)
  + [02\_layout\_algebra](../cpp/cute/02_layout_algebra.html)
  + [03\_tensor](../cpp/cute/03_tensor.html)
  + [04\_algorithms](../cpp/cute/04_algorithms.html)
  + [0t\_mma\_atom](../cpp/cute/0t_mma_atom.html)
  + [0x\_gemm\_tutorial](../cpp/cute/0x_gemm_tutorial.html)
  + [0y\_predication](../cpp/cute/0y_predication.html)
  + [0z\_tma\_tensors](../cpp/cute/0z_tma_tensors.html)
* [CUTLASS 3.x](../cpp/cutlass_3x.html)
  + [Design](../cpp/cutlass_3x_design.html)
  + [GEMM Backwards Compatibility](../cpp/cutlass_3x_backwards_compatibility.html)
  + [GEMM API](../cpp/gemm_api_3x.html)
* [CUTLASS 2.x](../cpp/cutlass_2x.html)
  + [Layouts and Tensors](../cpp/layout.html)
  + [GEMM API](../cpp/gemm_api.html)
  + [Tile Iterator Concepts](../cpp/tile_iterator_concept.html)
  + [Utilities](../cpp/utilities.html)
* [Code Organization](../cpp/code_organization.html)
* [Grouped Kernel Schedulers](../cpp/grouped_scheduler.html)
* [CUTLASS Convolution](../cpp/implicit_gemm_convolution.html)

Reference

* [Software License Agreement](license.html)

* Functionality

# Functionality[#](#functionality "Link to this heading")

The CUTLASS DSL 4.0 release supports **Python 3.12** only. It shares the same driver requirements
as the [CUDA Toolkit 12.9](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
Specifically, the driver version must be 575.51.03 or later.

Currently, only Linux x86\_64 is supported. Additional platform support will be added in future releases.

## Supported MMA Operations[#](#supported-mma-operations "Link to this heading")

**NVIDIA Ampere Architecture:**

* FP16 / BF16 tensor core instructions

**NVIDIA Hopper Architecture:**

* FP16 / BF16
* FP8

**NVIDIA Blackwell Architecture:**

* FP16 / BF16
* TF32
* I8
* F8

## Notable Limitations[#](#notable-limitations "Link to this heading")

For current constraints and unsupported features, refer to the [Limitations](limitations.html) section.

[previous

Overview](overview.html "previous page")
[next

Quick Start Guide](quick_start.html "next page")

On this page

* [Supported MMA Operations](#supported-mma-operations)
* [Notable Limitations](#notable-limitations)





[![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.