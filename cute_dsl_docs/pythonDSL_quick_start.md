

Quick Start Guide — NVIDIA CUTLASS Documentation



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
* [Functionality](functionality.html)
* [Quick Start Guide](#)
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

* Quick Start Guide

# Quick Start Guide[#](#quick-start-guide "Link to this heading")

The CUTLASS DSL 4.4 release currently supports **Linux** and **Python 3.10 - 3.14** only. To install CUTLASS DSLs (limited to CuTe DSL for now), use the following command

## Installation[#](#installation "Link to this heading")

Before installing the latest version, you need to uninstall any previous CUTLASS DSL Installation.

```python
pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y
```

To ensure compatibility with the examples and code on [GitHub](https://github.com/NVIDIA/cutlass/tree/main),
use the [setup.sh](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/setup.sh) file from the corresponding commit in the repository.

```python
git clone https://github.com/NVIDIA/cutlass.git

# For CUDA Toolkit 12.9:
./cutlass/python/CuTeDSL/setup.sh --cu12

# For CUDA Toolkit 13.1:
./cutlass/python/CuTeDSL/setup.sh --cu13
```

If you just want to try out the last known stable release of the CUTLASS DSL (may not be compatible with the latest examples and code), run:

```python
# For CUDA Toolkit 12.9:
pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.1:
pip install nvidia-cutlass-dsl[cu13]
```

The `nvidia-cutlass-dsl` wheel includes everything needed to generate GPU kernels. It requires
the same NVIDIA driver version as the corresponding [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(CUDA Toolkit 12.9 or CUDA Toolkit 13.1).

## Recommended Dependencies[#](#recommended-dependencies "Link to this heading")

To run examples and begin development, we recommend installing:

```python
pip install torch jupyter
```

We recommend installing JAX with CUDA support at version 0.8.1 to run JAX examples.

## Recommended Python environment variables for jupyter notebooks[#](#recommended-python-environment-variables-for-jupyter-notebooks "Link to this heading")

We recommend setting the following environment variable when running jupyter notebooks.

```python
export PYTHONUNBUFFERED=1
```

[previous

Functionality](functionality.html "previous page")
[next

CuTe DSL](cute_dsl.html "next page")

On this page

* [Installation](#installation)
* [Recommended Dependencies](#recommended-dependencies)
* [Recommended Python environment variables for jupyter notebooks](#recommended-python-environment-variables-for-jupyter-notebooks)





[![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.