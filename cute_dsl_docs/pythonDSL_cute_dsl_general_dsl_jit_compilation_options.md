

JIT Compilation Options — NVIDIA CUTLASS Documentation



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
  + [Introduction](dsl_introduction.html)
  + [Code Generation](dsl_code_generation.html)
  + [Control Flow](dsl_control_flow.html)
  + [JIT Argument Generation](dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](dsl_dynamic_layout.html)
  + [JIT Caching](dsl_jit_caching.html)
  + [JIT Compilation Options](#)
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
* JIT Compilation Options

# JIT Compilation Options[#](#jit-compilation-options "Link to this heading")

## JIT Compilation Options Overview[#](#jit-compilation-options-overview "Link to this heading")

When compiling a JIT function using CuTe DSL, you may want to control various aspects of the compilation process, such as optimization level, or debugging flags. CuTe DSL provides a flexible interface for specifying these compilation options when invoking `cute.compile`.

Compilation options allow you to customize how your JIT-compiled functions are built and executed. This can be useful for:

* Enabling or disabling specific compiler optimizations
* Generating debug information for troubleshooting

These options can be passed as keyword arguments to `cute.compile` or set globally for all JIT compilations. The available options and their effects are described in the following sections, along with usage examples to help you get started.

The CuTe DSL provides multiple ways to specify compilation options - either by specifying additional arguments to `cute.compile` or by using a more Pythonic approach with separate Python types for `cute.compile`.

## `cute.compile` Compilation Options as strings[#](#cute-compile-compilation-options-as-strings "Link to this heading")

You can provide additional compilation options as a string when calling `cute.compile`. The CuTe DSL uses `argparse` to parse these options and will raise an error if any invalid options are specified.

| **Option** | **Description** | **Default** | **Type** |
| --- | --- | --- | --- |
| `opt-level` | Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is [0, 3]. | 3 (highest level of optimization) | int |
| `enable-assertions` | Enable host and device code assertions. | False | bool |
| `keep-cubin` | Keep the generated CUBIN file. | False | bool |
| `keep-ptx` | Keep the generated PTX file. | False | bool |
| `ptxas-options` | The options to pass to the PTX Compiler library. | ââ | str |
| `generate-line-info` | Generate line information for debugging. | False | bool |
| `gpu-arch` | The GPU architecture to compile for. | ââ | str |
| `enable-tvm-ffi` | Enable Apache TVM FFI. | False | bool |

You can use the following code to specify compilation options:

```python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```

## `cute.compile` Compilation Options as separate Python types[#](#cute-compile-compilation-options-as-separate-python-types "Link to this heading")

Alternatively, you can also use a more Pythonic way to specify compilation options with separate Python types.
Compilation options can be programmatically composed using tuple and passed to `cute.compile` separately.

```python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

This approach causes invalid options to raise errors immediately, making it much easier to detect typos when specifying multiple options.
Notebly, boolean options are automatically converted to True instances of the option type for convenience.

```python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```

[previous

JIT Caching](dsl_jit_caching.html "previous page")
[next

Types](types.html "next page")

On this page

* [JIT Compilation Options Overview](#jit-compilation-options-overview)
* [`cute.compile` Compilation Options as strings](#cute-compile-compilation-options-as-strings)
* [`cute.compile` Compilation Options as separate Python types](#cute-compile-compilation-options-as-separate-python-types)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.