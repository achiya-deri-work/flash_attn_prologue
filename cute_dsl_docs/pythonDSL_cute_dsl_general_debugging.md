

Debugging — NVIDIA CUTLASS Documentation



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
  + [JIT Compilation Options](dsl_jit_compilation_options.html)
  + [JIT Types](types.html)
  + [Integration with Frameworks](framework_integration.html)
  + [Debugging with the DSL](#)
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
* Debugging

# Debugging[#](#debugging "Link to this heading")

This page provides an overview of debugging techniques and tools for CuTe DSL programs.

## Getting Familiar with the Limitations[#](#getting-familiar-with-the-limitations "Link to this heading")

Before diving into comprehensive debugging capabilities, itâs important to understand the limitations of CuTe DSL.
Understanding these limitations will help you avoid potential pitfalls from the start.

Please refer to [Limitations](../limitations.html) for more details.

## Source Code Correlation[#](#source-code-correlation "Link to this heading")

CuTe DSL provides Python code to PTX/SASS correlation to enable the profiling/debugging of generated kernels with debug symbols by generating line info when compiling the kernel.

You can enable that globally via the environment variable CUTE\_DSL\_LINEINFO=1. Alternative, you can use compilation options to enable that per kernel. Please refer to [JIT Compilation Options](dsl_jit_compilation_options.html) for more details.

## DSL Debugging[#](#dsl-debugging "Link to this heading")

CuTe DSL provides built-in logging mechanisms to help you understand the code execution flow and
some of the internal state.

### Enabling Logging[#](#enabling-logging "Link to this heading")

CuTe DSL provides environment variables to control logging level:

```python
# Enable console logging (default: False)
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False)
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
export CUTE_DSL_LOG_LEVEL=20
```

### Log Categories and Levels[#](#log-categories-and-levels "Link to this heading")

Similar to standard Python logging, different log levels provide varying degrees of detail:

| Level | Description |
| --- | --- |
| 0 | Disabled |
| 10 | Debug |
| 20 | Info |
| 30 | Warning |
| 40 | Error |
| 50 | Critical |

### Dump the generated IR[#](#dump-the-generated-ir "Link to this heading")

For users familiar with MLIR and compilers, CuTe DSL supports dumping the Intermediate Representation (IR).
This helps you verify whether the IR is generated as expected.

```python
# Dump Generated CuTe IR (default: False)
export CUTE_DSL_PRINT_IR=1

# Keep Generated CuTe IR in a file (default: False)
export CUTE_DSL_KEEP_IR=1
```

### Dump the generated PTX & CUBIN[#](#dump-the-generated-ptx-cubin "Link to this heading")

For users familiar with PTX and SASS, CuTe DSL supports dumping the generated PTX and CUBIN.

```python
# Dump generated PTX in a .ptx file (default: False)
export CUTE_DSL_KEEP_PTX=1

# Dump generated cubin in a .cubin file (default: False)
export CUTE_DSL_KEEP_CUBIN=1
```

To further get SASS from cubin, users can use `nvdisasm` (usually installed with CUDA toolkit) to disassemble the cubin.

```python
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```

### Access the dumped contents programmatically[#](#access-the-dumped-contents-programmatically "Link to this heading")

For compiled kernels, the generated PTX/CUBIN/IR can be accessed programmatically as well through following attributes:

* `__ptx__`: The generated PTX code of the compiled kernel.
* `__cubin__`: The generated CUBIN data of the compiled kernel.
* `__mlir__`: The generated IR code of the compiled kernel.

```python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

### Change the dump directory[#](#change-the-dump-directory "Link to this heading")

By default, all dumped files are saved in the current working directory. To specify a different directory for the dumped files, please set the environment variable CUTE\_DSL\_DUMP\_DIR accordingly.

## Kernel Functional Debugging[#](#kernel-functional-debugging "Link to this heading")

### Using Pythonâs `print` and CuTeâs `cute.printf`[#](#using-python-s-print-and-cute-s-cute-printf "Link to this heading")

CuTe DSL programs can use both Pythonâs native `print()` as well as our own `cute.printf()` to
print debug information during kernel generation and execution. They differ in a few key ways:

* Pythonâs `print()` executes during compile-time only (no effect on the generated kernel) and is
  typically used for printing static values (e.g. a fully static layouts).
* `cute.printf()` executes at runtime on the GPU itself and changes the PTX being generated. This
  can be used for printing values of tensors at runtime for diagnostics, but comes at a performance
  overhead similar to that of printf() in CUDA C.

For detailed examples of using these functions for debugging, please refer to the associated
notebook referenced in [Educational Notebooks](notebooks.html).

### Handling Unresponsive/Hung Kernels[#](#handling-unresponsive-hung-kernels "Link to this heading")

When a kernel becomes unresponsive and `SIGINT` (`CTRL+C`) fails to terminate it,
you can follow these steps to forcefully terminate the process:

1. Use `CTRL+Z` to suspend the unresponsive kernel
2. Execute the following command to terminate the suspended process:

```python
# Terminate the most recently suspended process
kill -9 $(jobs -p | tail -1)
```

CuTe DSL can also be debugged using standard NVIDIA CUDA tools.

### Using Compute-Sanitizer[#](#using-compute-sanitizer "Link to this heading")

For detecting memory errors and race conditions:

```python
compute-sanitizer --some_options python your_dsl_code.py
```

Please refer to the [compute-sanitizer documentation](https://developer.nvidia.com/compute-sanitizer) for more details.

### Set function name prefix[#](#set-function-name-prefix "Link to this heading")

By default, the function name (host function or kernel function) is automatically generated based on the function name and its parameters.
Sometimes you may want to attach some runtime information to the function name to make performance profiling and debugging easier,
e.g., the kernel configs or the rank ids. You can assign a name prefix to the name by calling the `set_name_prefix`
method on the host function or kernel function.

```python
@cute.kernel
def kernel(arg1, arg2, ...):
    ...
@cute.jit
def launch_kernel():
    kernel.set_name_prefix("your_custom_name_prefix")
    kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
```

For above example, the generated kernel name will be âyour\_custom\_name\_prefix\_xxxâ.

## Conclusion[#](#conclusion "Link to this heading")

This page covered several key methods for debugging CuTe DSL programs. Effective debugging typically requires a combination of these approaches.
If you encounter issues with DSL, you can enable logging and share the logs with the CUTLASS team as a GitHub issue to report a bug.

[previous

Integration with Frameworks](framework_integration.html "previous page")
[next

Guidance for Auto-Tuning](autotuning_gemm.html "next page")

On this page

* [Getting Familiar with the Limitations](#getting-familiar-with-the-limitations)
* [Source Code Correlation](#source-code-correlation)
* [DSL Debugging](#dsl-debugging)
  + [Enabling Logging](#enabling-logging)
  + [Log Categories and Levels](#log-categories-and-levels)
  + [Dump the generated IR](#dump-the-generated-ir)
  + [Dump the generated PTX & CUBIN](#dump-the-generated-ptx-cubin)
  + [Access the dumped contents programmatically](#access-the-dumped-contents-programmatically)
  + [Change the dump directory](#change-the-dump-directory)
* [Kernel Functional Debugging](#kernel-functional-debugging)
  + [Using Pythonâs `print` and CuTeâs `cute.printf`](#using-python-s-print-and-cute-s-cute-printf)
  + [Handling Unresponsive/Hung Kernels](#handling-unresponsive-hung-kernels)
  + [Using Compute-Sanitizer](#using-compute-sanitizer)
  + [Set function name prefix](#set-function-name-prefix)
* [Conclusion](#conclusion)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.