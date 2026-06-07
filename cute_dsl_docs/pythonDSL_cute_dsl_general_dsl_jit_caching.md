

JIT Caching — NVIDIA CUTLASS Documentation



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
  + [JIT Caching](#)
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
* JIT Caching

# JIT Caching[#](#jit-caching "Link to this heading")

## Zero Compile and JIT Executor[#](#zero-compile-and-jit-executor "Link to this heading")

Zero Compile is a feature that enables explicit kernel compilation on demand through `cute.compile`.
When `cute.compile` is called, it compiles the kernel and returns a JIT Executor instance.
This JIT Executor instance can be cached and reused directly for subsequent executions without compiling the kernel again.

The JIT Executor is a component that independently executes compiled code.
It can be created either through `cute.compile` or implicit compilation.
The JIT Executor instance behaves like a callable object to execute the compiled code.
Each JIT Executor instance maintains a single compiled host function.

It encompasses all necessary execution components:

* Host function pointer and its MLIR execution engine
* CUDA modules (optional)
* Argument specifications defining how Python arguments are converted to C ABI-compatible types. Note that arguments with the `cutlass.Constexpr` hint are excluded from argument specifications since they are evaluated at compile time rather than runtime.

For example, in the following code, `print_result` is a `cutlass.Constexpr` value that is **NOT** evaluated at runtime:

```python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # output: ``Result: 3``
```

The JIT Executor ensures all components are properly initialized and loaded after compilation.

For example, all CUDA modules are loaded (via `cuModuleLoad`) and kernel function pointers are extracted (via `cuModuleGetFunction`).

When calling a JIT Executor instance, it:

* Parses Python runtime arguments and converts them to C ABI-compatible types according to argument specifications
* Invokes the host function with the converted arguments

### Custom Caching with `cute.compile`[#](#custom-caching-with-cute-compile "Link to this heading")

`cute.compile` bypasses caching in CuTe DSL and always performs compilation, returning a fixed JIT Executor instance.
This allows implementing custom caching strategies as shown below:

```python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```

## Cache in CuTe DSL[#](#cache-in-dsl "Link to this heading")

By default, cache in CuTe DSL is implicitly enabled to avoid recompilation when kernels are called repeatedly without changes.

The cache is implemented as a map storing compiled JIT Executor instances within CuTe DSL.

The cache key combines hashes of:

* MLIR bytecode of the MLIR program generated by CuTe DSL
* All CuTe DSL Python source files
* All CuTe DSL shared libraries
* All CuTe DSL environment variables

The cache value is a compiled JIT Executor instance.

On a cache hit, compilation is skipped and the cached JIT Executor instance is reused.

On a cache miss, the kernel is compiled and the new JIT Executor instance is stored in the cache.

Here is an example demonstrating automatic caching of the `add` kernel:

```python
# Global variable
a = 1

@cute.jit
def add(b):
   return a + b

# Cache is empty at beginning

# First call: cache miss triggers compilation
result = add(2) # result = 3
# Cache now has one instance

# Second call: cache hit reuses cached JIT Executor
result = add(2) # result = 3

a = 2
# Third call: cache miss due to changed IR code triggers recompilation
result = add(2) # result = 4
# Cache now has two instances
```

The cache can be serialized to files for subsequent runs.
After serialization, compiled MLIR bytecode is stored in file.
The cache directory is `/tmp/{current_user}/cutlass_python_cache`.
During compilation, the cache loads the corresponding kernel from file (if it exists) into memory as needed, and after compilation, it saves any newly compiled executables back to file.

Note that for efficiency, the default cache directory is located in a temporary folder. However, this location is not persistent, it may be cleared by the system (for example, during a reboot or disk space cleanup).
If you wish to preserve the cache across sessions, set the `CUTE_DSL_CACHE_DIR` environment variable to point to a persistent directory.

The following environment variables control file caching:

```python
# Disable file caching while keeping in-memory cache available, defaults to False.
export CUTE_DSL_DISABLE_FILE_CACHING=True

# Cache directory, defaults to /tmp/{current_user}/cutlass_python_cache.
export CUTE_DSL_CACHE_DIR=/home/user/local_cutlass_python_cache/dense_gemm_cache/
```

### Limitations[#](#limitations "Link to this heading")

The intention of caching is to reduce the host launch overhead before each execution. As above example shows,
the consistency between the original Python code and the MLIR program is hard to maintain because of the impact of dynamic factors such as global variables.
Therefore, the MLIR program **MUST** always be generated to verify that the kernel content matches what was previously built.

For optimal host launch latency, we recommend using above custom caching method with `cute.compile`.

[previous

Static vs Dynamic layouts](dsl_dynamic_layout.html "previous page")
[next

JIT Compilation Options](dsl_jit_compilation_options.html "next page")

On this page

* [Zero Compile and JIT Executor](#zero-compile-and-jit-executor)
  + [Custom Caching with `cute.compile`](#custom-caching-with-cute-compile)
* [Cache in CuTe DSL](#cache-in-dsl)
  + [Limitations](#limitations)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.