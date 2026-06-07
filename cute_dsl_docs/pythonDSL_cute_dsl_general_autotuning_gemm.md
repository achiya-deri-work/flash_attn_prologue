

Guidance for Auto-Tuning — NVIDIA CUTLASS Documentation



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
  + [Debugging with the DSL](debugging.html)
  + [Autotuning with the DSL](#)
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
* Guidance for Auto-Tuning

# Guidance for Auto-Tuning[#](#guidance-for-auto-tuning "Link to this heading")

Numerous GEMM kernel code examples are offered within our codebase.
When integrating these kernels into frameworks, auto-tuning becomes essential
for achieving optimal performance. This involves selecting the appropriate
kernel parameters based on the inputs of real applications.
Next, weâll briefly introduce some tips on how to perform auto-tuning.

The auto-tuning process typically involves the following steps:

1. Define search space
2. Benchmark each configuration and select the kernel with the best performance
3. Enable caching to reduce the tuning cost

The search space defines the valid combinations of kernel parameters that can be used to run the kernels.
Different inputs (shapes, data types, etc.) typically require different kernel parameters to achieve optimal performance.
The search space is related to the kernel. We take the Blackwell GEMM persistent kernel as an example.
The search space is as follows:

* `mma_tiler_mn`: Defines the dimensions of the matrix tile that each Matrix Multiply-Accumulate (MMA) instruction processes in a single operation.
* `cluster_shape_mn`: Specifies the number of CTAs along each dimension within a cluster. Refer [Parallel Thread Execution ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions) for the possible mma tiler size and cluster shape for different tensor data types.
* `use_2cta_instrs`: Whether to utilize Blackwellâs 2 CTA instructions for MMA/Copy.
* `use_tma_store`: Whether to use Tensor Memory Access (TMA) instructions to store the result back to global memory.

After defining the search space, we could traverse all parameter combinations to find the optimal kernel.
The `autotune_gemm` function below demonstrates a simple exhaustive search approach - it iterates
through configurations, compiles and benchmarks each kernel, and returns the best performing one.
Since kernel compilation incurs overhead, itâs important to cache and reuse compiled kernels
to minimize host launch latency. CuTe DSL facilitates this through its separate compilation
and execution workflow. More details can be found in [JIT Caching](dsl_jit_caching.html#jit-caching).
As demonstrated in the `autotune_gemm` function
(between the `begin of cache the compiled GEMM kernel` and `end of cache the compiled GEMM kernel` comments),
we can use `cute.compile()` to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel
executions. We could maintain a global configuration-to-kernel dictionary (`config_kernel_dict`) to cache the compiled GEMM kernels,
where each key (`kernel_cache_key`) uniquely identifies a kernel based on its characteristics.
Usually we could use the {dtype + kernel configs} as the cached key for GEMM compilation. For example,

```python
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"
```

If the input tensorâs layout is static, we should add the shape in the cached key too.
Users can customize the `benchmark` function to measure kernel execution time.
For stable and reliable performance measurements:

1. Run a few warmup iterations (e.g., 5-10) to stabilize GPU temperature
2. Execute multiple timed iterations (e.g., 100-1000) for statistical significance
3. Use CUDA events and synchronization for precise timing
4. Lock GPU frequencies (SM and memory frequencies) with nvidia-smi
5. Process results by removing outliers and using min/avg statistics as measurements.

This ensures reliable kernel selection through proper benchmarking.

```python
# get the best GEMM kernel for given input tensors
def autotune_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    use_2cta_instrs_list: List[bool] = [True],
    use_tma_store_list: List[bool] = [True],
    mma_tiler_m_list: List[int] = [256],
    mma_tiler_n_list: List[int] = [256],
    cluster_shape_m_list: List[int] = [2],
    cluster_shape_n_list: List[int] = [1],
):
    best_kernel = None
    min_time = float("inf")
    # traverse the search space
    for use_2cta_instrs in use_2cta_instrs_list:
        for use_tma_store in use_tma_store_list:
            for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                    acc_dtype = cutlass.Float32
                    hardware_info = cutlass.utils.HardwareInfo()
                    max_active_clusters = hardware_info.get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    )
                    # instance a GEMM kernel
                    gemm = PersistentDenseGemmKernel(
                        acc_dtype,
                        use_2cta_instrs,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_tma_store,
                    )
                    # begin of cache the compiled GEMM kernel
                    if kernel_cache_key not in config_kernel_dict:
                        # compile gemm kernel
                        compiled_gemm = cute.compile(
                            gemm,
                            a,
                            b,
                            c,
                            max_active_clusters,
                            stream,
                        )
                        config_kernel_dict[kernel_cache_key] = compiled_gemm
                    else:
                        compiled_gemm = config_kernel_dict[kernel_cache_key]
                    # end of cache the compiled GEMM kernel
                    try:
                        # define a benchmark function to measure the execution time of the compiled GEMM kernel
                        cur_time = benchmark(
                            partial(compiled_gemm, a, b, c, stream),
                        )
                    except Exception as e:
                        print(f"Execution error: {e}")
                        cur_time = float("inf")
                    if cur_time < min_time:
                        min_time = cur_time
                        best_kernel = compiled_gemm
    if best_kernel is None:
        raise ValueError("No best kernel found")
    return best_kernel
```

This brute-force approach ensures we could find the optimal parameters, though at the cost of trying every possibilities.
For more advanced use cases, users can explore sophisticated optimization
techniques like search space pruning and genetic algorithms to reduce tuning overhead and discover better
configurations more efficiently.

To further optimize tuning performance, we can utilize caching mechanisms to avoid redundant computations.
We could cache the tuning results in a input-to-kernel dictionary (e.g., `input_kernel_dict`).
When processing inputs with matching `config_key` values, the cached kernel can be reused directly without re-tuning.
The `config_key` is related with the input tensorâs characteristics, such as the shape, data type, etc.
The setup of `config_key` is very flexible, users can customize it based on their own application.
For instance, if the data type is fixed in usersâ application, we could use the input tensorâs shape as the key, i.e., `(m, n, k)`.
To further reduce tuning overhead, we could consider using a simplified key like `config_key = (power_of_2(m), power_of_2(n), power_of_2(k))`,
where `m`, `n`, and `k` are rounded up to the nearest power of 2. This simplification can significantly reduce the number
of unique keys while still maintaining good performance in most cases. However, itâs important to validate that this
approximation doesnât negatively impact performance for your specific use case.

```python
config_key = (m, n, k)
if config_key in input_kernel_dict:
    compiled_gemm = input_kernel_dict[config_key]
else:
    compiled_gemm = autotune_gemm(...)
    input_kernel_dict[config_key] = compiled_gemm
# launch gemm kernel
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

By following the methods above, you can customize your own auto-tuner to find the optimal GEMM kernel configuration
for specific matrix dimensions and data types, significantly improving computational performance for models.

[previous

Debugging](debugging.html "previous page")
[next

Educational Notebooks](notebooks.html "next page")





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.