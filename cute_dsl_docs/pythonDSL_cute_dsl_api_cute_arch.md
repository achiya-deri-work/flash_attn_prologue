

arch — NVIDIA CUTLASS Documentation



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
  + [changelog](changelog.html)
  + [cute](cute.html)
    - [arch](#)
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
* [cutlass.cute](cute.html)
* arch

# arch[#](#arch "Link to this heading")

The `cute.arch` module provides lightweight wrappers for NVVM Operation builders which implement CUDA built-in
device functions such as `thread_idx`. It integrates seamlessly with CuTe DSL types.

These wrappers enable source location tracking through the `@dsl_user_op`
decorator. The module includes the following functionality:

* Core CUDA built-in functions such as `thread_idx`, `warp_idx`, `block_dim`, `grid_dim`, `cluster_dim`, and related functions
* Memory barrier management functions including `mbarrier_init`, `mbarrier_arrive`, `mbarrier_wait`, and associated operations
* Low-level shared memory (SMEM) management capabilities, with `SmemAllocator` as the recommended interface
* Low-level tensor memory (TMEM) management capabilities, with `TmemAllocator` as the recommended interface

## API documentation[#](#module-cutlass.cute.arch "Link to this heading")

cutlass.cute.arch.make\_warp\_uniform( : *value: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.make_warp_uniform "Link to this definition")
:   Provides a compiler hint indicating that the specified value is invariant across all threads in the warp,
    which may enable performance optimizations.

    Parameters:
    :   **value** (*Int*) â The integer value to be marked as warp-uniform.

    Returns:
    :   The input value, marked as warp-uniform.

    Return type:
    :   Int32

cutlass.cute.arch.elect\_one(*\**, *loc=None*, *ip=None*) → IfOpRegion[#](#cutlass.cute.arch.elect_one "Link to this definition")
:   Elects one thread within a warp.

    ```python
    with elect_one():
        # Only one thread in the warp executes the code in this context
        pass
    ```

cutlass.cute.arch.mbarrier\_init( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *cnt: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.mbarrier_init "Link to this definition")
:   Initializes a mbarrier with the specified thread arrival count.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **cnt** (*Int*) â The arrival count of the mbarrier

cutlass.cute.arch.mbarrier\_init\_fence(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.mbarrier_init_fence "Link to this definition")
:   A fence operation that applies to the mbarrier initializations.

cutlass.cute.arch.mbarrier\_arrive\_and\_expect\_tx( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *bytes: cutlass.cute.typing.Int*, : *peer\_cta\_rank\_in\_cluster=None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.mbarrier_arrive_and_expect_tx "Link to this definition")
:   Arrives on a mbarrier and expects a specified number of transaction bytes.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **bytes** (*Int*) â The number of transaction bytes
        * **peer\_cta\_rank\_in\_cluster** â An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTAâs
          SMEM.

cutlass.cute.arch.mbarrier\_expect\_tx( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *bytes: cutlass.cute.typing.Int*, : *peer\_cta\_rank\_in\_cluster=None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.mbarrier_expect_tx "Link to this definition")
:   Expects a specified number of transaction bytes without an arrive.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **bytes** (*Int*) â The number of transaction bytes
        * **peer\_cta\_rank\_in\_cluster** â An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTAâs
          SMEM.

cutlass.cute.arch.mbarrier\_wait( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.mbarrier_wait "Link to this definition")
:   Waits on a mbarrier with a specified phase.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **phase** (*Int*) â The phase to wait for (either 0 or 1)

cutlass.cute.arch.mbarrier\_try\_wait( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.arch.mbarrier_try_wait "Link to this definition")
:   Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **phase** (*Int*) â The phase to wait for (either 0 or 1)

    Returns:
    :   A boolean value indicating whether the wait operation was successful

    Return type:
    :   Boolean

cutlass.cute.arch.mbarrier\_conditional\_try\_wait( : *cond*, : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.arch.mbarrier_conditional_try_wait "Link to this definition")
:   Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    Parameters:
    :   * **cond** â A boolean predicate
        * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **phase** (*Int*) â The phase to wait for (either 0 or 1)

    Returns:
    :   A boolean value indicating whether the wait operation was successful

    Return type:
    :   Boolean

cutlass.cute.arch.mbarrier\_arrive( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *peer\_cta\_rank\_in\_cluster: cutlass.cute.typing.Int | None = None*, : *arrive\_count: cutlass.cute.typing.Int = 1*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.mbarrier_arrive "Link to this definition")
:   Arrives on an mbarrier.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **peer\_cta\_rank\_in\_cluster** â An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTAâs
          SMEM.

cutlass.cute.arch.lane\_idx(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.lane_idx "Link to this definition")
:   Returns the lane index of the current thread within the warp.

cutlass.cute.arch.warp\_idx(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.warp_idx "Link to this definition")
:   Returns the warp index within a CTA.

cutlass.cute.arch.thread\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.thread_idx "Link to this definition")
:   Returns the thread index within a CTA.

cutlass.cute.arch.block\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.block_dim "Link to this definition")
:   Returns the number of threads in each dimension of the CTA.

cutlass.cute.arch.block\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.block_idx "Link to this definition")
:   Returns the CTA identifier within a grid.

cutlass.cute.arch.grid\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.grid_dim "Link to this definition")
:   Returns the number of CTAs in each dimension of the grid.

cutlass.cute.arch.cluster\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.cluster_idx "Link to this definition")
:   Returns the cluster identifier within a grid.

cutlass.cute.arch.cluster\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.cluster_dim "Link to this definition")
:   Returns the number of clusters in each dimension of the grid.

cutlass.cute.arch.cluster\_size(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.cluster_size "Link to this definition")
:   Returns the number of CTA within the cluster.

cutlass.cute.arch.block\_in\_cluster\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.block_in_cluster_idx "Link to this definition")
:   Returns the CTA index within a cluster across all dimensions.

cutlass.cute.arch.block\_in\_cluster\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.block_in_cluster_dim "Link to this definition")
:   Returns the dimensions of the cluster.

cutlass.cute.arch.block\_idx\_in\_cluster( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.block_idx_in_cluster "Link to this definition")
:   Returns the linearized identifier of the CTA within the cluster.

cutlass.cute.arch.dynamic\_smem\_size(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.dynamic_smem_size "Link to this definition")
:   Returns the launch dynamic smem size.

cutlass.cute.arch.barrier( : *\**, : *barrier\_id=None*, : *number\_of\_threads=None*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.barrier "Link to this definition")
:   Creates a barrier, optionally named.

cutlass.cute.arch.barrier\_arrive( : *\**, : *barrier\_id=None*, : *number\_of\_threads=None*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.barrier_arrive "Link to this definition")

cutlass.cute.arch.sync\_threads(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.sync_threads "Link to this definition")
:   Synchronizes all threads within a CTA.

cutlass.cute.arch.sync\_warp( : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.sync_warp "Link to this definition")
:   Performs a warp-wide sync with an optional mask.

cutlass.cute.arch.fence\_acq\_rel\_cta(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.fence_acq_rel_cta "Link to this definition")
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_cluster(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.fence_acq_rel_cluster "Link to this definition")
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_gpu(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.fence_acq_rel_gpu "Link to this definition")
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_sys(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.fence_acq_rel_sys "Link to this definition")
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.cp\_async\_commit\_group(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cp_async_commit_group "Link to this definition")
:   Commits all prior initiated but uncommitted cp.async instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group).

cutlass.cute.arch.cp\_async\_wait\_group(*n*, *\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cp_async_wait_group "Link to this definition")
:   Waits till only a specified numbers of cp.async groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).

cutlass.cute.arch.cp\_async\_bulk\_commit\_group(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cp_async_bulk_commit_group "Link to this definition")
:   Commits all prior initiated but uncommitted cp.async.bulk instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group).

cutlass.cute.arch.cp\_async\_bulk\_wait\_group( : *group*, : *\**, : *read=None*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.cp_async_bulk_wait_group "Link to this definition")
:   Waits till only a specified numbers of cp.async.bulk groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

cutlass.cute.arch.cluster\_wait(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cluster_wait "Link to this definition")
:   A cluster-wide wait operation.

cutlass.cute.arch.cluster\_arrive(*\**, *aligned=None*, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cluster_arrive "Link to this definition")
:   A cluster-wide arrive operation.

cutlass.cute.arch.cluster\_arrive\_relaxed(*\**, *aligned=None*, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.cluster_arrive_relaxed "Link to this definition")
:   A cluster-wide arrive operation with relaxed semantics.

cutlass.cute.arch.vote\_ballot\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32[#](#cutlass.cute.arch.vote_ballot_sync "Link to this definition")
:   Performs a ballot operation across the warp.

    It copies the predicate from each thread in mask into the corresponding bit position of
    destination register d, where the bit position corresponds to the threadâs lane id.

    Parameters:
    :   * **pred** (*Boolean*) â The predicate value for the current thread
        * **mask** (*Int**,* *optional*) â A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)

    Returns:
    :   A 32-bit integer where each bit represents a threadâs predicate value

    Return type:
    :   Int32

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_any\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.arch.vote_any_sync "Link to this definition")
:   True if source predicate is True for any non-exited threads in mask. Negate the source
    predicate to compute .none.

    Parameters:
    :   * **pred** (*Boolean*) â The predicate value for the current thread
        * **mask** (*Int**,* *optional*) â A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_all\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.arch.vote_all_sync "Link to this definition")
:   True if source predicate is True for all non-exited threads in mask. Negate the source
    predicate to compute .none.

    Parameters:
    :   * **pred** (*Boolean*) â The predicate value for the current thread
        * **mask** (*Int**,* *optional*) â A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_uni\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.arch.vote_uni_sync "Link to this definition")
:   True f source predicate has the same value in all non-exited threads in mask. Negating
    the source predicate also computes .uni

    Parameters:
    :   * **pred** (*Boolean*) â The predicate value for the current thread
        * **mask** (*Int**,* *optional*) â A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

cutlass.cute.arch.warp\_redux\_sync( : *value: cutlass.cute.typing.Numeric*, : *kind: Literal['fmax', 'fmin', 'max', 'min', 'add', 'xor', 'or', 'and']*, : *mask\_and\_clamp: cutlass.cute.typing.Int = 4294967295*, : *\**, : *abs: bool | None = None*, : *nan: bool | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.warp_redux_sync "Link to this definition")
:   Perform warp-level reduction operation across threads.

    Reduces values from participating threads in a warp according to the specified operation.
    All threads in the mask receive the same result.

    Parameters:
    :   * **value** (*Numeric*) â Input value to reduce
        * **kind** (*Literal**[**"add"**,* *"and"**,* *"max"**,* *"min"**,* *"or"**,* *"xor"**,* *"fmin"**,* *"fmax"**]*) â Reduction operation. Supported operations:
          - Integer types (Int32/Uint32): âaddâ, âandâ, âmaxâ, âminâ, âorâ, âxorâ
          - Float types (Float32): âfmaxâ, âfminâ (or âmaxâ/âminâ which auto-convert to âfmaxâ/âfminâ)
        * **mask\_and\_clamp** (*Int*) â Warp participation mask (default: FULL\_MASK = 0xFFFFFFFF)
        * **abs** (*bool*) â Apply absolute value before reduction (float types only)
        * **nan** (*Optional**[**bool**]*) â Enable NaN propagation for fmax/fmin operations (float types only)

    Returns:
    :   Reduced value (same for all participating threads)

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_max\_float32( : *ptr*, : *value: cutlass.cute.typing.Float32*, : *\**, : *positive\_only: bool = True*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32[#](#cutlass.cute.arch.atomic_max_float32 "Link to this definition")
:   Performs an atomic max operation on a float32 value in global memory.

    This implementation works correctly for non-negative values (>= 0) using direct bitcast.

    Parameters:
    :   * **ptr** â Pointer to the memory location
        * **value** (*Float32*) â The float32 value to compare and potentially store (should be >= 0 for correct results)
        * **positive\_only** (*bool*) â If True (default), assumes input values are non-negative.
          This parameter is provided for API compatibility and future extensions.

    Returns:
    :   The old value at the memory location

    Return type:
    :   Float32

cutlass.cute.arch.atomic\_add( : *ptr*, : *val: cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value[#](#cutlass.cute.arch.atomic_add "Link to this definition")
:   Performs an atomic addition operation.

    Atomically adds val to the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Union**[**Numeric**,* *ir.Value**]*) â Value to add (scalar Numeric or vector ir.Value)
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Union[Numeric, ir.Value]

cutlass.cute.arch.atomic\_and( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_and "Link to this definition")
:   Performs an atomic bitwise AND operation.

    Atomically computes bitwise AND of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value for AND operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_or( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_or "Link to this definition")
:   Performs an atomic bitwise OR operation.

    Atomically computes bitwise OR of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value for OR operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_xor( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_xor "Link to this definition")
:   Performs an atomic bitwise XOR operation.

    Atomically computes bitwise XOR of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value for XOR operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_max( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_max "Link to this definition")
:   Performs an atomic maximum operation.

    Atomically computes maximum of val and the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value for MAX operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_min( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_min "Link to this definition")
:   Performs an atomic minimum operation.

    Atomically computes minimum of val and the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value for MIN operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_exch( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_exch "Link to this definition")
:   Performs an atomic exchange operation.

    Atomically exchanges val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location
        * **val** (*Numeric*) â Value to exchange
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_cas( : *ptr*, : *\**, : *cmp: cutlass.cute.typing.Numeric*, : *val: cutlass.cute.typing.Numeric*, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.atomic_cas "Link to this definition")
:   Performs an atomic compare-and-swap (CAS) operation.

    Atomically compares the value at the memory location with cmp. If they are equal,
    stores val at the memory location and returns the old value.

    Parameters:
    :   * **ptr** â Pointer to memory location. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **cmp** (*Numeric*) â Value to compare against current memory value
        * **val** (*Numeric*) â Value to store if comparison succeeds
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) â Memory semantic (ârelaxedâ, âreleaseâ, âacquireâ, âacq\_relâ)
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) â Memory scope (âgpuâ, âctaâ, âclusterâ, âsysâ)

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.store( : *ptr*, : *val: cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value*, : *\**, : *level1\_eviction\_priority: Literal['evict\_normal', 'evict\_first', 'evict\_last', 'evict\_no\_allocate', 'evict\_unchanged'] | None = None*, : *cop: Literal['wb', 'cg', 'cs', 'wt'] | None = None*, : *ss: Literal['cta', 'cluster'] | None = None*, : *sem: Literal['relaxed', 'release'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.store "Link to this definition")
:   Store a value to a memory location.

    Parameters:
    :   * **ptr** â Pointer to store to. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **val** (*Union**[**Numeric**,* *ir.Value**]*) â Value to store (scalar Numeric or vector ir.Value)
        * **level1\_eviction\_priority** â L1 cache eviction policy string literal:
          âevict\_normalâ : .level1::eviction\_priority = .L1::evict\_normal
          âevict\_firstâ : .level1::eviction\_priority = .L1::evict\_first
          âevict\_lastâ : .level1::eviction\_priority = .L1::evict\_last
          âevict\_no\_allocateâ : .level1::eviction\_priority = .L1::no\_allocate
          âevict\_unchangedâ : .level1::eviction\_priority = .L1::evict\_unchanged
        * **cop** â Store cache modifier string literal:
        * **ss** â Shared memory space string literal:
          âctaâ : .ss = .shared::cta
          âclusterâ : .ss = .shared::cluster
          None : .ss = .global
        * **sem** â Memory semantic string literal:
        * **scope** â Memory scope string literal:

cutlass.cute.arch.load( : *ptr*, : *dtype: type[cutlass.cute.typing.Numeric] | cutlass.\_mlir.ir.VectorType*, : *\**, : *sem: Literal['relaxed', 'acquire'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *level1\_eviction\_priority: Literal['evict\_normal', 'evict\_first', 'evict\_last', 'evict\_no\_allocate', 'evict\_unchanged'] | None = None*, : *cop: Literal['ca', 'cg', 'cs', 'lu', 'cv'] | None = None*, : *ss: Literal['cta', 'cluster'] | None = None*, : *level\_prefetch\_size: Literal['size\_64b', 'size\_128b', 'size\_256b'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value[#](#cutlass.cute.arch.load "Link to this definition")
:   Load a value from a memory location.

    Parameters:
    :   * **ptr** â Pointer to load from. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **dtype** (*Union**[**type**[**Numeric**]**,* *ir.VectorType**]*) â Data type to load. Can be:
          - Scalar: Numeric type class (Int8, Uint8, Int32, Float32, etc.)
          - Vector: ir.VectorType for vectorized load (e.g., ir.VectorType.get([4], Int64.mlir\_type))
        * **sem** â Memory semantic string literal:
        * **scope** â Memory scope string literal:
        * **level1\_eviction\_priority** â L1 cache eviction policy string literal:
          âevict\_normalâ : .level1::eviction\_priority = .L1::evict\_normal
          âevict\_firstâ : .level1::eviction\_priority = .L1::evict\_first
          âevict\_lastâ : .level1::eviction\_priority = .L1::evict\_last
          âevict\_no\_allocateâ : .level1::eviction\_priority = .L1::no\_allocate
          âevict\_unchangedâ : .level1::eviction\_priority = .L1::evict\_unchanged
        * **cop** â Load cache modifier string literal:
        * **ss** â Shared memory space string literal:
          âctaâ : .ss = .shared::cta
          âclusterâ : .ss = .shared::cluster
          None : .ss = .global
        * **level\_prefetch\_size** â L2 cache prefetch size hint string literal:
          âsize\_64bâ : .level::prefetch\_size = .L2::64B
          âsize\_128bâ : .level::prefetch\_size = .L2::128B
          âsize\_256bâ : .level::prefetch\_size = .L2::256B

    Returns:
    :   Loaded value (scalar Numeric or vector ir.Value)

    Return type:
    :   Union[Numeric, ir.Value]

cutlass.cute.arch.popc( : *value: cutlass.cute.typing.Numeric*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric[#](#cutlass.cute.arch.popc "Link to this definition")
:   Performs a population count operation.

cutlass.cute.arch.fence\_proxy( : *kind: Literal['alias', 'async', 'async.global', 'async.shared', 'tensormap', 'generic']*, : *\**, : *space: Literal['cta', 'cluster'] | None = None*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.fence_proxy "Link to this definition")
:   Fence operation to ensure memory consistency between proxies.

    Parameters:
    :   * **kind** (*Literal**[**"alias"**,* *"async"**,* *"async.global"**,* *"async.shared"**,* *"tensormap"**,* *"generic"**]*) â Proxy kind string literal:
          - âaliasâ : Alias proxy
          - âasyncâ : Async proxy
          - âasync.globalâ : Async global proxy
          - âasync.sharedâ : Async shared proxy
          - âtensormapâ : Tensormap proxy
          - âgenericâ : Generic proxy
        * **space** (*Optional**[**Literal**[**"cta"**,* *"cluster"**]**]*) â Shared memory space scope string literal (optional):
          - âctaâ : CTA (Cooperative Thread Array) scope
          - âclusterâ : Cluster scope

cutlass.cute.arch.warpgroup\_reg\_alloc(*reg\_count: int*, *\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.warpgroup_reg_alloc "Link to this definition")

cutlass.cute.arch.warpgroup\_reg\_dealloc(*reg\_count: int*, *\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.arch.warpgroup_reg_dealloc "Link to this definition")

cutlass.cute.arch.setmaxregister\_increase(*reg\_count: int*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.setmaxregister_increase "Link to this definition")

cutlass.cute.arch.setmaxregister\_decrease(*reg\_count: int*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.setmaxregister_decrease "Link to this definition")

cutlass.cute.arch.fmax( : *a: float | cutlass.cute.typing.Float32*, : *b: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32[#](#cutlass.cute.arch.fmax "Link to this definition")

cutlass.cute.arch.rcp\_approx( : *a: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.arch.rcp_approx "Link to this definition")

cutlass.cute.arch.exp2( : *a: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32[#](#cutlass.cute.arch.exp2 "Link to this definition")

cutlass.cute.arch.cvt\_i8x4\_to\_f32x4(*src\_vec4*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8x4_to_f32x4 "Link to this definition")

cutlass.cute.arch.cvt\_i8x2\_to\_f32x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8x2_to_f32x2 "Link to this definition")

cutlass.cute.arch.cvt\_i8\_bf16(*src\_i8*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8_bf16 "Link to this definition")

cutlass.cute.arch.cvt\_i8x2\_to\_bf16x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8x2_to_bf16x2 "Link to this definition")

cutlass.cute.arch.cvt\_i8x4\_to\_bf16x4(*src\_vec4*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8x4_to_bf16x4 "Link to this definition")

cutlass.cute.arch.cvt\_f32x2\_bf16x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_f32x2_bf16x2 "Link to this definition")

cutlass.cute.arch.alloc\_smem( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *size\_in\_elems: int*, : *alignment: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.arch.alloc_smem "Link to this definition")
:   Statically allocates SMEM.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) â The pointee type of the pointer.
        * **size\_in\_elems** (*int*) â The size of the allocation in terms of number of elements of the
          pointee type
        * **alignment** (*int*) â An optional pointer alignment for the allocation

    Returns:
    :   A pointer to the start of the allocation

    Return type:
    :   Pointer

cutlass.cute.arch.get\_dyn\_smem( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *alignment: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.arch.get_dyn_smem "Link to this definition")
:   Retrieves a pointer to a dynamic SMEM allocation.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) â The pointee type of the pointer.
        * **alignment** (*int*) â An optional pointer alignment, the result pointer is offset appropriately

    Returns:
    :   A pointer to the start of the dynamic SMEM allocation with a correct
        alignement

    Return type:
    :   Pointer

cutlass.cute.arch.get\_dyn\_smem\_size(*\**, *loc=None*, *ip=None*) → int[#](#cutlass.cute.arch.get_dyn_smem_size "Link to this definition")
:   Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time.
    This can be used for bounds checking during shared memory allocation.

    Returns:
    :   The size of dynamic shared memory in bytes

    Return type:
    :   int

cutlass.cute.arch.get\_max\_tmem\_alloc\_cols(*compute\_capability: str*) → int[#](#cutlass.cute.arch.get_max_tmem_alloc_cols "Link to this definition")
:   Get the tensor memory capacity in columns for a given compute capability.

    Returns the maximum TMEM capacity in columns available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*str*) â The compute capability string (e.g. âsm\_100â, âsm\_103â)

    Returns:
    :   The TMEM capacity in columns

    Return type:
    :   int

    Raises:
    :   **ValueError** â If the compute capability is not supported

cutlass.cute.arch.get\_min\_tmem\_alloc\_cols(*compute\_capability: str*) → int[#](#cutlass.cute.arch.get_min_tmem_alloc_cols "Link to this definition")
:   Get the minimum TMEM allocation columns for a given compute capability.

    Returns the minimum TMEM allocation columns available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*str*) â The compute capability string (e.g. âsm\_100â, âsm\_103â)

    Returns:
    :   The minimum TMEM allocation columns

    Return type:
    :   int

    Raises:
    :   **ValueError** â If the compute capability is not supported

cutlass.cute.arch.retrieve\_tmem\_ptr( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *alignment: int*, : *ptr\_to\_buffer\_holding\_addr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.arch.retrieve_tmem_ptr "Link to this definition")
:   Retrieves a pointer to TMEM with the provided element type and alignment.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) â The pointee type of the pointer.
        * **alignment** (*int*) â The alignment of the result pointer
        * **ptr\_to\_buffer\_holding\_addr** (*Pointer*) â A pointer to a SMEM buffer holding the TMEM address of the
          start of the allocation allocation

    Returns:
    :   A pointer to TMEM

    Return type:
    :   Pointer

cutlass.cute.arch.alloc\_tmem( : *num\_columns: cutlass.cute.typing.Int*, : *smem\_ptr\_to\_write\_address: cutlass.cute.typing.Pointer*, : *is\_two\_cta=None*, : *\**, : *arch: str = 'sm\_100'*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.alloc_tmem "Link to this definition")
:   Allocates TMEM.

    Parameters:
    :   * **num\_columns** (*Int*) â The number of TMEM columns to allocate
        * **smem\_ptr\_to\_write\_address** (*Pointer*) â A pointer to a SMEM buffer where the TMEM address is written
          to
        * **is\_two\_cta** â Optional boolean parameter for 2-CTA MMAs
        * **arch** (*str*) â The architecture of the GPU.

cutlass.cute.arch.relinquish\_tmem\_alloc\_permit( : *is\_two\_cta=None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.relinquish_tmem_alloc_permit "Link to this definition")
:   Relinquishes the right to allocate TMEM so that other CTAs potentially in a different grid can
    allocate.

cutlass.cute.arch.dealloc\_tmem( : *tmem\_ptr: cutlass.cute.typing.Pointer*, : *num\_columns: cutlass.cute.typing.Int*, : *is\_two\_cta=None*, : *\**, : *arch: str = 'sm\_100'*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.dealloc_tmem "Link to this definition")
:   Deallocates TMEM using the provided pointer and number of columns.

    Parameters:
    :   * **tmem\_ptr** (*Pointer*) â A pointer to the TMEM allocation to de-allocate
        * **num\_columns** (*Int*) â The number of columns in the TMEM allocation
        * **is\_two\_cta** â Optional boolean parameter for 2-CTA MMAs

cutlass.cute.arch.prmt(*src*, *src\_reg\_shifted*, *prmt\_indices*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.prmt "Link to this definition")

cutlass.cute.arch.cvt\_i8\_bf16\_intrinsic(*vec\_i8*, *length*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.arch.cvt_i8_bf16_intrinsic "Link to this definition")
:   Fast conversion from int8 to bfloat16. It converts a vector of int8 to a vector of bfloat16.

    Parameters:
    :   * **vec\_i8** (*1D vector* *of* *int8*) â The input vector of int8.
        * **length** (*int*) â The length of the input vector.

    Returns:
    :   The output 1D vector of bfloat16 with the same length as the input vector.

    Return type:
    :   1D vector of bfloat16

cutlass.cute.arch.cvt\_i4\_bf16\_intrinsic( : *vec\_i4*, : *length*, : *\**, : *with\_shuffle=False*, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.arch.cvt_i4_bf16_intrinsic "Link to this definition")
:   Fast conversion from int4 to bfloat16. It converts a vector of int4 to a vector of bfloat16.

    Parameters:
    :   * **vec\_i4** (*1D vector* *of* *int4*) â The input vector of int4.
        * **length** (*int*) â The length of the input vector.
        * **with\_shuffle** (*bool*) â Whether the input vec\_i4 follows a specific shuffle pattern.
          If True, for consecutive 8 int4 values with indices of (0, 1, 2, 3, 4, 5, 6, 7),
          the input elements are shuffled to (0, 2, 1, 3, 4, 6, 5, 7). For tailing elements less than 8,
          the shuffle pattern is (0, 2, 1, 3) for 4 elements. No shuffle is needed for less than 4 elements.
          Shuffle could help to produce converted bf16 values in the natural order of (0, 1, 2 ,3 ,4 ,5 ,6 ,7)
          without extra prmt instructions and thus better performance.

    Returns:
    :   The output 1D vector of bfloat16 with the same length as the input vector.

    Return type:
    :   1D vector of bfloat16

cutlass.cute.arch.issue\_clc\_query( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.arch.issue_clc_query "Link to this definition")
:   The clusterlaunchcontrol.try\_cancel instruction requests atomically cancelling the launch
    of a cluster that has not started running yet. It asynchronously writes an opaque response
    to shared memory indicating whether the operation succeeded or failed. On success, the
    opaque response contains the ctaid of the first CTA of the canceled cluster.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier address in SMEM
        * **clc\_response\_ptr** (*Pointer*) â A pointer to the cluster launch control response address in SMEM

cutlass.cute.arch.clc\_response( : *result\_addr: cutlass.cute.typing.Pointer*, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32][#](#cutlass.cute.arch.clc_response "Link to this definition")
:   After loading response from clusterlaunchcontrol.try\_cancel instruction into 16-byte
    register, it can be further queried using clusterlaunchcontrol.query\_cancel instruction.
    If the cluster is canceled successfully, predicate p is set to true; otherwise, it is
    set to false. If the request succeeded, clusterlaunchcontrol.query\_cancel.get\_first\_ctaid
    extracts the CTA id of the first CTA in the canceled cluster. By default, the instruction
    returns a .v4 vector whose first three elements are the x, y and z coordinate of first CTA
    in canceled cluster.

    Parameters:
    :   **result\_addr** (*Pointer*) â A pointer to the cluster launch control response address in SMEM

[previous

cutlass.cute](cute.html "previous page")
[next

Runtime](cute_runtime.html "next page")

On this page

* [API documentation](#module-cutlass.cute.arch)
  + [`make_warp_uniform()`](#cutlass.cute.arch.make_warp_uniform)
  + [`elect_one()`](#cutlass.cute.arch.elect_one)
  + [`mbarrier_init()`](#cutlass.cute.arch.mbarrier_init)
  + [`mbarrier_init_fence()`](#cutlass.cute.arch.mbarrier_init_fence)
  + [`mbarrier_arrive_and_expect_tx()`](#cutlass.cute.arch.mbarrier_arrive_and_expect_tx)
  + [`mbarrier_expect_tx()`](#cutlass.cute.arch.mbarrier_expect_tx)
  + [`mbarrier_wait()`](#cutlass.cute.arch.mbarrier_wait)
  + [`mbarrier_try_wait()`](#cutlass.cute.arch.mbarrier_try_wait)
  + [`mbarrier_conditional_try_wait()`](#cutlass.cute.arch.mbarrier_conditional_try_wait)
  + [`mbarrier_arrive()`](#cutlass.cute.arch.mbarrier_arrive)
  + [`lane_idx()`](#cutlass.cute.arch.lane_idx)
  + [`warp_idx()`](#cutlass.cute.arch.warp_idx)
  + [`thread_idx()`](#cutlass.cute.arch.thread_idx)
  + [`block_dim()`](#cutlass.cute.arch.block_dim)
  + [`block_idx()`](#cutlass.cute.arch.block_idx)
  + [`grid_dim()`](#cutlass.cute.arch.grid_dim)
  + [`cluster_idx()`](#cutlass.cute.arch.cluster_idx)
  + [`cluster_dim()`](#cutlass.cute.arch.cluster_dim)
  + [`cluster_size()`](#cutlass.cute.arch.cluster_size)
  + [`block_in_cluster_idx()`](#cutlass.cute.arch.block_in_cluster_idx)
  + [`block_in_cluster_dim()`](#cutlass.cute.arch.block_in_cluster_dim)
  + [`block_idx_in_cluster()`](#cutlass.cute.arch.block_idx_in_cluster)
  + [`dynamic_smem_size()`](#cutlass.cute.arch.dynamic_smem_size)
  + [`barrier()`](#cutlass.cute.arch.barrier)
  + [`barrier_arrive()`](#cutlass.cute.arch.barrier_arrive)
  + [`sync_threads()`](#cutlass.cute.arch.sync_threads)
  + [`sync_warp()`](#cutlass.cute.arch.sync_warp)
  + [`fence_acq_rel_cta()`](#cutlass.cute.arch.fence_acq_rel_cta)
  + [`fence_acq_rel_cluster()`](#cutlass.cute.arch.fence_acq_rel_cluster)
  + [`fence_acq_rel_gpu()`](#cutlass.cute.arch.fence_acq_rel_gpu)
  + [`fence_acq_rel_sys()`](#cutlass.cute.arch.fence_acq_rel_sys)
  + [`cp_async_commit_group()`](#cutlass.cute.arch.cp_async_commit_group)
  + [`cp_async_wait_group()`](#cutlass.cute.arch.cp_async_wait_group)
  + [`cp_async_bulk_commit_group()`](#cutlass.cute.arch.cp_async_bulk_commit_group)
  + [`cp_async_bulk_wait_group()`](#cutlass.cute.arch.cp_async_bulk_wait_group)
  + [`cluster_wait()`](#cutlass.cute.arch.cluster_wait)
  + [`cluster_arrive()`](#cutlass.cute.arch.cluster_arrive)
  + [`cluster_arrive_relaxed()`](#cutlass.cute.arch.cluster_arrive_relaxed)
  + [`vote_ballot_sync()`](#cutlass.cute.arch.vote_ballot_sync)
  + [`vote_any_sync()`](#cutlass.cute.arch.vote_any_sync)
  + [`vote_all_sync()`](#cutlass.cute.arch.vote_all_sync)
  + [`vote_uni_sync()`](#cutlass.cute.arch.vote_uni_sync)
  + [`warp_redux_sync()`](#cutlass.cute.arch.warp_redux_sync)
  + [`atomic_max_float32()`](#cutlass.cute.arch.atomic_max_float32)
  + [`atomic_add()`](#cutlass.cute.arch.atomic_add)
  + [`atomic_and()`](#cutlass.cute.arch.atomic_and)
  + [`atomic_or()`](#cutlass.cute.arch.atomic_or)
  + [`atomic_xor()`](#cutlass.cute.arch.atomic_xor)
  + [`atomic_max()`](#cutlass.cute.arch.atomic_max)
  + [`atomic_min()`](#cutlass.cute.arch.atomic_min)
  + [`atomic_exch()`](#cutlass.cute.arch.atomic_exch)
  + [`atomic_cas()`](#cutlass.cute.arch.atomic_cas)
  + [`store()`](#cutlass.cute.arch.store)
  + [`load()`](#cutlass.cute.arch.load)
  + [`popc()`](#cutlass.cute.arch.popc)
  + [`fence_proxy()`](#cutlass.cute.arch.fence_proxy)
  + [`warpgroup_reg_alloc()`](#cutlass.cute.arch.warpgroup_reg_alloc)
  + [`warpgroup_reg_dealloc()`](#cutlass.cute.arch.warpgroup_reg_dealloc)
  + [`setmaxregister_increase()`](#cutlass.cute.arch.setmaxregister_increase)
  + [`setmaxregister_decrease()`](#cutlass.cute.arch.setmaxregister_decrease)
  + [`fmax()`](#cutlass.cute.arch.fmax)
  + [`rcp_approx()`](#cutlass.cute.arch.rcp_approx)
  + [`exp2()`](#cutlass.cute.arch.exp2)
  + [`cvt_i8x4_to_f32x4()`](#cutlass.cute.arch.cvt_i8x4_to_f32x4)
  + [`cvt_i8x2_to_f32x2()`](#cutlass.cute.arch.cvt_i8x2_to_f32x2)
  + [`cvt_i8_bf16()`](#cutlass.cute.arch.cvt_i8_bf16)
  + [`cvt_i8x2_to_bf16x2()`](#cutlass.cute.arch.cvt_i8x2_to_bf16x2)
  + [`cvt_i8x4_to_bf16x4()`](#cutlass.cute.arch.cvt_i8x4_to_bf16x4)
  + [`cvt_f32x2_bf16x2()`](#cutlass.cute.arch.cvt_f32x2_bf16x2)
  + [`alloc_smem()`](#cutlass.cute.arch.alloc_smem)
  + [`get_dyn_smem()`](#cutlass.cute.arch.get_dyn_smem)
  + [`get_dyn_smem_size()`](#cutlass.cute.arch.get_dyn_smem_size)
  + [`get_max_tmem_alloc_cols()`](#cutlass.cute.arch.get_max_tmem_alloc_cols)
  + [`get_min_tmem_alloc_cols()`](#cutlass.cute.arch.get_min_tmem_alloc_cols)
  + [`retrieve_tmem_ptr()`](#cutlass.cute.arch.retrieve_tmem_ptr)
  + [`alloc_tmem()`](#cutlass.cute.arch.alloc_tmem)
  + [`relinquish_tmem_alloc_permit()`](#cutlass.cute.arch.relinquish_tmem_alloc_permit)
  + [`dealloc_tmem()`](#cutlass.cute.arch.dealloc_tmem)
  + [`prmt()`](#cutlass.cute.arch.prmt)
  + [`cvt_i8_bf16_intrinsic()`](#cutlass.cute.arch.cvt_i8_bf16_intrinsic)
  + [`cvt_i4_bf16_intrinsic()`](#cutlass.cute.arch.cvt_i4_bf16_intrinsic)
  + [`issue_clc_query()`](#cutlass.cute.arch.issue_clc_query)
  + [`clc_response()`](#cutlass.cute.arch.clc_response)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.