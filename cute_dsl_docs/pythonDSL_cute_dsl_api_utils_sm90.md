

Utilities for SM90 — NVIDIA CUTLASS Documentation



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
    - [Utilities for SM90](#)
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
* [cutlass.utils](utils.html)
* Utilities for SM90

# Utilities for SM90[#](#module-cutlass.utils.sm90 "Link to this heading")

cutlass.utils.sm90.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.utils.sm90.get_smem_store_op "Link to this definition")
:   Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

    ## Parameters:[#](#parameters "Link to this heading")

    layout\_dLayoutEnum
    :   The layout enum of the output tensor D.

    elem\_ty\_dType[Numeric]
    :   The element type for output tensor D.

    elem\_ty\_accType[Numeric]
    :   The element type for accumulator.

    ## Returns:[#](#returns "Link to this heading")

    Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

cutlass.utils.sm90.make\_smem\_layout\_a( : *a\_layout: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm90.make_smem_layout_a "Link to this definition")
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **a\_layout** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum for tensor A
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape
        * **a\_dtype** (*Type**[**Numeric**]*) â The element type for tensor A
        * **num\_stages** (*int*) â The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.make\_smem\_layout\_b( : *b\_layout: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm90.make_smem_layout_b "Link to this definition")
:   This function helps with:

    1. Get the partitioned shape of the B tensor based on the MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **b\_layout** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum for tensor B
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape
        * **b\_dtype** (*Type**[**Numeric**]*) â The element type for tensor B
        * **num\_stages** (*int*) â The number of pipeline stages for tensor B

    Returns:
    :   SMEM layout for tensor B

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.make\_smem\_layout\_epi( : *epi\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_layout: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, : *smem\_trg\_shape: cutlass.cute.typing.Layout | None = None*, : *smem\_order: tuple | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm90.make_smem_layout_epi "Link to this definition")
:   This function helps:

    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensorâs majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **epi\_dtype** (*Type**[**Numeric**]*) â The element type for the epilog tensor.
        * **epi\_layout** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum for the epilog tensor.
        * **epi\_tile** (*cute.cute.Tile*) â The epilogue tile shape.
        * **epi\_stage** (*int*) â The stage of the epilog tensor.
        * **smem\_trg\_shape** (*cute.Layout* *|* *None*) â Target shape for SMEM layout (optional).
        * **smem\_order** (*tuple* *|* *None*) â Order for SMEM layout (optional).

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.compute\_tile\_shape\_or\_override( : *tile\_shape\_mnk: tuple[int, int, int]*, : *element\_type: type[cutlass.cutlass\_dsl.Numeric]*, : *is\_cooperative: bool = False*, : *epi\_tile\_override: tuple[int, int] | None = None*, ) → tuple[int, int][#](#cutlass.utils.sm90.compute_tile_shape_or_override "Link to this definition")
:   Compute the epilogue tile shape or use override if provided.

    Parameters:
    :   * **tile\_shape\_mnk** (*Tuple**[**int**,* *int**,* *int**]*) â CTA tile shape (M,N,K)
        * **element\_type** (*type**[**Numeric**]*) â Data type of elements
        * **is\_cooperative** (*bool*) â Whether to use cooperative approach
        * **epi\_tile\_override** (*Tuple**[**int**,* *int**] or* *None*) â Optional override for epilogue tile shape

    Returns:
    :   Computed epilogue tile shape

    Return type:
    :   Tuple[int, int]

[previous

cutlass.utils](utils.html "previous page")
[next

Utilities for SM100](utils_sm100.html "next page")

On this page

* [`get_smem_store_op()`](#cutlass.utils.sm90.get_smem_store_op)
* [`make_smem_layout_a()`](#cutlass.utils.sm90.make_smem_layout_a)
* [`make_smem_layout_b()`](#cutlass.utils.sm90.make_smem_layout_b)
* [`make_smem_layout_epi()`](#cutlass.utils.sm90.make_smem_layout_epi)
* [`compute_tile_shape_or_override()`](#cutlass.utils.sm90.compute_tile_shape_or_override)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.