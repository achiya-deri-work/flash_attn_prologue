

cpasync submodule — NVIDIA CUTLASS Documentation



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
    - [cpasync submodule](#)
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
* [cutlass.cute.nvgpu](cute_nvgpu.html)
* cpasync submodule

# cpasync submodule[#](#module-cutlass.cute.nvgpu.cpasync "Link to this heading")

*class* cutlass.cute.nvgpu.cpasync.LoadCacheMode(*value*)[#](#cutlass.cute.nvgpu.cpasync.LoadCacheMode "Link to this definition")
:   Bases: `Enum`

    An enumeration for the possible cache modes of a non-bulk `cp.async` instruction.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

*class* cutlass.cute.nvgpu.cpasync.CopyG2SOp( : *cache\_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode "cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode") = cutlass.\_mlir.dialects.cute\_nvgpu.LoadCacheMode.always*, )[#](#cutlass.cute.nvgpu.cpasync.CopyG2SOp "Link to this definition")
:   Bases: `CopyOp`

    Non-bulk asynchronous GMEM to SMEM Copy Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy).

    \_\_init\_\_( : *cache\_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode "cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode") = cutlass.\_mlir.dialects.cute\_nvgpu.LoadCacheMode.always*, ) → None[#](#cutlass.cute.nvgpu.cpasync.CopyG2SOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "Link to this definition")
:   Bases: `TmaCopyOp`

    Bulk tensor asynchrnous GMEM to SMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")* *= 1*[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp.cta_group "Link to this definition")

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "Link to this definition")
:   Bases: `TmaCopyOp`

    Bulk tensor asynchrnous multicast GMEM to SMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")* *= 1*[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp.cta_group "Link to this definition")

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp "Link to this definition")
:   Bases: `TmaCopyOp`

    Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    \_\_init\_\_() → None[#](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp( : *reduction\_kind: cutlass.\_mlir.dialects.cute.ReductionOp = cutlass.\_mlir.dialects.cute.ReductionOp.ADD*, )[#](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp "Link to this definition")
:   Bases: `TmaCopyOp`

    Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk).
    This Operation uses TMA in the `.tile` mode.

    \_\_init\_\_( : *reduction\_kind: cutlass.\_mlir.dialects.cute.ReductionOp = cutlass.\_mlir.dialects.cute.ReductionOp.ADD*, ) → None[#](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp[#](#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp "Link to this definition")
:   Bases: `CopyOp`

    Asynchronous Store operation to DSMEM with explicit synchronization.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async).

    \_\_init\_\_() → None[#](#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp.__init__ "Link to this definition")

cutlass.cute.nvgpu.cpasync.make\_tiled\_tma\_atom( : *op: [CopyBulkTensorTileG2SOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileS2GOp") | [CopyReduceBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyReduceBulkTensorTileS2GOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout\_: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *cta\_tiler: cutlass.cute.typing.Tiler*, : *num\_multicast: int = 1*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor][#](#cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom "Link to this definition")
:   Makes a TMA Copy Atom in the `.tile` mode to copy tiles of a GMEM tensor to/from SMEM
    buffer with the given Layout.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a CTA-level Tiler

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    âTMA vector lengthâ to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
    layout while maintaining consistency with the provided Tiler.

    This function returns two results:

    1. the Copy Atom
    2. a TMA tensor that maps logical coordinates of the GMEM tensor to coordinates consumed by the TMA unit. TMA tensors contain basis stride elements that enable their associated layout to compute coordinates. Like other CuTe tensors, TMA tensors can be partitioned.

    Parameters:
    :   * **op** (*TMAOp*) â The TMA Copy Operation to construct an Atom
        * **gmem\_tensor** (*Tensor*) â The GMEM tensor involved in the Copy
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) â The SMEM layout to construct the Copy Atom, either w/ or w/o the stage mode
        * **cta\_tiler** (*Tiler*) â The CTA Tiler to use
        * **num\_multicast** (*int*) â The multicast factor
        * **internal\_type** (*Type**[**Numeric**]*) â Optional internal data type to use when the tensor data type is not supported by the TMA unit

    Returns:
    :   A TMA Copy Atom associated with the TMA tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

cutlass.cute.nvgpu.cpasync.tma\_partition( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *cta\_coord: cutlass.cute.typing.Coord*, : *cta\_layout: cutlass.cute.typing.Layout*, : *smem\_tensor: cutlass.cute.typing.Tensor*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor][#](#cutlass.cute.nvgpu.cpasync.tma_partition "Link to this definition")
:   Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.

cutlass.cute.nvgpu.cpasync.create\_tma\_multicast\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *cta\_coord\_vmnk: cutlass.cute.typing.Coord*, : *mcast\_mode: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int16[#](#cutlass.cute.nvgpu.cpasync.create_tma_multicast_mask "Link to this definition")
:   Computes a multicast mask for a TMA load Copy.

    Parameters:
    :   * **cta\_layout\_vmnk** (*Layout*) â The VMNK layout of the cluster
        * **cta\_coord\_vmnk** (*Coord*) â The VMNK coordinate of the current CTA
        * **mcast\_mode** (*int*) â The tensor mode in which to multicast

    Returns:
    :   The resulting mask

    Return type:
    :   Int16

cutlass.cute.nvgpu.cpasync.prefetch\_descriptor( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.cpasync.prefetch_descriptor "Link to this definition")
:   Prefetches the TMA descriptor associated with the TMA Atom.

cutlass.cute.nvgpu.cpasync.copy\_tensormap( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tensormap\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.cpasync.copy_tensormap "Link to this definition")
:   Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
    pointer.

    Parameters:
    :   * **tma\_atom** ([*CopyAtom*](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â The TMA Copy Atom
        * **tensormap\_ptr** (*Pointer*) â The pointer to the memory location to copy the tensormap to

cutlass.cute.nvgpu.cpasync.update\_tma\_descriptor( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.cpasync.update_tma_descriptor "Link to this definition")
:   Updates the TMA descriptor in the memory location pointed to by the provided pointer using
    information from a TMA Copy Atom and the provided GMEM tensor.

    Specifically, the following fields of the TMA descriptor will be updated:

    1. the GMEM tensor base address
    2. the GMEM tensor shape
    3. the GMEM tensor stride

    Other fields of the TMA descriptor are left unchanged.

    Parameters:
    :   * **tma\_atom** ([*CopyAtom*](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â The TMA Copy Atom
        * **gmem\_tensor** (*Tensor*) â The GMEM tensor
        * **tensormap\_ptr** (*Pointer*) â The pointer to the memory location of the descriptor to udpate

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_acquire( : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.cpasync.fence_tma_desc_acquire "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.nvgpu.cpasync.cp\_fence\_tma\_desc\_release( : *tma\_desc\_global\_ptr: cutlass.cute.typing.Pointer*, : *tma\_desc\_shared\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.cpasync.cp_fence_tma_desc_release "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy).

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_release(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.nvgpu.cpasync.fence_tma_desc_release "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

[previous

warpgroup submodule](cute_nvgpu_warpgroup.html "previous page")
[next

tcgen05 submodule](cute_nvgpu_tcgen05.html "next page")

On this page

* [`LoadCacheMode`](#cutlass.cute.nvgpu.cpasync.LoadCacheMode)
* [`CopyG2SOp`](#cutlass.cute.nvgpu.cpasync.CopyG2SOp)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyG2SOp.__init__)
* [`CopyBulkTensorTileG2SOp`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)
  + [`cta_group`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp.cta_group)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp.__init__)
* [`CopyBulkTensorTileG2SMulticastOp`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp)
  + [`cta_group`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp.cta_group)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp.__init__)
* [`CopyBulkTensorTileS2GOp`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp.__init__)
* [`CopyReduceBulkTensorTileS2GOp`](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp.__init__)
* [`CopyDsmemStoreOp`](#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp)
  + [`__init__()`](#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp.__init__)
* [`make_tiled_tma_atom()`](#cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom)
* [`tma_partition()`](#cutlass.cute.nvgpu.cpasync.tma_partition)
* [`create_tma_multicast_mask()`](#cutlass.cute.nvgpu.cpasync.create_tma_multicast_mask)
* [`prefetch_descriptor()`](#cutlass.cute.nvgpu.cpasync.prefetch_descriptor)
* [`copy_tensormap()`](#cutlass.cute.nvgpu.cpasync.copy_tensormap)
* [`update_tma_descriptor()`](#cutlass.cute.nvgpu.cpasync.update_tma_descriptor)
* [`fence_tma_desc_acquire()`](#cutlass.cute.nvgpu.cpasync.fence_tma_desc_acquire)
* [`cp_fence_tma_desc_release()`](#cutlass.cute.nvgpu.cpasync.cp_fence_tma_desc_release)
* [`fence_tma_desc_release()`](#cutlass.cute.nvgpu.cpasync.fence_tma_desc_release)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.