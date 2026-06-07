

Utilities for SM100 — NVIDIA CUTLASS Documentation



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
    - [Utilities for SM90](utils_sm90.html)
    - [Utilities for SM100](#)
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
* Utilities for SM100

# Utilities for SM100[#](#module-cutlass.utils.sm100 "Link to this heading")

cutlass.utils.sm100.compute\_epilogue\_tile\_shape( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *use\_2cta\_instrs: bool*, : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *\**, : *layout\_c: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum") | None = None*, : *elem\_ty\_c: Type[cutlass.cutlass\_dsl.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tile[#](#cutlass.utils.sm100.compute_epilogue_tile_shape "Link to this definition")
:   Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) â A tuple or list representing the dimensions of the CTA tile, where
          cta\_tile\_shape[0] corresponds to the height (M) and cta\_tile\_shape[1]
          corresponds to the width (N) of the tile.
        * **use\_2cta\_instrs** (*bool*) â A flag indicating whether the configuration is for a 2SM setup.
        * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type of output tensor D.
        * **layout\_c** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")*,* *optional*) â The layout enum of the input tensor C. Defaults to None.
        * **elem\_ty\_c** (*Union**[**Type**[**Numeric**]**,* *None**]**,* *optional*) â The element type for input tensor C. Defaults to None.

    Returns:
    :   Returns epilog tiler, which is used in subsequent epilog partitions.

    Return type:
    :   cute.Tile

    Raises:
    :   **ValueError** â If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

cutlass.utils.sm100.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *tiled\_tmem\_load: [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.utils.sm100.get_smem_store_op "Link to this definition")
:   Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM\_LOADâs thread-value ownership.

    Parameters:
    :   * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) â The element type for accumulator.
        * **tiled\_tmem\_load** ([*cute.TiledCopy*](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) â An instance of TiledCopy that represents the tmem load operation.

    Returns:
    :   Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

cutlass.utils.sm100.get\_tmem\_load\_op( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_tile: cutlass.cute.typing.Tile*, : *use\_2cta\_instrs: bool*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.utils.sm100.get_tmem_load_op "Link to this definition")
:   Finds a performant TMEM\_LOAD copy op for the selected epilogue
    tile (epi\_tile), element types, and tcgen05.mma instruction used.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) â A tuple or list representing the dimensions of the CTA tile.
        * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) â The element type for accumulation.
        * **epi\_tile** (*cute.Tile*) â The epilogue tile configuration.
        * **use\_2cta\_instrs** (*bool*) â A flag indicating whether the configuration is for 2 SMs.

    Returns:
    :   An instance of Sm100TmemLoad with the computed configuration.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

    Raises:
    :   **ValueError** â If the function cannot handle the given combination of accumulation
        and dimension types, or if it cannot determine the appropriate configuration based on
        the input parameters.

cutlass.utils.sm100.make\_smem\_layout\_a( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm100.make_smem_layout_a "Link to this definition")
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â The tiled MMA used to partition tensor A
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape
        * **a\_dtype** (*Type**[**Numeric**]*) â The element type for tensor A
        * **num\_stages** (*int*) â The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_smem\_layout\_b( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm100.make_smem_layout_b "Link to this definition")
:   This function helps:

    1. Get the partitioned shape of the B tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â The tiled MMA which is used to partition the B tensor.
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape.
        * **b\_dtype** (*Type**[**Numeric**]*) â The element type for the B tensor.
        * **num\_stages** (*int*) â The stage of the B tensor.

    Returns:
    :   SMEM layout for the B tensor.

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_smem\_layout\_epi( : *epi\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_layout: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.sm100.make_smem_layout_epi "Link to this definition")
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

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *acc\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")[#](#cutlass.utils.sm100.make_trivial_tiled_mma "Link to this definition")
:   Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) â Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand B (1 for K, 0 for M/N).
        * **acc\_dtype** (*type**[**Numeric**]*) â Data type of the accumulator.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) â The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) â The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) â The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** â If the data type is not supported.

cutlass.utils.sm100.make\_blockscaled\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *sf\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *sf\_vec\_size: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")[#](#cutlass.utils.sm100.make_blockscaled_trivial_tiled_mma "Link to this definition")
:   Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) â Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand B (1 for K, 0 for M/N).
        * **sf\_dtype** (*type**[**Numeric**]*) â Data type of the Scale Factor.
        * **sf\_vec\_size** (*int*) â The vector size of the Scale Factor.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) â The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) â The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) â The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** â If the data type is not supported.

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_A( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")[#](#cutlass.utils.sm100.cluster_shape_to_tma_atom_A "Link to this definition")
:   Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) â The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) â The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** â If the atom\_sm\_cnt is invalid
        * **ValueError** â If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_B( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")[#](#cutlass.utils.sm100.cluster_shape_to_tma_atom_B "Link to this definition")
:   Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) â The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) â The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** â If the atom\_sm\_cnt is invalid
        * **ValueError** â If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_SFB( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")[#](#cutlass.utils.sm100.cluster_shape_to_tma_atom_SFB "Link to this definition")
:   Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) â The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) â The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** â If the atom\_sm\_cnt is invalid
        * **ValueError** â If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.get\_permutation\_mnk( : *tile\_shape\_mnk: cutlass.cute.typing.Shape*, : *sf\_vec\_size: int*, : *use\_mxf8f6f4: bool*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[int, int, int][#](#cutlass.utils.sm100.get_permutation_mnk "Link to this definition")
:   Get the permutation of M, N, K for the tiled MMA.

    Parameters:
    :   * **tile\_shape\_mnk** (*cute.Shape*) â The shape of the tile
        * **sf\_vec\_size** (*int*) â The vector size of the Scale Factor.
        * **use\_mxf8f6f4** (*bool*) â Whether to use MXF8F6F4 or MXF4NVF4.

    Returns:
    :   The permutation of M, N, K

    Return type:
    :   Tuple[int, int, int]

    Raises:
    :   **ValueError** â If the tile shape is not divisible by the sf\_vec\_size

cutlass.utils.sm100.get\_num\_tmem\_alloc\_cols( : *tmem\_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor]*, : *rounding=True*, : *\**, : *loc=None*, : *ip=None*, ) → int[#](#cutlass.utils.sm100.get_num_tmem_alloc_cols "Link to this definition")

[previous

Utilities for SM90](utils_sm90.html "previous page")
[next

Limitations](../limitations.html "next page")

On this page

* [`compute_epilogue_tile_shape()`](#cutlass.utils.sm100.compute_epilogue_tile_shape)
* [`get_smem_store_op()`](#cutlass.utils.sm100.get_smem_store_op)
* [`get_tmem_load_op()`](#cutlass.utils.sm100.get_tmem_load_op)
* [`make_smem_layout_a()`](#cutlass.utils.sm100.make_smem_layout_a)
* [`make_smem_layout_b()`](#cutlass.utils.sm100.make_smem_layout_b)
* [`make_smem_layout_epi()`](#cutlass.utils.sm100.make_smem_layout_epi)
* [`make_trivial_tiled_mma()`](#cutlass.utils.sm100.make_trivial_tiled_mma)
* [`make_blockscaled_trivial_tiled_mma()`](#cutlass.utils.sm100.make_blockscaled_trivial_tiled_mma)
* [`cluster_shape_to_tma_atom_A()`](#cutlass.utils.sm100.cluster_shape_to_tma_atom_A)
* [`cluster_shape_to_tma_atom_B()`](#cutlass.utils.sm100.cluster_shape_to_tma_atom_B)
* [`cluster_shape_to_tma_atom_SFB()`](#cutlass.utils.sm100.cluster_shape_to_tma_atom_SFB)
* [`get_permutation_mnk()`](#cutlass.utils.sm100.get_permutation_mnk)
* [`get_num_tmem_alloc_cols()`](#cutlass.utils.sm100.get_num_tmem_alloc_cols)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.