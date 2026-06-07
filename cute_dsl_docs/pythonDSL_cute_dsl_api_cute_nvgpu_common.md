

Common — NVIDIA CUTLASS Documentation



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
    - [Common](#)
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
* [cutlass.cute.nvgpu](cute_nvgpu.html)
* Common

# Common[#](#module-cutlass.cute.nvgpu "Link to this heading")

*class* cutlass.cute.nvgpu.OpError(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.nvgpu.OpError "Link to this definition")
:   Bases: `DSLBaseError`

    An exception class for Op construction errors.

cutlass.cute.nvgpu.normalize\_field\_to\_ir\_name(*field*, *admissible\_fields*) → str[#](#cutlass.cute.nvgpu.normalize_field_to_ir_name "Link to this definition")
:   Normalize a field specifier to its IR logical field name.

    Accepted inputs:

    * Enum value present in admissible\_fields (must expose \_to\_ir\_field\_name()).
    * Exact string IR name (e.g., âaccum\_câ, âneg\_aâ, âsf\_aâ).

    Any other form is rejected.

*class* cutlass.cute.nvgpu.MmaUniversalOp(*abacc\_dtype: Type[cutlass.cute.typing.Numeric]*)[#](#cutlass.cute.nvgpu.MmaUniversalOp "Link to this definition")
:   Bases: `MmaOp`

    The universal MMA Operation.

    This Operation currently expects the A/B operands as well as the accumulator to share the same
    data types.

    Parameters:
    :   **abacc\_dtype** (*Type**[**Numeric**]*) â The data type for the A/B operands and the accumulator

    abacc\_dtype*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.nvgpu.MmaUniversalOp.abacc_dtype "Link to this definition")

*class* cutlass.cute.nvgpu.MmaUniversalTrait(*value: cutlass.\_mlir.ir.Value*)[#](#cutlass.cute.nvgpu.MmaUniversalTrait "Link to this definition")
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.CopyUniversalOp[#](#cutlass.cute.nvgpu.CopyUniversalOp "Link to this definition")
:   Bases: `CopyOp`

    The universal Copy Operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyUniversalOp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL
    )
    ```

    * `tensor_dtype` is the data type used to build the reference TV Layout (either the source or the destination TV Layout) in unit of tensor elements and is used for partitioning by `TiledCopy` for example
    * `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom execution. This can be larger than the width of the above data type. When not provided, the compiler will do a best effort at auto-vectorizing.
    * `l1c_evict_priority` is a kw argument specifying the L1 cache eviction priority hint for the copy operation. Defaults to `EVICT_NORMAL` if not provided.
    * `invariant` is a kw argument specifying whether the load is invariant (read-only data that never changes). This enables compiler optimizations like instruction reordering. Defaults to `False` if not provided.

*class* cutlass.cute.nvgpu.CopyUniversalTrait(*value: cutlass.\_mlir.ir.Value*)[#](#cutlass.cute.nvgpu.CopyUniversalTrait "Link to this definition")
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.MemoryOrder(*value*)[#](#cutlass.cute.nvgpu.MemoryOrder "Link to this definition")
:   Bases: `Enum`

    An enumeration.

*class* cutlass.cute.nvgpu.MemoryScope(*value*)[#](#cutlass.cute.nvgpu.MemoryScope "Link to this definition")
:   Bases: `Enum`

    An enumeration.

*class* cutlass.cute.nvgpu.CacheEvictionPriority(*value*)[#](#cutlass.cute.nvgpu.CacheEvictionPriority "Link to this definition")
:   Bases: `Enum`

    An enumeration.

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_A( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor][#](#cutlass.cute.nvgpu.make_tiled_tma_atom_A "Link to this definition")
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the MK projections of the TiledMMA for A tensor loads.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a MMA Tiler
    * a TiledMma
    * a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    âTMA vector lengthâ to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the M-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   * **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) â The Copy Operation to construct an Atom for
        * **gmem\_tensor** (*Tensor*) â The GMEM tensor to be loaded by this copy atom
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) â Shared memory layout to load the tensor into (PDSL)
        * **mma\_tiler\_mnk** (*Shape*) â The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        * **tiled\_mma** ([*atom.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")) â The TiledMMA that will consume the load as operands
        * **cluster\_shape\_vmnk** (*Shape*) â The Cluster-level shape in VMNK dimensions
        * **internal\_type** (*Type**[**Numeric**]*) â An optional parameter for the internal data type to when element
          type does not match the copy type

    Returns:
    :   A copy atom for this operation and the associated TMA coord tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_B( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor][#](#cutlass.cute.nvgpu.make_tiled_tma_atom_B "Link to this definition")
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the NK projections of the TiledMMA for B tensor loads.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a MMA Tiler
    * a TiledMma
    * a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    âTMA vector lengthâ to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the N-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   * **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) â The Copy Operation to construct an Atom for
        * **gmem\_tensor** (*Tensor*) â The GMEM tensor to be loaded by this copy atom
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) â Shared memory layout to load the tensor into (PDSL)
        * **mma\_tiler\_mnk** (*Shape*) â The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        * **tiled\_mma** (*core.TiledMma*) â The TiledMMA that will consume the load as operands
        * **cluster\_shape\_vmnk** (*Shape*) â The Cluster-level shape in VMNK dimensions
        * **internal\_type** (*Type**[**Numeric**]*) â An optional parameter for the internal data type to when element
          type does not match the copy type

    Returns:
    :   A Copy Atom for this Operation and the associated TMA tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

[previous

cutlass.cute.nvgpu](cute_nvgpu.html "previous page")
[next

warp submodule](cute_nvgpu_warp.html "next page")

On this page

* [`OpError`](#cutlass.cute.nvgpu.OpError)
* [`normalize_field_to_ir_name()`](#cutlass.cute.nvgpu.normalize_field_to_ir_name)
* [`MmaUniversalOp`](#cutlass.cute.nvgpu.MmaUniversalOp)
  + [`abacc_dtype`](#cutlass.cute.nvgpu.MmaUniversalOp.abacc_dtype)
* [`MmaUniversalTrait`](#cutlass.cute.nvgpu.MmaUniversalTrait)
* [`CopyUniversalOp`](#cutlass.cute.nvgpu.CopyUniversalOp)
* [`CopyUniversalTrait`](#cutlass.cute.nvgpu.CopyUniversalTrait)
* [`MemoryOrder`](#cutlass.cute.nvgpu.MemoryOrder)
* [`MemoryScope`](#cutlass.cute.nvgpu.MemoryScope)
* [`CacheEvictionPriority`](#cutlass.cute.nvgpu.CacheEvictionPriority)
* [`make_tiled_tma_atom_A()`](#cutlass.cute.nvgpu.make_tiled_tma_atom_A)
* [`make_tiled_tma_atom_B()`](#cutlass.cute.nvgpu.make_tiled_tma_atom_B)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.