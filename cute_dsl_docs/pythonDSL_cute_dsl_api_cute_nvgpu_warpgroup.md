

warpgroup submodule — NVIDIA CUTLASS Documentation



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
    - [warpgroup submodule](#)
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
* warpgroup submodule

# warpgroup submodule[#](#module-cutlass.cute.nvgpu.warpgroup "Link to this heading")

*class* cutlass.cute.nvgpu.warpgroup.OperandMajorMode(*value*)[#](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "Link to this definition")
:   Bases: `Enum`

    An enumeration for the majorness of the input operands of the MMA.

*class* cutlass.cute.nvgpu.warpgroup.OperandSource(*value*)[#](#cutlass.cute.nvgpu.warpgroup.OperandSource "Link to this definition")
:   Bases: `Enum`

    An enumeration for the source memory location of the A input operand of the MMA.

*class* cutlass.cute.nvgpu.warpgroup.Field(*value*)[#](#cutlass.cute.nvgpu.warpgroup.Field "Link to this definition")
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    ACCUMULATE *= 'accum\_c'*[#](#cutlass.cute.nvgpu.warpgroup.Field.ACCUMULATE "Link to this definition")

*class* cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op "Link to this definition")
:   Bases: `MmaOp`

    F16/BF16 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

    descriptive\_name *= 'warpgroup F16/BF16 MMA Operation'*[#](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warpgroup.MmaF8Op( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.warpgroup.MmaF8Op "Link to this definition")
:   Bases: `MmaOp`

    F8 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

    descriptive\_name *= 'warpgroup F8 MMA Operation'*[#](#cutlass.cute.nvgpu.warpgroup.MmaF8Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.warpgroup.MmaF8Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind(*value*)[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "Link to this definition")
:   Bases: `Enum`

    Enum class for the kinds of SMEM layout atoms for SM90.

    Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
    be used to construct an SMEM layout using blocked product for operand A or B such that the
    resulting layout is legal for both TMA and UMMA.

    Note that there are other ways of creating legal layouts for operand A and B.

    MN\_INTER *= 1*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER "Link to this definition")

    MN\_SW32 *= 2*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32 "Link to this definition")

    MN\_SW64 *= 3*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64 "Link to this definition")

    MN\_SW128 *= 4*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128 "Link to this definition")

    K\_INTER *= 5*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER "Link to this definition")

    K\_SW32 *= 6*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32 "Link to this definition")

    K\_SW64 *= 7*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64 "Link to this definition")

    K\_SW128 *= 8*[#](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128 "Link to this definition")

cutlass.cute.nvgpu.warpgroup.make\_smem\_layout\_atom( : *kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.mma.SmemLayoutAtomKind")*, : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.ComposedLayout[#](#cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom "Link to this definition")
:   Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    Parameters:
    :   * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind")) â The kind of layout Atom
        * **element\_type** (*Type**[**Numeric**]*) â The element data type to construct the layout for

    Returns:
    :   The SMEM layout atom

    Return type:
    :   ComposedLayout

cutlass.cute.nvgpu.warpgroup.fence(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.nvgpu.warpgroup.fence "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence).

cutlass.cute.nvgpu.warpgroup.commit\_group(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.nvgpu.warpgroup.commit_group "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).

cutlass.cute.nvgpu.warpgroup.wait\_group(*group*, *\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.nvgpu.warpgroup.wait_group "Link to this definition")
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group).

[previous

warp submodule](cute_nvgpu_warp.html "previous page")
[next

cpasync submodule](cute_nvgpu_cpasync.html "next page")

On this page

* [`OperandMajorMode`](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode)
* [`OperandSource`](#cutlass.cute.nvgpu.warpgroup.OperandSource)
* [`Field`](#cutlass.cute.nvgpu.warpgroup.Field)
  + [`ACCUMULATE`](#cutlass.cute.nvgpu.warpgroup.Field.ACCUMULATE)
* [`MmaF16BF16Op`](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.__init__)
* [`MmaF8Op`](#cutlass.cute.nvgpu.warpgroup.MmaF8Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.warpgroup.MmaF8Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.warpgroup.MmaF8Op.__init__)
* [`SmemLayoutAtomKind`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind)
  + [`MN_INTER`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER)
  + [`MN_SW32`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32)
  + [`MN_SW64`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64)
  + [`MN_SW128`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128)
  + [`K_INTER`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER)
  + [`K_SW32`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32)
  + [`K_SW64`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64)
  + [`K_SW128`](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128)
* [`make_smem_layout_atom()`](#cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom)
* [`fence()`](#cutlass.cute.nvgpu.warpgroup.fence)
* [`commit_group()`](#cutlass.cute.nvgpu.warpgroup.commit_group)
* [`wait_group()`](#cutlass.cute.nvgpu.warpgroup.wait_group)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.