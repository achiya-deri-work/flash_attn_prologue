

warp submodule — NVIDIA CUTLASS Documentation



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
    - [warp submodule](#)
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
* warp submodule

# warp submodule[#](#module-cutlass.cute.nvgpu.warp "Link to this heading")

*class* cutlass.cute.nvgpu.warp.Field(*value*)[#](#cutlass.cute.nvgpu.warp.Field "Link to this definition")
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    ACCUMULATE *= 'accum\_c'*[#](#cutlass.cute.nvgpu.warp.Field.ACCUMULATE "Link to this definition")

    SFA *= 'sf\_a'*[#](#cutlass.cute.nvgpu.warp.Field.SFA "Link to this definition")

    SFB *= 'sf\_b'*[#](#cutlass.cute.nvgpu.warp.Field.SFB "Link to this definition")

*class* cutlass.cute.nvgpu.warp.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *shape\_mnk: cutlass.cute.typing.Shape*, )[#](#cutlass.cute.nvgpu.warp.MmaF16BF16Op "Link to this definition")
:   Bases: `WarpMmaOp`

    F16/BF16 warp-level MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
    This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

    ab\_dtype*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.ab_dtype "Link to this definition")

    acc\_dtype*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.acc_dtype "Link to this definition")

    shape\_mnk*: cutlass.cute.typing.Shape*[#](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.shape_mnk "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *shape\_mnk: cutlass.cute.typing.Shape*, ) → None[#](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.MmaMXF4Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *sf\_type: Type[cutlass.cute.typing.Numeric]*, )[#](#cutlass.cute.nvgpu.warp.MmaMXF4Op "Link to this definition")
:   Bases: `MmaSM120BlockScaledOp`

    MXF4 warp-level MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
    This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
    .kind = {.kind::mxf4};
    .scale\_vec\_size = {.scale\_vec::2X};
    .stype = {.ue8m0};

    descriptive\_name *= 'warp-level MXF4 MMA Operation'*[#](#cutlass.cute.nvgpu.warp.MmaMXF4Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *sf\_type: Type[cutlass.cute.typing.Numeric]*, ) → None[#](#cutlass.cute.nvgpu.warp.MmaMXF4Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *sf\_type: Type[cutlass.cute.typing.Numeric]*, )[#](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op "Link to this definition")
:   Bases: `MmaSM120BlockScaledOp`

    MXF4NVF4 warp-level MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
    This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
    .kind = {.kind::mxf4nvf4};
    .scale\_vec\_size = {.scale\_vec::2X, .scale\_vec::4X};
    .stype = {.ue8m0, .ue4m3};

    descriptive\_name *= 'warp-level MXF4NVF4 MMA Operation'*[#](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *sf\_type: Type[cutlass.cute.typing.Numeric]*, ) → None[#](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, )[#](#cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp "Link to this definition")
:   Bases: `BaseOp`

    8x8 `ldmatrix` Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
    This operation corresponds to the `.m8n8` qualifier.

    \_\_init\_\_( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, ) → None[#](#cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, )[#](#cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp "Link to this definition")
:   Bases: `BaseOp`

    16x8 8b `ldmatrix` Operation with transpose

    There is no direct PTX correspondance to this Op.
    This actually lowers to ldmatrix with the `.m16n16` qualifier and
    additional address and value permutations to match stmatrix.m16n8.trans.
    Useful for vectorizing with Ampere-style 8x8 matrix thread-value layouts

    \_\_init\_\_( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, ) → None[#](#cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, )[#](#cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp "Link to this definition")
:   Bases: `BaseOp`

    16x16 `ldmatrix` Operation with transpose and optional unpacking to 8b container.
    Packed source container is 16x4b elements with 64b padding
    or 16x6b elements with 32b padding (total 128b per 16 elements)

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
    This operation corresponds to the `.m16n16` and the `.b4x16_p64`,``.b6x16\_p32``,``.b8`` qualifiers.

    \_\_init\_\_( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, ) → None[#](#cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, )[#](#cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp "Link to this definition")
:   Bases: `BaseOp`

    8x8 `stmatrix` Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
    This operation corresponds to the `m8n8` qualifier.

    \_\_init\_\_( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, ) → None[#](#cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, )[#](#cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp "Link to this definition")
:   Bases: `BaseOp`

    16x8 `stmatrix` Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
    This operation corresponds to the `m16n8` qualifier.

    \_\_init\_\_( : *transpose: bool = False*, : *num\_matrices: int = 1*, : *unpack\_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None*, ) → None[#](#cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp.__init__ "Link to this definition")

[previous

Common](cute_nvgpu_common.html "previous page")
[next

warpgroup submodule](cute_nvgpu_warpgroup.html "next page")

On this page

* [`Field`](#cutlass.cute.nvgpu.warp.Field)
  + [`ACCUMULATE`](#cutlass.cute.nvgpu.warp.Field.ACCUMULATE)
  + [`SFA`](#cutlass.cute.nvgpu.warp.Field.SFA)
  + [`SFB`](#cutlass.cute.nvgpu.warp.Field.SFB)
* [`MmaF16BF16Op`](#cutlass.cute.nvgpu.warp.MmaF16BF16Op)
  + [`ab_dtype`](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.ab_dtype)
  + [`acc_dtype`](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.acc_dtype)
  + [`shape_mnk`](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.shape_mnk)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.MmaF16BF16Op.__init__)
* [`MmaMXF4Op`](#cutlass.cute.nvgpu.warp.MmaMXF4Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.warp.MmaMXF4Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.MmaMXF4Op.__init__)
* [`MmaMXF4NVF4Op`](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.__init__)
* [`LdMatrix8x8x16bOp`](#cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp.__init__)
* [`LdMatrix16x8x8bOp`](#cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp.__init__)
* [`LdMatrix16x16x8bOp`](#cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp.__init__)
* [`StMatrix8x8x16bOp`](#cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp.__init__)
* [`StMatrix16x8x8bOp`](#cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp.__init__)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.