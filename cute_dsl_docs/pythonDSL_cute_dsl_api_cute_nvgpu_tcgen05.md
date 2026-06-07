

tcgen05 submodule — NVIDIA CUTLASS Documentation



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
    - [tcgen05 submodule](#)
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
* tcgen05 submodule

# tcgen05 submodule[#](#module-cutlass.cute.nvgpu.tcgen05 "Link to this heading")

*class* cutlass.cute.nvgpu.tcgen05.Repetition(*value*)[#](#cutlass.cute.nvgpu.tcgen05.Repetition "Link to this definition")
:   Bases: `Enum`

    An enumeration for the number of repetitions of a given TMEM copy within the instruction.

    x1 *= 1*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x1 "Link to this definition")

    x2 *= 2*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x2 "Link to this definition")

    x4 *= 4*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x4 "Link to this definition")

    x8 *= 8*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x8 "Link to this definition")

    x16 *= 16*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x16 "Link to this definition")

    x32 *= 32*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x32 "Link to this definition")

    x64 *= 64*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x64 "Link to this definition")

    x128 *= 128*[#](#cutlass.cute.nvgpu.tcgen05.Repetition.x128 "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp(*value*)[#](#cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp "Link to this definition")
:   Bases: `Enum`

    An enumeration for the possible reduce operations for TMEM load operations.

*class* cutlass.cute.nvgpu.tcgen05.Pack(*value*)[#](#cutlass.cute.nvgpu.tcgen05.Pack "Link to this definition")
:   Bases: `Enum`

    An enumeration for the possible packing patterns for TMEM to RMEM copies.

    NONE *= 1*[#](#cutlass.cute.nvgpu.tcgen05.Pack.NONE "Link to this definition")

    PACK\_16b\_IN\_32b *= 2*[#](#cutlass.cute.nvgpu.tcgen05.Pack.PACK_16b_IN_32b "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Unpack(*value*)[#](#cutlass.cute.nvgpu.tcgen05.Unpack "Link to this definition")
:   Bases: `Enum`

    An enumeration for the possible unpacking patterns for RMEM to TMEM copies.

    NONE *= 1*[#](#cutlass.cute.nvgpu.tcgen05.Unpack.NONE "Link to this definition")

    UNPACK\_32b\_IN\_16b *= 2*[#](#cutlass.cute.nvgpu.tcgen05.Unpack.UNPACK_32b_IN_16b "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Ld16x64bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.Ld16x64bOp "Link to this definition")
:   Bases: `_LdBase`

    16x64b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x64b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.Ld16x64bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Ld16x128bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.Ld16x128bOp "Link to this definition")
:   Bases: `_LdBase`

    16x128b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x128b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.Ld16x128bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Ld16x256bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.Ld16x256bOp "Link to this definition")
:   Bases: `_LdBase`

    16x256b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x256b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.Ld16x256bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op "Link to this definition")
:   Bases: `_LdBase`

    16x32bx2 TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x32bx2` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Ld32x32bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.Ld32x32bOp "Link to this definition")
:   Bases: `_LdBase`

    32x32b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.32x32` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.Ld32x32bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.St16x64bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.St16x64bOp "Link to this definition")
:   Bases: `_StBase`

    16x64b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x64` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.St16x64bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.St16x128bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.St16x128bOp "Link to this definition")
:   Bases: `_StBase`

    16x128b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x128` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.St16x128bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.St16x256bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.St16x256bOp "Link to this definition")
:   Bases: `_StBase`

    16x256b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x256` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.St16x256bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.St16x32bx2Op( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.St16x32bx2Op "Link to this definition")
:   Bases: `_StBase`

    16x32x2b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x32x2` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.St16x32bx2Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.St32x32bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )[#](#cutlass.cute.nvgpu.tcgen05.St32x32bOp "Link to this definition")
:   Bases: `_StBase`

    32x32b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.32x32` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.St32x32bOp.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.OperandMajorMode(*value*)[#](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "Link to this definition")
:   Bases: `Enum`

    An enumeration for the majorness of the input operands of the MMA.

*class* cutlass.cute.nvgpu.tcgen05.OperandSource(*value*)[#](#cutlass.cute.nvgpu.tcgen05.OperandSource "Link to this definition")
:   Bases: `Enum`

    An enumeration for the source memory location of the A input operand of the MMA.

*class* cutlass.cute.nvgpu.tcgen05.CtaGroup(*value*)[#](#cutlass.cute.nvgpu.tcgen05.CtaGroup "Link to this definition")
:   Bases: `Enum`

    An enumeration for the `cta_group` qualifier of the MMA.

    ONE *= 1*[#](#cutlass.cute.nvgpu.tcgen05.CtaGroup.ONE "Link to this definition")

    TWO *= 2*[#](#cutlass.cute.nvgpu.tcgen05.CtaGroup.TWO "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.Field(*value*)[#](#cutlass.cute.nvgpu.tcgen05.Field "Link to this definition")
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    NEGATE\_A *= 'neg\_a'*[#](#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_A "Link to this definition")

    NEGATE\_B *= 'neg\_b'*[#](#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_B "Link to this definition")

    ACCUMULATE *= 'accum\_c'*[#](#cutlass.cute.nvgpu.tcgen05.Field.ACCUMULATE "Link to this definition")

    SFA *= 'sf\_a'*[#](#cutlass.cute.nvgpu.tcgen05.Field.SFA "Link to this definition")

    SFB *= 'sf\_b'*[#](#cutlass.cute.nvgpu.tcgen05.Field.SFB "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaTF32Op( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op "Link to this definition")
:   Bases: `MmaOp`

    TF32 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::tf32` qualifier.

    descriptive\_name *= 'tcgen05 TF32 MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op "Link to this definition")
:   Bases: `MmaOp`

    F16/BF16 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::f16` qualifier.

    descriptive\_name *= 'tcgen05 F16/BF16 MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaI8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaI8Op "Link to this definition")
:   Bases: `MmaOp`

    I8 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::i8` qualifier.

    descriptive\_name *= 'tcgen05 I8 MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaI8Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaI8Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaFP8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op "Link to this definition")
:   Bases: `MmaOp`

    F8 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).

    descriptive\_name *= 'tcgen05 F8 MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op "Link to this definition")
:   Bases: `BlockScaledMmaOp`

    MXF8 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

    descriptive\_name *= 'tcgen05 MXF8 BlockScaled MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF4Op( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op "Link to this definition")
:   Bases: `BlockScaledMmaOp`

    MXF4 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf4` qualifier.

    descriptive\_name *= 'tcgen05 MXF4 BlockScaled MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op( : *sf\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, )[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op "Link to this definition")
:   Bases: `BlockScaledMmaOp`

    MXF4NVF4 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf4nvf4` qualifier.

    descriptive\_name *= 'tcgen05 MXF4NVF4 BlockScaled MMA Operation'*[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.descriptive_name "Link to this definition")

    \_\_init\_\_( : *sf\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.__init__ "Link to this definition")

*class* cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind(*value*)[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "Link to this definition")
:   Bases: `Enum`

    Enum class for the kinds of SMEM layout atoms for SM100.

    Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be
    used to construct an SMEM layout using blocked product for operand A or B such that the
    resulting layout is legal for both TMA and UMMA.

    Note that there are other ways of creating legal layouts for operand A and B.

    MN\_INTER *= 1*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER "Link to this definition")

    MN\_SW32 *= 2*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW32 "Link to this definition")

    MN\_SW64 *= 3*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64 "Link to this definition")

    MN\_SW128 *= 4*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128 "Link to this definition")

    MN\_SW128\_32B *= 5*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128_32B "Link to this definition")

    K\_INTER *= 6*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER "Link to this definition")

    K\_SW32 *= 7*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW32 "Link to this definition")

    K\_SW64 *= 8*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64 "Link to this definition")

    K\_SW128 *= 9*[#](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128 "Link to this definition")

cutlass.cute.nvgpu.tcgen05.make\_smem\_layout\_atom( : *kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.mma.SmemLayoutAtomKind")*, : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.ComposedLayout[#](#cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom "Link to this definition")
:   Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    Parameters:
    :   * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind")) â The kind of layout Atom
        * **element\_type** (*Type**[**Numeric**]*) â The element data type to construct the layout for

    Returns:
    :   The SMEM layout atom

    Return type:
    :   ComposedLayout

cutlass.cute.nvgpu.tcgen05.tile\_to\_mma\_shape( : *atom*, : *mma\_tile\_shape: cutlass.cute.typing.Shape*, : *order: cutlass.cute.typing.IntTuple | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape "Link to this definition")
:   Tiles a layout to an MMA shape.

cutlass.cute.nvgpu.tcgen05.commit( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *mask=None*, : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.nvgpu.tcgen05.commit "Link to this definition")
:   Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) â A pointer to the mbarrier in SMEM
        * **mask** (*Int*) â An optional multicast mask for the CTAs in the cluster to signal arrival to

cutlass.cute.nvgpu.tcgen05.is\_tmem\_load(*atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*) → bool[#](#cutlass.cute.nvgpu.tcgen05.is_tmem_load "Link to this definition")
:   Returns whether a CopyAtom instance is a TMEM load.

cutlass.cute.nvgpu.tcgen05.is\_tmem\_store(*atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*) → bool[#](#cutlass.cute.nvgpu.tcgen05.is_tmem_store "Link to this definition")
:   Returns whether a CopyAtom instance is a TMEM store.

cutlass.cute.nvgpu.tcgen05.get\_tmem\_copy\_properties( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, ) → Tuple[int, int, int, [Pack](#cutlass.cute.nvgpu.tcgen05.Pack "cutlass.cute.nvgpu.tcgen05.copy.Pack") | [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack "cutlass.cute.nvgpu.tcgen05.copy.Unpack")][#](#cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties "Link to this definition")
:   Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
    and whether packing/unpacking is used).

cutlass.cute.nvgpu.tcgen05.find\_tmem\_tensor\_col\_offset( : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int[#](#cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset "Link to this definition")
:   Computes the TMEM column offset given a TMEM tensor.

    Parameters:
    :   **tmem\_tensor** (*Tensor*) â The TMEM tensor to use to compute the columns offset

    Returns:
    :   The columns offset

    Return type:
    :   Int

cutlass.cute.nvgpu.tcgen05.make\_tmem\_copy( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")[#](#cutlass.cute.nvgpu.tcgen05.make_tmem_copy "Link to this definition")
:   Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

cutlass.cute.nvgpu.tcgen05.make\_s2t\_copy( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")[#](#cutlass.cute.nvgpu.tcgen05.make_s2t_copy "Link to this definition")
:   Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

cutlass.cute.nvgpu.tcgen05.get\_s2t\_smem\_desc\_tensor( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *smem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor "Link to this definition")
:   Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

cutlass.cute.nvgpu.tcgen05.make\_umma\_smem\_desc( : *src: cutlass.cute.typing.Pointer*, : *layout: cutlass.cute.typing.Layout*, : *major: str*, : *next\_src: cutlass.cute.typing.Pointer | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc "Link to this definition")
:   Construct shared memory descriptor for UMMA.

    The make\_umma\_smem\_desc operation accepts an input cute.ptr (optionally a nextSrc
    pointer for the second buffer in a circular buffer scheme), alongside a cute.layout
    and a major attr, then constructs the shared memory descriptor and returns it.
    The layout must be describing the buffer pointed to by the input pointer and the
    iterator must carry valid swizzle information.

    There are 5 supported swizzle variants:
    - S<0, 4, 3> | SWIZZLE\_NONE
    - S<1, 4, 3> | SWIZZLE\_32B
    - S<2, 4, 3> | SWIZZLE\_64B
    - S<3, 4, 3> | SWIZZLE\_128B
    - S<2, 5, 2> | SWIZZLE\_128B\_BASE32B

    The cute.ptr must carry shared address space and must be aligned to 16B.

    Parameters:
    :   * **src** (*Pointer*) â The source pointer to shared memory
        * **layout** (*Layout*) â The layout describing the buffer
        * **major** (*str*) â The major mode attribute
        * **next\_src** (*Optional**[**Pointer**]*) â Optional next source pointer for circular buffer scheme

    Returns:
    :   The shared memory descriptor

    Return type:
    :   SmemDescType

[previous

cpasync submodule](cute_nvgpu_cpasync.html "previous page")
[next

cutlass.pipeline](pipeline.html "next page")

On this page

* [`Repetition`](#cutlass.cute.nvgpu.tcgen05.Repetition)
  + [`x1`](#cutlass.cute.nvgpu.tcgen05.Repetition.x1)
  + [`x2`](#cutlass.cute.nvgpu.tcgen05.Repetition.x2)
  + [`x4`](#cutlass.cute.nvgpu.tcgen05.Repetition.x4)
  + [`x8`](#cutlass.cute.nvgpu.tcgen05.Repetition.x8)
  + [`x16`](#cutlass.cute.nvgpu.tcgen05.Repetition.x16)
  + [`x32`](#cutlass.cute.nvgpu.tcgen05.Repetition.x32)
  + [`x64`](#cutlass.cute.nvgpu.tcgen05.Repetition.x64)
  + [`x128`](#cutlass.cute.nvgpu.tcgen05.Repetition.x128)
* [`TmemLoadRedOp`](#cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp)
* [`Pack`](#cutlass.cute.nvgpu.tcgen05.Pack)
  + [`NONE`](#cutlass.cute.nvgpu.tcgen05.Pack.NONE)
  + [`PACK_16b_IN_32b`](#cutlass.cute.nvgpu.tcgen05.Pack.PACK_16b_IN_32b)
* [`Unpack`](#cutlass.cute.nvgpu.tcgen05.Unpack)
  + [`NONE`](#cutlass.cute.nvgpu.tcgen05.Unpack.NONE)
  + [`UNPACK_32b_IN_16b`](#cutlass.cute.nvgpu.tcgen05.Unpack.UNPACK_32b_IN_16b)
* [`Ld16x64bOp`](#cutlass.cute.nvgpu.tcgen05.Ld16x64bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.Ld16x64bOp.__init__)
* [`Ld16x128bOp`](#cutlass.cute.nvgpu.tcgen05.Ld16x128bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.Ld16x128bOp.__init__)
* [`Ld16x256bOp`](#cutlass.cute.nvgpu.tcgen05.Ld16x256bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.Ld16x256bOp.__init__)
* [`Ld16x32bx2Op`](#cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op.__init__)
* [`Ld32x32bOp`](#cutlass.cute.nvgpu.tcgen05.Ld32x32bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.Ld32x32bOp.__init__)
* [`St16x64bOp`](#cutlass.cute.nvgpu.tcgen05.St16x64bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.St16x64bOp.__init__)
* [`St16x128bOp`](#cutlass.cute.nvgpu.tcgen05.St16x128bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.St16x128bOp.__init__)
* [`St16x256bOp`](#cutlass.cute.nvgpu.tcgen05.St16x256bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.St16x256bOp.__init__)
* [`St16x32bx2Op`](#cutlass.cute.nvgpu.tcgen05.St16x32bx2Op)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.St16x32bx2Op.__init__)
* [`St32x32bOp`](#cutlass.cute.nvgpu.tcgen05.St32x32bOp)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.St32x32bOp.__init__)
* [`OperandMajorMode`](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)
* [`OperandSource`](#cutlass.cute.nvgpu.tcgen05.OperandSource)
* [`CtaGroup`](#cutlass.cute.nvgpu.tcgen05.CtaGroup)
  + [`ONE`](#cutlass.cute.nvgpu.tcgen05.CtaGroup.ONE)
  + [`TWO`](#cutlass.cute.nvgpu.tcgen05.CtaGroup.TWO)
* [`Field`](#cutlass.cute.nvgpu.tcgen05.Field)
  + [`NEGATE_A`](#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_A)
  + [`NEGATE_B`](#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_B)
  + [`ACCUMULATE`](#cutlass.cute.nvgpu.tcgen05.Field.ACCUMULATE)
  + [`SFA`](#cutlass.cute.nvgpu.tcgen05.Field.SFA)
  + [`SFB`](#cutlass.cute.nvgpu.tcgen05.Field.SFB)
* [`MmaTF32Op`](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaTF32Op.__init__)
* [`MmaF16BF16Op`](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.__init__)
* [`MmaI8Op`](#cutlass.cute.nvgpu.tcgen05.MmaI8Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaI8Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaI8Op.__init__)
* [`MmaFP8Op`](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaFP8Op.__init__)
* [`MmaMXF8Op`](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.__init__)
* [`MmaMXF4Op`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.__init__)
* [`MmaMXF4NVF4Op`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op)
  + [`descriptive_name`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.descriptive_name)
  + [`__init__()`](#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.__init__)
* [`SmemLayoutAtomKind`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind)
  + [`MN_INTER`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER)
  + [`MN_SW32`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW32)
  + [`MN_SW64`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64)
  + [`MN_SW128`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128)
  + [`MN_SW128_32B`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128_32B)
  + [`K_INTER`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER)
  + [`K_SW32`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW32)
  + [`K_SW64`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64)
  + [`K_SW128`](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128)
* [`make_smem_layout_atom()`](#cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom)
* [`tile_to_mma_shape()`](#cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape)
* [`commit()`](#cutlass.cute.nvgpu.tcgen05.commit)
* [`is_tmem_load()`](#cutlass.cute.nvgpu.tcgen05.is_tmem_load)
* [`is_tmem_store()`](#cutlass.cute.nvgpu.tcgen05.is_tmem_store)
* [`get_tmem_copy_properties()`](#cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties)
* [`find_tmem_tensor_col_offset()`](#cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset)
* [`make_tmem_copy()`](#cutlass.cute.nvgpu.tcgen05.make_tmem_copy)
* [`make_s2t_copy()`](#cutlass.cute.nvgpu.tcgen05.make_s2t_copy)
* [`get_s2t_smem_desc_tensor()`](#cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor)
* [`make_umma_smem_desc()`](#cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.