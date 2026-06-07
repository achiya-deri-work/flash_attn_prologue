

Overview — NVIDIA CUTLASS Documentation



[Skip to main content](#main-content)

Back to top




`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Search
`Ctrl`+`K`

Search
`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Table of Contents

* [Changelog](../../../CHANGELOG.html)

CuTe DSL

* [Overview](#)
* [Functionality](functionality.html)
* [Quick Start Guide](quick_start.html)
* [CuTe DSL](cute_dsl.html)
  + [Introduction](cute_dsl_general/dsl_introduction.html)
  + [Code Generation](cute_dsl_general/dsl_code_generation.html)
  + [Control Flow](cute_dsl_general/dsl_control_flow.html)
  + [JIT Argument Generation](cute_dsl_general/dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](cute_dsl_general/dsl_dynamic_layout.html)
  + [JIT Caching](cute_dsl_general/dsl_jit_caching.html)
  + [JIT Compilation Options](cute_dsl_general/dsl_jit_compilation_options.html)
  + [JIT Types](cute_dsl_general/types.html)
  + [Integration with Frameworks](cute_dsl_general/framework_integration.html)
  + [Debugging with the DSL](cute_dsl_general/debugging.html)
  + [Autotuning with the DSL](cute_dsl_general/autotuning_gemm.html)
  + [Educational Notebooks](cute_dsl_general/notebooks.html)
  + [Deprecation Policy](deprecation.html)
  + [Compile with TVM FFI](cute_dsl_general/compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](cute_dsl_general/dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](cute_dsl_general/resources.html)
* [CuTe DSL API](cute_dsl_api.html)
  + [changelog](cute_dsl_api/changelog.html)
  + [cute](cute_dsl_api/cute.html)
    - [arch](cute_dsl_api/cute_arch.html)
    - [Runtime](cute_dsl_api/cute_runtime.html)
  + [cute\_nvgpu](cute_dsl_api/cute_nvgpu.html)
    - [Common](cute_dsl_api/cute_nvgpu_common.html)
    - [warp submodule](cute_dsl_api/cute_nvgpu_warp.html)
    - [warpgroup submodule](cute_dsl_api/cute_nvgpu_warpgroup.html)
    - [cpasync submodule](cute_dsl_api/cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](cute_dsl_api/cute_nvgpu_tcgen05.html)
  + [pipeline](cute_dsl_api/pipeline.html)
  + [utils](cute_dsl_api/utils.html)
    - [Utilities for SM90](cute_dsl_api/utils_sm90.html)
    - [Utilities for SM100](cute_dsl_api/utils_sm100.html)
* [Limitations](limitations.html)
* [FAQs](faqs.html)

CUTLASS C++

* [Overview](../../../overview.html)
* [Getting Started](../cpp/getting_started.html)
  + [Quickstart](../cpp/quickstart.html)
  + [IDE Setup](../cpp/ide_setup.html)
  + [Build](../cpp/build/index.html)
    - [Building on Windows with Visual Studio](../cpp/build/building_in_windows_with_visual_studio.html)
    - [Building with Clang as host compiler](../cpp/build/building_with_clang_as_host_compiler.html)
  + [Functionality](../cpp/functionality.html)
  + [Terminology](../cpp/terminology.html)
  + [Fundamental Types](../cpp/fundamental_types.html)
  + [Programming Guidelines](../cpp/programming_guidelines.html)
  + [GEMM Heuristics](../cpp/heuristics.html)
* [Efficient GEMM in CUDA](../cpp/efficient_gemm.html)
* [Synchronization primitives](../cpp/pipeline.html)
* [CUTLASS Profiler](../cpp/profiler.html)
* [Dependent Kernel Launch](../cpp/dependent_kernel_launch.html)
* [Blackwell Specific](../cpp/blackwell.html)
  + [Blackwell SM100 GEMMs](../cpp/blackwell_functionality.html)
  + [Blackwell Cluster Launch Control](../cpp/blackwell_cluster_launch_control.html)
* [CuTe](../cpp/cute/index.html)
  + [00\_quickstart](../cpp/cute/00_quickstart.html)
  + [01\_layout](../cpp/cute/01_layout.html)
  + [02\_layout\_algebra](../cpp/cute/02_layout_algebra.html)
  + [03\_tensor](../cpp/cute/03_tensor.html)
  + [04\_algorithms](../cpp/cute/04_algorithms.html)
  + [0t\_mma\_atom](../cpp/cute/0t_mma_atom.html)
  + [0x\_gemm\_tutorial](../cpp/cute/0x_gemm_tutorial.html)
  + [0y\_predication](../cpp/cute/0y_predication.html)
  + [0z\_tma\_tensors](../cpp/cute/0z_tma_tensors.html)
* [CUTLASS 3.x](../cpp/cutlass_3x.html)
  + [Design](../cpp/cutlass_3x_design.html)
  + [GEMM Backwards Compatibility](../cpp/cutlass_3x_backwards_compatibility.html)
  + [GEMM API](../cpp/gemm_api_3x.html)
* [CUTLASS 2.x](../cpp/cutlass_2x.html)
  + [Layouts and Tensors](../cpp/layout.html)
  + [GEMM API](../cpp/gemm_api.html)
  + [Tile Iterator Concepts](../cpp/tile_iterator_concept.html)
  + [Utilities](../cpp/utilities.html)
* [Code Organization](../cpp/code_organization.html)
* [Grouped Kernel Schedulers](../cpp/grouped_scheduler.html)
* [CUTLASS Convolution](../cpp/implicit_gemm_convolution.html)

Reference

* [Software License Agreement](license.html)

* Overview

# Overview[#](#overview "Link to this heading")

CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development.
By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables
faster iteration, easier prototyping, and a gentler learning curve for high-performance linear
algebra on NVIDIA GPUs.

Overall we envision CUTLASS DSLs as a family of domain-specific languages (DSLs).
With the release of 4.0, we are releasing the first of these in CuTe DSL.
This is a low level programming model that is fully consistent with CuTe C++ abstractions â exposing
core concepts such as layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.

# Why CUTLASS DSLs?[#](#why-cutlass-dsls "Link to this heading")

While CUTLASS offers exceptional performance through its C++ template abstractions, the complexity
can present challenges for many developers. CUTLASS 4.x addresses this by:

* **Simplifying metaprogramming**: Metaprogramming in Python is a lot more intuitive than with C++
* **Accelerating Iteration**: Rapid prototyping with familiar Python syntax and blazing fast compile times
* **Lowering Barriers**: Reduced learning curve for GPU programming concepts and consistency between CuTe C++ and DSL
* **Maintaining Performance**: Generated code leverages optimized CUTLASS primitives

Students can learn GPU programming concepts without the complexity of C++ templates.
Researchers and performance engineers can rapidly explore algorithms, prototype, and tune
kernels before moving to production implementations.

# Key Concepts and Approach[#](#key-concepts-and-approach "Link to this heading")

CUTLASS DSLs translate Python code into a custom intermediate representation (IR),
which is then Just-In-Time (JIT) compiled into optimized CUDA kernels using MLIR and ptxas.

## Core CuTe DSL Abstractions[#](#core-cute-dsl-abstractions "Link to this heading")

* **Layouts** â Describe how data is organized in memory and across threads.
* **Tensors** â Combine data pointers or iterators with layout metadata.
* **Atoms** â Represent fundamental hardware operations like matrix multiply-accumulate (MMA) or memory copy.
* **Tiled Operations** â Define how atoms are applied across thread blocks and warps (e.g., `TiledMma`, `TiledCopy`).

For more on CuTe abstractions, refer to the [CuTe C++ library documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md).

**Pythonic Kernel Expression**

Developers express kernel logic, data movement, and computation using familiar Python syntax and control flow.

The DSLs simplify expressing loop tiling, threading strategies, and data transformations using concise Python code.

**JIT Compilation**

Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIAâs `ptxas` toolchain,
enabling rapid iteration and interactive debugging.

# Relationship to CUTLASS C++[#](#relationship-to-cutlass-c "Link to this heading")

CUTLASS DSLs are not a replacement for the CUTLASS C++ library or its 2.x and 3.x APIs. Instead, it aims to be a high-productivity kernel
authoring framework that shares all concepts with CUTLASS 3.x C++ API such as CuTe, pipelines, schedulers etc.

* **Performance**: Generated kernels aim to match CUTLASS C++ kernels in performance; however, some performance gaps
  may exist due to missing optimizations that have been added over the years to CUTLASS C++ and may be missing in the DSLs examples.
* **Library**: The CUTLASS DSLs do not currently ship with a full GEMM/Conv autotuning profiler or library interface
  akin to CUTLASS C++. Instead, it focuses on generating and autotuning individual kernel instances (for example: via tile size exploration) and via native integration DL frameworks that support auto-tuning.

# Getting Started[#](#getting-started "Link to this heading")

* [Quick Start Guide](quick_start.html) â Initial setup and installation.
* [CuTe DSL](cute_dsl.html) â Overview of the typical development and workflow using CuTe DSL.
* [CuTe DSL API](cute_dsl_api.html) â Refer to the full API documentation.
* [Limitations](limitations.html) â Understand current CuTe DSL constraints and differences from C++.
* [FAQs](faqs.html) â Common questions and known issues.

# Current Status & Roadmap[#](#current-status-roadmap "Link to this heading")

CuTe DSL is in public beta and actively evolving. Interfaces and features are subject to
change as we improve the system.

## Upcoming Milestones[#](#upcoming-milestones "Link to this heading")

* Public release targeted for **Summer 2025**
* Expanded support for additional data types and kernel types
* Usability improvements: better error messages, debugging tools, and streamlined APIs
* Broader integration of CUTLASS primitives and features

For known issues and workarounds, please consult the [Limitations](limitations.html) and [FAQs](faqs.html).

# Community & Feedback[#](#community-feedback "Link to this heading")

We welcome contributions and feedback from the developer community!

You can:

* Submit bug reports or feature requests via our [GitHub Issues page](https://github.com/NVIDIA/cutlass/issues)
* Join the CUTLASS community on [Discord](https://discord.com/channels/1019361803752456192/1150868614921064590) to ask questions and share ideas
* Contribute examples, tutorials, or enhancements to the DSLs
* Report unclear or missing documentation
* Propose support for additional data types or kernel variants
* Help prioritize roadmap features by upvoting GitHub issues

Thank you for helping shape the future of CUTLASS DSLs!

[previous

Changelog](../../../CHANGELOG.html "previous page")
[next

Functionality](functionality.html "next page")

On this page

* [Overview](#)
* [Why CUTLASS DSLs?](#why-cutlass-dsls)
* [Key Concepts and Approach](#key-concepts-and-approach)
  + [Core CuTe DSL Abstractions](#core-cute-dsl-abstractions)
* [Relationship to CUTLASS C++](#relationship-to-cutlass-c)
* [Getting Started](#getting-started)
* [Current Status & Roadmap](#current-status-roadmap)
  + [Upcoming Milestones](#upcoming-milestones)
* [Community & Feedback](#community-feedback)





[![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.