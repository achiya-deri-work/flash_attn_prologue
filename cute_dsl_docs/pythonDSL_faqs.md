

FAQs — NVIDIA CUTLASS Documentation



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

* [Overview](overview.html)
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
* [FAQs](#)

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

* FAQs

# FAQs[#](#faqs "Link to this heading")

## General[#](#general "Link to this heading")

**Are the DSLs replacing C++ templates?**

> TL;DR: No - but also yes. The CUTLASS 4.0 release (CuTe DSL), along with all
> future extensions to our Python-native programming models, does not come at the
> expense of CUTLASS C++. CUTLASS 2.x and 3.x C++ APIs are both going to continue
> receiving fixes and updates for the architectures we support them for. However,
> CUTLASS 4.x CuTe DSL is fully isomorphic in its programming model and performance
> with CuTe C++ for Blackwell, and it is our hope that the community embraces this
> for much easier while still equally performant custom kernel development. This is
> why we are releasing CuTe DSL with support for all architectures starting with the
> NVIDIA Ampere Architecture.

**What is the difference between CuTe DSL, CUTLASS Python, and CUTLASS DSLs?**

> CUTLASS Python was the Python interface for instantiating C++ kernels via a Python
> frontend. This is now deprecated with the release of CUTLASS 4.0. CUTLASS DSLs are
> a family of Python DSLs for native device programming in Python. Currently, this is
> limited to our initial release of CuTe DSL, but future versions will include higher-level
> abstractions that gradually trade off control for convenience.

**What should I learn, CUTLASS C++ or the Python DSLs?**

> We believe the Python DSLs will significantly improve the learning curve and recommend starting
> with them for all newcomers, as they eliminate the inherent complexity of learning C++
> metaprogramming for GPU kernel programming. Since CuTe C++ and CuTe DSL share fully isomorphic
> programming models and patterns, any knowledge gained can eventually be applied to C++.

**Where will the code live? PIP wheel or GitHub repo? Do I have to build it myself?**

> This is a major change compared to CUTLASS C++ and Python DSLs. Going forward,
> the GitHub code only exists as a way for users to file issues and pull requests against.
> While it can be used with the pip wheel, we do not recommend most users do so unless they are
> hacking on the DSL itself. For all other users, we recommend they
> simply `pip install nvidia-cutlass-dsl` and use the pip wheel as the single source
> of truth for the dialect compiler and DSL implementation. CUTLASS GitHub repository will
> contain a `requirements.txt` file pinning the version of the wheel consistent with the state
> of the OSS repository (please see [Quick Start Guide](quick_start.html)). This means getting started with
> CUTLASS is easier than ever: no more CMake command lines to learn and no more builds to kick
> off. Simply install the pip wheel and start running the examples.

## Migration[#](#migration "Link to this heading")

**Should I port my code from C++ templates to Python?**

> Almost certainly not, unless you need extremely fast JIT times for your kernel and C++ compile times
> are a blocker for you. The 2.x and 3.x APIs will continue to be supported, and Nvidiaâs Hopper and
> Blackwell architectures 3.x will continue to improve in terms of features
> and performance.

**Are portability promises different with Python?**

> For the initial release while the DSL is still in beta, we do not promise any portability
> as we may make changes to the DSL itself. While we do not expect any changes to the CuTe operations,
> the DSL utilities, decorators, helper classes like pipelines and schedulers may change as we refine them
> with community feedback. We encourage users to file issues and discussions on GitHub during this
> beta period with their feedback!
>
> In the long term, we plan to continue to treat the OSS community with care.
> Just like the prior history of CUTLASS, we plan not to break users unless necessary,
> but we reserve the right to make limited breaking changes in case we believe it is a
> net benefit to the community and project. These will be announced ahead of time and/or
> clearly highlighted in the CHANGELOG of each release.

## Technical[#](#technical "Link to this heading")

**What NVIDIA architectures will it support?**

> CuTe DSL will support all NVIDIA GPU architectures starting with NVIDIA Ampere Architecture (SM80).

**Will it be compatible with DL frameworks (e.g., PyTorch, JAX)?**

> Yes, we will provide utilities to convert from DLPack-supported tensor formats
> to `cute.Tensor`. This should allow a user to never have to leave Python
> when writing model code in their framework of choice. Our JAX interoperability story is not
> as strong as PyTorchâs today, however, we are actively working on improving it
> and welcome contributions in this space.

**Does it compile to PTX or SASS?**

> CuTe DSL compiles the program down to PTX. After that, we currently use the PTX compiler that
> ships with the CUDA toolkit to compile the PTX down to SASS. We plan to remove
> this limitation in the future and allow the use of the PTX JIT that is included in the
> CUDA driver in case a user does not have a CUDA toolkit installed.

**Do I need to use NVCC or NVRTC?**

> No, the `nvidia-cutlass-dsl` wheel packages is everything needed to generate GPU kernels. It
> shares the driver requirements of the 12.9 toolkit which can be found
> [here](https://developer.nvidia.com/cuda-toolkit-archive).

**How would one debug the code?**

> Since CuTe DSL is not native python and an embedded DSL instead, tools like pdb
> cannot be used. However, if you have experience with GPU kernel programming, the debugging
> techniques will be nearly identical. Typically, compile time and runtime printing
> of types and values are the most expedient. Please see [documentation on printing](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb)
> to learn how to print types and values at both compile time and runtime.
> You can also use `cuda-gdb` to set breakpoints in the program and step through the execution
> or use tools such as `compute-sanitizer` to detect and triage bugs in your program. As the DSL
> matures, our source location tracking from Python user programs will also improve to provide
> more helpful source-level mapping when setting breakpoints and using other tools such as nsight.

**How would one implement warp specialization in CuTe DSL?**

> Exactly the same way you would in C++ but in a Python-native syntax instead.
> Consult our [Control Flow](cute_dsl_general/dsl_control_flow.html) and
> [âBlackwell kernel exampleâ](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)
> for a detailed how-to guide.

**Can I call functions from other functions or use OOP?**

> Yes. We frequently call functions from one another and set up class
> hierarchies to organize and modularize our code for pipelines and schedulers.
> Consult the [Introduction](cute_dsl_general/dsl_introduction.html) documentation or our examples for more details.

## License[#](#license "Link to this heading")

**What is the license for CuTe DSL and the associated GitHub samples?**

> CuTe DSL components available [on Github](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) and via the nvidia-cutlass-dsl Python pip wheel
> are released under the [âNVIDIA Software End User License Agreement (EULA)â](https://github.com/NVIDIA/cutlass/tree/main/EULA.txt).
> Because the pip package includes a compiler that shares several components with the CUDA Toolkit,
> it is subject to usage terms and restrictions similar to those of the CUDA SDK. Please refer to the EULA for specific terms of use.
>
> CuTe DSL samples and Jupyter notbooks, released [on GitHub](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) are provided under
> the BSD 3-Clause License and may be used and redistributed under those terms. This distinction ensures that developers have flexibility
> when using or modifying the code samples, independent of the compiler and runtime components governed by the EULA.
>
> If you have any questions or need clarification, feel free to contact us.

[previous

Limitations](limitations.html "previous page")
[next

Overview](../../../overview.html "next page")

On this page

* [General](#general)
* [Migration](#migration)
* [Technical](#technical)
* [License](#license)





[![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.