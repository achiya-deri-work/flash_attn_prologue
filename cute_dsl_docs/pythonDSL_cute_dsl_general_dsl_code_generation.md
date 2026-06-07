

End-to-End Code Generation — NVIDIA CUTLASS Documentation



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
  + [Introduction](dsl_introduction.html)
  + [Code Generation](#)
  + [Control Flow](dsl_control_flow.html)
  + [JIT Argument Generation](dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](dsl_dynamic_layout.html)
  + [JIT Caching](dsl_jit_caching.html)
  + [JIT Compilation Options](dsl_jit_compilation_options.html)
  + [JIT Types](types.html)
  + [Integration with Frameworks](framework_integration.html)
  + [Debugging with the DSL](debugging.html)
  + [Autotuning with the DSL](autotuning_gemm.html)
  + [Educational Notebooks](notebooks.html)
  + [Deprecation Policy](../deprecation.html)
  + [Compile with TVM FFI](compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](resources.html)
* [CuTe DSL API](../cute_dsl_api.html)
  + [changelog](../cute_dsl_api/changelog.html)
  + [cute](../cute_dsl_api/cute.html)
    - [arch](../cute_dsl_api/cute_arch.html)
    - [Runtime](../cute_dsl_api/cute_runtime.html)
  + [cute\_nvgpu](../cute_dsl_api/cute_nvgpu.html)
    - [Common](../cute_dsl_api/cute_nvgpu_common.html)
    - [warp submodule](../cute_dsl_api/cute_nvgpu_warp.html)
    - [warpgroup submodule](../cute_dsl_api/cute_nvgpu_warpgroup.html)
    - [cpasync submodule](../cute_dsl_api/cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](../cute_dsl_api/cute_nvgpu_tcgen05.html)
  + [pipeline](../cute_dsl_api/pipeline.html)
  + [utils](../cute_dsl_api/utils.html)
    - [Utilities for SM90](../cute_dsl_api/utils_sm90.html)
    - [Utilities for SM100](../cute_dsl_api/utils_sm100.html)
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

* [CuTe DSL](../cute_dsl.html)
* End-to-End Code Generation

# End-to-End Code Generation[#](#end-to-end-code-generation "Link to this heading")

## 1. Hybrid DSL: Python Metaprogramming, Structured GPU Code[#](#hybrid-dsl-python-metaprogramming-structured-gpu-code "Link to this heading")

CuTe DSL is a **hybrid DSL** that combines two compilation techniques: *AST rewrite*
and *tracing*. This combination gives you the best of both worlds:

* **Program structure is preserved** â control flow (loops, branches) is
  captured via AST rewrite, compiling to proper structured code instead of
  flattened traces.
* **Python stays Python** â arithmetic and tensor operations are captured via
  tracing, so dynamic shapes, metaprogramming, and Pythonâs rich expression
  language work naturally.

To understand why this matters, letâs look at each technique.

### 1.1 AST Rewrite[#](#ast-rewrite "Link to this heading")

The functionâs abstract-syntax tree is analysed **before** execution.
Python control-flow (`for`/`while`, `if`/`else`) and built-ins are converted to structured intermediate representation (IR)
constructs. Computation inside each region is left untouched at this stage.

*Advantages*

* Sees the entire program, so every branch and loop is preserved.
* Keeps loop structure intact for optimization such as tiling, vectorisation
  or GPU thread mapping.

*Disadvantages*

* Requires a well-defined Python subset that the rewriter understands.

### 1.2 Tracing[#](#tracing "Link to this heading")

The decorated function is executed once with *proxy* arguments; overloaded
operators record every tensor operation that actually runs and produce a flat
trace that is lowered to intermediate representation (IR).

*Advantages*

* Near-zero compile latency, ideal for straight-line arithmetic.
* No need to parse Python source, so it supports many dynamic Python
  features, and Python has many features.

*Disadvantages*

* Untaken branches vanish, so the generated kernel may be wrong for other
  inputs.
* Loops are flattened to the iteration count observed during tracing.
* Data-dependent control-flow freezes to a single execution path.

### 1.3 The Hybrid Solution[#](#the-hybrid-solution "Link to this heading")

As shown above, neither technique alone is sufficientâbut together they
complement each other perfectly.

**Why this works: GPU kernels are simple at runtime**

High-performance GPU kernels are structurally simple at runtime: they avoid
deep call hierarchies, complex branching, and dynamic dispatch. However,
*authoring* such kernels benefits greatly from Pythonâs abstractionsâclasses,
metaprogramming, and polymorphic patterns improve readability and
maintainability.

The hybrid approach resolves this tension by evaluating Python abstractions at
compile time while emitting simple, optimized code for runtime execution.

**How |DSL| divides the work:**

1. **AST rewrite handles structure** â loops (`for`, `while`) and branches
   (`if`/`else`) are converted to structured intermediate representation (IR) *before* execution.
   This solves tracingâs control-flow problem.
2. **Tracing handles arithmetic** â inside each structured region, the tracer
   records tensor operations exactly as they execute. No need to model Pythonâs
   complex semanticsâjust run Python and record what happens. This solves AST
   rewritingâs complexity problem.

The result:

* Loops compile to real loops, not unrolled traces.
* All branches are preserved, even if not taken during tracing.
* Dynamic shapes, metaprogramming, and Python idioms work naturally.
* The rewriter only needs to understand control flow, not all of Python.

## 2. CuTe DSL Compilation Flow: Meta-Stage to Object-Stage[#](#dsl-compilation-flow-meta-stage-to-object-stage "Link to this heading")

CuTe DSL bridges Python and GPU hardware through a three-stage pipeline.

[![../../../../_images/dsl_modes.png](../../../../_images/dsl_modes.png)](../../../../_images/dsl_modes.png)


Figure 1 *Left*: tracing mode records only the path that executed.
*Right*: preprocessor mode emits structured intermediate representation (IR) for every branch and loop
before tracing the arithmetic.[#](#id1 "Link to this image")

The default CuTe DSL compilation pipeline (mode 2): Python source flows through AST preprocessing
and interpreter-driven tracing to produce intermediate representation (IR), which is then lowered and
compiled to device code.

**Stage 1: Pre-Staging (Python AST)**

Before any code executes, the AST preprocessor rewrites the decorated function.
It inserts *callbacks* around control-flow constructsâloops, branches, and
function boundariesâso that program structure is captured explicitly rather than
lost during execution.

**Stage 2: Meta-Stage (Python Interpreter)**

The rewritten function runs in the Python interpreter with proxy tensor
arguments. As execution proceeds:

* Callbacks fire at control-flow boundaries, emitting structured intermediate representation (IR) (loops,
  branches, etc.).
* Tensor operations are traced: each operator invocation records the
  corresponding operation.
* Compile-time constants are *partially evaluated*âvalues known at JIT time
  fold directly into the intermediate representation (IR), enabling aggressive specialization.

The result is a complete representation of the kernel, with both high-level
structure and low-level arithmetic intact.

**Stage 3: Object-Stage (Compiler Backend)**

The internal representation passes through a lowering pipeline:

1. High-level operations are progressively lowered toward hardware-specific
   representations.
2. Optimization passes (tiling, vectorization, memory promotion) reshape the
   code for the target architecture.
3. The final code is translated to PTX/SASS (for NVIDIA GPUs) and assembled
   into a device binary.

At runtime, the compiled kernel is loaded and launched on the accelerator.

## 3. Meta-Programming vs Runtime: Two Worlds in One Function[#](#meta-programming-vs-runtime-two-worlds-in-one-function "Link to this heading")

A key insight for understanding CuTe DSL is that **your Python code runs twice**,
in two very different contexts:

1. **Meta-programming time (compilation)** â Python executes to *build* the
   kernel. This happens on the host CPU when you call a `@jit` function.
2. **Runtime (execution)** â The compiled kernel runs on the GPU with actual
   tensor data.

This distinction determines what you can observe and when.

### `print()` vs `cute.printf()`: Meta-Stage vs Object-Stage Output[#](#print-vs-cute-printf-meta-stage-vs-object-stage-output "Link to this heading")

CuTe DSL provides two ways to print values, each operating at a different stage:

* **Pythonâs** `print()` â executes during the **meta-stage** (compilation).
  Use it to inspect what the compiler sees.
* `cute.printf()` â compiles into the kernel and executes at **runtime** on
  the GPU. Use it to observe actual tensor values during execution.

The following examples demonstrate how the same `result` variable appears
differently depending on when and how you print it.

**Example 1: Dynamic variables (both** `a` **and** `b` **are runtime values)**

```python
@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_dynamicexpr(5.0)
```

```python
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

At meta-stage, `result` is a proxyâits value is unknown until the kernel runs.
At runtime, `cute.printf()` prints the actual GPU-computed value.

**Example 2: Compile-time constants (both** `a` **and** `b` **are Constexpr)**

```python
@cute.jit
def add_constexpr(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_constexpr(5.0)
```

```python
$> python myprogram.py
[meta-stage] result = 7.0
[object-stage] result = 7.000000
```

Both values are known at compile time, so Python evaluates `2.0 + 5.0 = 7.0`
during tracing. The constant is baked into the compiled kernel.

**Example 3: Hybrid (** `a` **is dynamic,** `b` **is Constexpr)**

```python
@cute.jit
def add_hybrid(b: cutlass.Constexpr):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_hybrid(5.0)
```

```python
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

The constant `b = 5.0` is folded in, but since `a` is dynamic, the result
remains a proxy at meta-stage. The GPU computes the final answer at runtime.

### Practical Implications[#](#practical-implications "Link to this heading")

* **Use** `print()` **to debug your meta-program** â inspect shapes, strides,
  tile sizes, and compile-time decisions.
* **Constexpr parameters enable specialization** â the compiler can generate
  tighter code when values are known at JIT time.
* **Dynamic parameters preserve generality** â a single compiled kernel can
  handle varying input sizes without recompilation.

## 4. CuTe DSL Code-Generation Modes[#](#dsl-code-generation-modes "Link to this heading")

CuTeâs Python front-end combines the techniques above into **two mutually
exclusive modes** (see [Left: tracing mode records only the path that executed.
Right: preprocessor mode emits structured intermediate representation (IR) for every branch and loop
before tracing the arithmetic.](#fig-dsl-modes)), selectable with the `preprocessor` flag of the
`@jit` decorator:

1. Tracing mode `@jit(preprocess=False)` â tracing only.
This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be
straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.

2. Preprocessor mode (**default**) `@jit(preprocess=True)` â **AST rewrite + tracing**.
The AST pass captures every loop and branch, eliminating the correctness and
optimisation problems of pure tracing; tracing then fills in the arithmetic.
This hybrid âpreprocessorâ pipeline is unique to CuTe DSL and was designed
specifically to overcome the disadvantages identified above.

[previous

Introduction](dsl_introduction.html "previous page")
[next

Control Flow](dsl_control_flow.html "next page")

On this page

* [1. Hybrid DSL: Python Metaprogramming, Structured GPU Code](#hybrid-dsl-python-metaprogramming-structured-gpu-code)
  + [1.1 AST Rewrite](#ast-rewrite)
  + [1.2 Tracing](#tracing)
  + [1.3 The Hybrid Solution](#the-hybrid-solution)
* [2. CuTe DSL Compilation Flow: Meta-Stage to Object-Stage](#dsl-compilation-flow-meta-stage-to-object-stage)
* [3. Meta-Programming vs Runtime: Two Worlds in One Function](#meta-programming-vs-runtime-two-worlds-in-one-function)
  + [`print()` vs `cute.printf()`: Meta-Stage vs Object-Stage Output](#print-vs-cute-printf-meta-stage-vs-object-stage-output)
  + [Practical Implications](#practical-implications)
* [4. CuTe DSL Code-Generation Modes](#dsl-code-generation-modes)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.