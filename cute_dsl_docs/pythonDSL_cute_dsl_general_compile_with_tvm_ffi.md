

Compile with TVM FFI — NVIDIA CUTLASS Documentation



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
  + [Code Generation](dsl_code_generation.html)
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
  + [Compile with TVM FFI](#)
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
* Compile with TVM FFI

# Compile with TVM FFI[#](#id1 "Link to this heading")

Apache TVM FFI is an open ABI and FFI for machine learning systems. More information can be found in
the [official documentation](https://tvm.apache.org/ffi/).

To install TVM FFI, you can run the following command:

```python
pip install apache-tvm-ffi
# optional package for improved torch tensor calling performance
pip install torch-c-dlpack-ext
```

In CuTe DSL, TVM FFI can be enabled as an option for JIT-compiled functions. Using TVM FFI can lead to faster
JIT function invocation and provides better interoperability with machine learning frameworks
(e.g., directly take `torch.Tensor` as arguments).

## Enable Apache TVM FFI in CuTe DSL[#](#enable-apache-tvm-ffi-in-dsl "Link to this heading")

First, install the `tvm-ffi` package by following its [installation guide](https://tvm.apache.org/ffi/#installation).

There are two ways to enable TVM FFI in CuTe DSL:

1. Use the `options` argument in `cute.compile` to specify the TVM FFI option. For example:

```python
# Assuming you have defined a function `add` decorated with @cute.jit
def example_compile():
   a_torch = torch.randn(10, 20, 30).to(torch.float16)
   b_torch = torch.randn(10, 20, 30).to(torch.float16)
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add = cute.compile(add, a_torch, b_torch, options="--enable-tvm-ffi")
```

Note that the object returned by `cute.compile` is a Python function specific to TVM FFI.

2. Alternatively, you can enable TVM FFI globally by setting the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. Please note that this setting will apply to all JIT compilations within the environment.

## Minimizing Host Overhead[#](#minimizing-host-overhead "Link to this heading")

Eager kernel invocation overhead on the CPU host can sometimes become a bottleneck
for latency-sensitive applications. TVM FFI can help greatly reduce this overhead.
To maximize performance benefits, we recommend setting up your workflow as follows
(detailed instructions are provided in subsequent sections):

* **Compile the kernel with TVM FFI enabled.**
* **Declare shape constraints using fake tensors** and reuse the compiled function
  throughout your execution.
* **Pass PyTorch tensors directly** to the compiled function to avoid explicit DLPack conversion.
* **Use the environment stream flag** to implicitly pass the current PyTorch stream.
* **Rely on compiled argument validation** instead of Python-side attribute validation,
  as TVM FFI functions perform fast compiled checks.

Following these steps can significantly reduce the host-side overhead of eager kernel execution.
The sections below provide detailed examples and explanations for each step.
You may find it helpful to refer back to this summary after you review the implementation details.

## Fake tensor for compilation[#](#fake-tensor-for-compilation "Link to this heading")

The TVM FFI function accepts DLPack-compatible tensors as arguments, such as those from torch or jax.
However, during compilation, it is necessary to specify the tensorsâ dynamic properties in CuTe DSL.
To clearly distinguish between the compilation phase and runtime,
CuTe DSL provides a âfake tensorâ that can be used for compilation. For example:

```python
import cutlass.cute as cute
import torch

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
   )

def example_add_one():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # now compiled_add_one is a TVM-FFI function that can be called with torch.Tensor as input
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(a_torch, b_torch)
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

The fake tensor is a placeholder that mimics the interface of a real tensor but does not hold real data or allow indexing.
It is used in compilation or testing scenarios where only shape/type/layout information is needed.
All attempts to access or mutate data will raise errors.

### Note on Stride Order[#](#note-on-stride-order "Link to this heading")

Note that CuTeâs convention is to write the stride order for dimensions from left to right,
where a lower order number means higher priority. In the context of the `make_fake_compact_tensor` API,
for shape `(2, 3, 4)` and stride order `(0, 1, 2)`, the stride is `(1, 2, 6)`.
This is commonly known as column-major order. If you want to create a fake tensor with compact row-major order,
you should explicitly pass in `stride_order=tuple(reversed(range(len(shape))))`
to `make_fake_compact_tensor`. Alternatively, you can always precisely control the
stride via the `stride` argument in the `make_fake_tensor` API.

## `cute.Tensor` adapter for TVM FFI[#](#cute-tensor-adapter-for-tvm-ffi "Link to this heading")

To adapt the `cute.Tensor` to the TVM FFI function, you can use the `cute.runtime.from_dlpack` function with the
`enable_tvm_ffi=True` option or the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. For example:

```python
def example_from_dlpack():
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add_one(a_cute, b_cute)
```

Note that because the `cute.runtime.from_dlpack` function performs an explicit DLPack conversion, it is less efficient than passing the `torch.Tensor` directly.
You can also use `cute.Tensor` as an argument hint for `cute.compile`.

```python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```

## Working with torch Tensors[#](#working-with-torch-tensors "Link to this heading")

As you may have noticed in the examples above, TVM FFI-compiled functions can
directly accept `torch.Tensor` objects (and other DLPack-compatible tensors) as inputs.
The resulting functions add minimal overhead, enabling faster eager invocations
thanks to the optimized calling path.

## Working with Streams[#](#working-with-streams "Link to this heading")

In many cases, a CuTe kernel needs to run on a specific CUDA stream.
CuTe DSL provides two ways to work with streams through TVM FFI.
The first is to pass the stream explicitly as an argument.
The following example demonstrates this approach; the function accepts `torch.cuda.Stream`,
`CUstream` or any stream class that implements the CUDA stream protocol.

```python
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one_with_stream(a: cute.Tensor, b: cute.Tensor, stream: CUstream):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
      stream=stream,
   )

def example_add_one_with_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   stream = cute.runtime.make_fake_stream()
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   compiled_add_one(a_torch, b_torch, torch_stream)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch, torch_stream)")
   print(b_torch)
```

### Using Environment Stream[#](#using-environment-stream "Link to this heading")

The second option is to rely on the environment stream flag.
Pass `use_tvm_ffi_env_stream=True` to `make_fake_stream` to mark the stream argument as an
environment stream, which means it no longer needs to be provided explicitly.
TVM FFI will automatically use its environment stream (i.e., the current PyTorch stream)
as the stream argument. The example below demonstrates this flow:

```python
def example_add_one_with_env_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   # we will use TVM FFI environment stream
   stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   with torch.cuda.stream(torch_stream):
      # no need to pass in the stream explicitly, env stream will be synced
      # to torch.cuda.current_stream() before the function call.
      compiled_add_one(a_torch, b_torch)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

Using the environment stream flag both speeds up calls and simplifies integration
with frameworks such as PyTorch, since no explicit stream parameter is required.
We recommend using the environment stream flag to both simplify framework integration
and minimize host-side calling overhead.

## Working with Tuples[#](#working-with-tuples "Link to this heading")

TVM FFI functions can also accept tuples as arguments. Tuples can be recursively
composed of the types that are supported by TVM FFI. The example below shows how to use tuples as arguments:

```python
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, c: cute.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_tuple(a: Tuple[cute.Tensor, cute.Tensor, cute.Float32]):
   n = a[0].shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a[0], a[1], a[2]).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one = cute.compile(
      add_one_with_tuple, (a_cute, b_cute, cute.Float32(4)),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one((a_torch, b_torch, 5))
   print("result of b_torch after compiled_add_one((a_torch, b_torch, 5))")
   print(b_torch)

example_add_one_with_tuple()
```

## Working with Variadic Tuples[#](#working-with-variadic-tuples "Link to this heading")

Sometimes it is helpful to annotate a tuple with no explicit element types.
This can be useful to build up a generic template for a function that accepts
a variable number of elements. The compiled functionâs signature will be
determined by the tuple argument passed to the `cute.compile` function.
The following example shows how to use a variadic tuple to build such a
generic template.

```python
import cutlass
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      if cutlass.const_expr(len(extra_value) != 0):
            b[tid] = a[tid] + 1 + extra_value[0]
      else:
            b[tid] = a[tid] + 1

@cute.jit
def add_one_with_extra_value(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b, extra_value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_variadic_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one_no_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (),
      options="--enable-tvm-ffi"
   )
   compiled_add_one_with_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (cute.Float32(4),),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one_no_extra(a_torch, b_torch, ())
   print("result of b_torch after compiled_add_one_no_extra(a_torch, b_torch, ())")
   print(b_torch)
   compiled_add_one_with_extra(a_torch, b_torch, (4,))
   print("result of b_torch after compiled_add_one_with_extra(a_torch, b_torch, (4,))")
   print(b_torch)

example_add_one_with_variadic_tuple()
```

### Working with Named Tuples[#](#working-with-named-tuples "Link to this heading")

Named tuples are also supported and help logically group related arguments together.
The example below shows how to use named tuples as arguments. Under the hood, named tuples
are passed as unnamed tuples at the ABI level. When errors occur, the function signature in
error messages will display unnamed tuple arguments.
Ensure that the compile-time CuTe named tuple type definition has the same fields
as the runtime PyTorch named tuple.
Currently, users need to explicitly unpack the named tuple outside of conditionals and then
use the unpacked variables inside the conditionals.

```python
from typing import NamedTuple
from cutlass import cute
import torch

class CuteNamedTuple(NamedTuple):
   a: cute.Tensor
   b: cute.Tensor
   c: cute.Float32 = cute.Float32(1)

   def __new_from_mlir_values__(self, values):
      return CuteNamedTuple(*values)

class TorchNamedTuple(NamedTuple):
   a: torch.Tensor
   b: torch.Tensor
   c: float = 1

@cute.kernel
def device_add_one_named_tuple(value: CuteNamedTuple):
   tid = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
   # need to unpack namedtuple outside conditionals
   a = value.a
   b = value.b
   c = value.c
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_named_tuple(value: CuteNamedTuple):
   n = value.a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one_named_tuple(value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_named_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

   compiled_add_one = cute.compile(
      add_one_with_named_tuple, CuteNamedTuple(a=a_cute, b=b_cute),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(TorchNamedTuple(a=a_torch, b=b_torch))
   print("result of b_torch")
   print(b_torch)

example_add_one_with_named_tuple()
```

## Supported types[#](#supported-types "Link to this heading")

The TVM FFI function supports the following CuTe DSL-specific types as arguments:

* `cute.Tensor`
* `cutlass.Boolean`, `cutlass.Int8`, `cutlass.Int16`, `cutlass.Int32`, `cutlass.Int64`, `cutlass.Uint8`, `cutlass.Uint16`, `cutlass.Uint32`, `cutlass.Uint64`, `cutlass.Float32`, `cutlass.Float64`
* `cute.Shape`, `cute.Stride`, `cute.Coord`, `cute.Tile`, `cute.IntTuple`

| Compile-time type | Call-time type |
| --- | --- |
| `cute.Pointer` | `ctypes.c_void_p` or a class that implements `__tvm_ffi_opaque_ptr__` protocol. |
| `cute.runtime.FakeTensor` | `torch.Tensor` and other DLPack-compatible tensors. |
| Scalar types (e.g. `cutlass.Boolean`, `cutlass.Int32`) | Python scalars (e.g. True, 123). |
| CuTe algebra types (e.g. `cute.Shape`, `cute.Stride`) | `tvm_ffi.Shape` or python tuple of ints. |
| CUDA stream `cuda.CUstream` | A stream class that implements the CUDA stream protocol (e.g. `torch.cuda.Stream`, `cuda.CUstream`). |
| Tuple of types (e.g. `Tuple[cute.Tensor, cute.Tensor, cutlass.Int32]`) | Python tuple of corresponding call-time types. |

## Error handling[#](#error-handling "Link to this heading")

TVM FFI functions will enable validation of arguments to make sure they match the expected type
and value constraints declared by the user. These checks are compiled into the function, run very fast,
and have no observable overhead during function invocation. Each of those errors will translate
into a proper Python exception that can be caught and handled. The example below shows some
example error cases that can be checked:

```python
def example_constraint_checks():
   n = cute.sym_int(divisibility=16)
   # assume align to 16 bytes (4 int32), both should share same shape variable n
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   a = torch.zeros(128, dtype=torch.float32, device="cuda")
   b = torch.zeros(128, dtype=torch.float32, device="cuda")

   try:
      # raises type mismatch error because we expect a and b to be float32
      compiled_add_one(a, 1)
   except TypeError as e:
      # Mismatched type on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected Tensor
      print(f"TypeError: {e}")

   try:
      # raises shape mismatch error because we expect both a and b have shap [n]
      compiled_add_one(a, b[:126])
   except ValueError as e:
      # Mismatched b.shape[0] on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected to match a.shape[0]
      print(f"ValueError: {e}")

   try:
      # triggers divisibility mismatch error because 126 is not divisible by 16
      compiled_add_one(a[:126], b[:126])
   except ValueError as e:
      # Invalid a.shape[0] on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected to be divisible by 16
      print(f"ValueError: {e}")

   try:
      a = torch.zeros(129, dtype=torch.float32, device="cuda")
      b = torch.zeros(129, dtype=torch.float32, device="cuda")
      # triggers data alignment mismatch error because x and y are not aligned to 16 bytes
      compiled_add_one(a[1:], b[1:])
   except ValueError as e:
      # raises: Misaligned Tensor data on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected data alignment=16 bytes
      print(f"ValueError: {e}")
```

Any CUDA errors encountered will also be automatically converted into Python exceptions by the TVM FFI function.

```python
@cute.jit
def add_one_invalid_launch(a: cute.Tensor, b: cute.Tensor):
   # Intentionally exceed the maximum block dimension (1024 threads) so the
   # CUDA runtime reports an invalid configuration error.
   device_add_one(a, b).launch(grid=(1, 1, 1), block=(4096, 1, 1))

def example_error_cuda_error():
   a_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")
   b_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")

   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True)
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True)
   compiled_add_one_invalid_launch = cute.compile(
      add_one_invalid_launch, a_cute, b_cute, options="--enable-tvm-ffi"
   )

   try:
      compiled_add_one_invalid_launch(a_torch, b_torch)
   except RuntimeError as e:
      # raises RuntimeError: CUDA Error: cudaErrorInvalidValue
      print(f"RuntimeError: {e}")
```

## Working with Devices[#](#working-with-devices "Link to this heading")

TVM FFI-compiled functions naturally work across GPU devices.
The device index of the first input GPU tensor determines the kernelâs device context.
The TVM FFI function calls `cudaSetDevice` to set the correct device
before launching the kernel based on that tensorâs device index.
For advanced scenarios that pass raw pointers instead of tensors, you should call
`cudaSetDevice` explicitly through the CUDA Python API.

## Exporting Compiled Module[#](#exporting-compiled-module "Link to this heading")

The TVM FFI function supports exporting the compiled module to an object file
for further use. For example:

```python
import subprocess
import cutlass.cute as cute

def example_add_one_export():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_one.export_to_c("./add_one.o", function_name="add_one")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_one.so", "./add_one.o", *runtime_libs]
   print(cmd)
   subprocess.run(cmd, check=True)
   print(f"Successfully created shared library: ./add_one.so")
```

Then you can load back the exported module and use it in different ways:

```python
import torch
from cutlass import cute

def example_load_module_add_one():
   mod = cute.runtime.load_module("./add_one.so", enable_tvm_ffi=True)
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   mod.add_one(a_torch, b_torch)
   print("result of b_torch after mod.add_one(a_torch, b_torch)")
   print(b_torch)
```

The exported object file exposes the function symbol `__tvm_ffi_add_one` that is
compatible with TVM FFI and can be used in various frameworks and programming languages.
You can either build a shared library and load it back, or link the object file directly
into your application and invoke the function via the `InvokeExternC` mechanism in TVM FFI.
For more information, see the [quick start guide](https://tvm.apache.org/ffi/get_started/quickstart)
in the official documentation.

When you build your own libraries, make sure you link against the necessary runtime libraries.
You can use `cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)` to get the path to these libraries.
`cute.runtime.load_module(path, enable_tvm_ffi=True)` will load these libraries automatically before loading
an exported module. You can also manually load these libraries in advanced use cases.

For low-level cute ABI AOT compilation support without TVM FFI, you can refer to [Ahead-of-Time (AOT) Compilation](dsl_ahead_of_time_compilation.html).

### Keyword Arguments and Defaults[#](#keyword-arguments-and-defaults "Link to this heading")

The function returned by `cute.compile` supports keyword arguments and defaults.
The example below shows how to use keyword arguments and defaults:

```python
import torch
from cutlass import cute

@cute.kernel
def device_add_scalar(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + offset

@cute.jit
def add_constant(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32=cutlass.Float32(1)):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_scalar(a, b, offset).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_constant(a_torch, b_torch)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch)")
   print(b_torch)
   compiled_add_constant(a_torch, b_torch, offset=4)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch, offset=4)")
   print(b_torch)
```

For efficiency and portability reasons, TVM FFI ABI supports functions with positional-only arguments.
If you export the compiled module to an object file and then load it back, the function
will only accept positional arguments in the order of the arguments in the function signature.
You can rewrap the function or use the TVM FFI wrapper generator to generate a kwargs wrapper.
The code block below shows how to do this:

```python
def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_constant.export_to_c("./add_constant.o", function_name="add_constant")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_constant.so", "./add_constant.o", *runtime_libs]
   subprocess.run(cmd, check=True)

   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")

   mod = cute.runtime.load_module("./add_constant.so")
   try:
      mod.add_constant(a_torch, b_torch)
   except Exception as e:
      # Raises a missing arguments error because kwargs and default information are lost
      print(e)
   # We rewrap the function to regain argument and kwargs support.
   # Alternatively, use the TVM FFI wrapper generator to generate a kwargs wrapper function.
   from tvm_ffi.utils import kwargs_wrapper
   # arg_defaults are aligned to the end of the argument list
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper(
      mod.add_constant, arg_names=["a", "b", "offset"], arg_defaults=(1,)
   )
   wrapped_func(a_torch, b_torch)
   print("result of b_torch after wrapped_func(a_torch, b_torch)")
   print(b_torch)
   # You can also use the signature of the original function
   # to generate a kwargs wrapper function. Make sure to exclude
   # arguments that are not included in the runtime,
   # such as 'self', constexpr, and env stream arguments.
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper_from_signature(
      mod.add_constant, signature=inspect.signature(add_constant),
      exclude_arg_names=["self"]
   )
   wrapped_func(a_torch, b_torch, offset=4)
   print("result of b_torch after wrapped_func(a_torch, b_torch, offset=4)")
   print(b_torch)
```

## Limitations[#](#limitations "Link to this heading")

The Fake Tensor flow is ONLY compatible with TVM FFI because TVM FFI supports more flexible constraints on Tensor arguments.
For instance, fake tensor can specify per-mode static shape or constraints on shape and strides which are not supported by
existing `from_dlpack` flow. Itâs expected that JIT function compiled with fake tensor will have different ABI compared to
tensor converted by `from_dlpack`.

```python
import cutlass.cute as cute
import torch

n = cute.sym_int()
# Dynamic Shape
fake_a = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

# Compile without tvm-ffi
compiled_fn = cute.compile(foo, fake_a)

# Wrong, in compatible ABI
compiled_fn(from_dlpack(a))
```

In order to avoid such issue, itâs recommended to use fake tensor only with TVM FFI backend. Practically speaking,
as we only want to call `from_dlpack` once and reuse for both compilation and runtime, the benefit of
using fake tensor is limited in this case.

[previous

Deprecation Policy](../deprecation.html "previous page")
[next

Ahead-of-Time (AOT) Compilation](dsl_ahead_of_time_compilation.html "next page")

On this page

* [Enable Apache TVM FFI in CuTe DSL](#enable-apache-tvm-ffi-in-dsl)
* [Minimizing Host Overhead](#minimizing-host-overhead)
* [Fake tensor for compilation](#fake-tensor-for-compilation)
  + [Note on Stride Order](#note-on-stride-order)
* [`cute.Tensor` adapter for TVM FFI](#cute-tensor-adapter-for-tvm-ffi)
* [Working with torch Tensors](#working-with-torch-tensors)
* [Working with Streams](#working-with-streams)
  + [Using Environment Stream](#using-environment-stream)
* [Working with Tuples](#working-with-tuples)
* [Working with Variadic Tuples](#working-with-variadic-tuples)
  + [Working with Named Tuples](#working-with-named-tuples)
* [Supported types](#supported-types)
* [Error handling](#error-handling)
* [Working with Devices](#working-with-devices)
* [Exporting Compiled Module](#exporting-compiled-module)
  + [Keyword Arguments and Defaults](#keyword-arguments-and-defaults)
* [Limitations](#limitations)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.