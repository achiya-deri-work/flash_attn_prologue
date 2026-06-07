

JIT Function Argument Generation — NVIDIA CUTLASS Documentation



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
  + [JIT Argument Generation](#)
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
* JIT Function Argument Generation

# JIT Function Argument Generation[#](#jit-function-argument-generation "Link to this heading")

## In a nutshell[#](#in-a-nutshell "Link to this heading")

When using the `@jit` or `@kernel` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT functionâs signature.
CuTe DSL provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the CuTe DSL will take care of the rest for you.

Specifically, CuTe DSL honors following when generating the JIT functionâs arguments:

* JIT function arguments are assumed to be **dynamic arguments** by default.
* If an argument is explicitly type annotated with `cutlass.Constexpr`, it is treated as a **compile-time constant**.
* If type annotation is provided, CuTe DSL validates the argument type at compile time for **type safety**.
* CuTe DSL provides **runtime checkable protocols** (`JitArgument` and `DynamicExpression`) for generating JIT function arguments for customized types.

More details below for each of the above.

## Static argument vs. Dynamic argument[#](#static-argument-vs-dynamic-argument "Link to this heading")

CuTe DSL supports both static and dynamic arguments for JIT functions.

1. **Static arguments** hold values that are known at compile time. It is not included in the generated JIT function signature.
2. **Dynamic arguments** hold values that are only known at runtime.

By default, CuTe DSL assumes dynamic arguments and tries to infer the argument types from the call-site argument types. An explicit type annotation `cutlass.Constexpr` can be used to specify a static argument.

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)        # Prints x = ?
    print("y = ", y)        # Prints y = 2
    cute.printf("x: {}", x) # Prints x: 2
    cute.printf("y: {}", y) # Prints y: 2

foo(2, 2)
```

In the example above, `x` is a dynamic argument with type cutlass.Int32 and `y` is a static argument.

With the `cutlass.Constexpr` annotation, a more sophisticated uses case of static argument in the JIT functions can be something like:

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
    epi_tile: cute.Tile,
    epilogue_op: cutlass.Constexpr,
):
    ...

    # Perform epilogue op on accumulator and convert to C type
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)
```

In this example, `epilogue_op` is a static argument in the JIT kernel where the argument is used for the epilogue fusion. Upon calling the kernel,
an elementwise lambda function can be passed in as the `epilogue_op` argument. For example, a ReLU can be applied for epilogue fusion by simply setting the
`epilogue_op` to `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`

Refer to the [Blackwell dense GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) for a complete example.

## Type safety[#](#type-safety "Link to this heading")

CuTe DSL makes good use of type annotation in JIT function signature and validates the JIT function argument types at compile time for **type safety**.

```python
import cutlass
import cutlass.cute as cute
import numpy as np

@cute.jit
def foo(x: cute.Tensor, y: cutlass.Float16):
    ...

a = np.random.randn(10, 10).astype(np.float16)
b = 32

foo(a, b)
foo(b, a)  # This will fail at compile time due to type mismatch
```

The type safety check helps catch the type mismatch issue early at the compile time with clear error message to avoid tricky runtime errors which is usually more expensive to debug.
In the example above, the second call to `foo` will fail at compile time due to the type mismatch with a clear error message:

```python
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>
```

## JIT function arguments with customized types[#](#jit-function-arguments-with-custom-types "Link to this heading")

CuTe DSL supports customized types for JIT function arguments by providing two runtime checkable protocols:

* `JitArgument` which is used for host JIT functions to be called from Python.
  :   + `__c_pointers__`: Generate a list of ctypes pointers for the current object.
      + `__get_mlir_types__`: Generate a list of MLIR types for the current object.
      + `__new_from_mlir_values__`: Create a new object from MLIR values.
* `DynamicExpression` which is used for device JIT functions to be called from the host JIT functions.
  :   + `__extract_mlir_values__`: Generate a dynamic expression for the current object.
      + `__new_from_mlir_values__`: Create a new object from MLIR values.

Refer to [typing.py](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py) for more details on these protocol APIs.

Depending on different cases of the customized types, CuTe DSL provides easy ways to adopt customized types for JIT function arguments.

### 1. Direct protocol implementation in customized types[#](#direct-protocol-implementation-in-custom-types "Link to this heading")

One way is to implement the protocol methods directly in the customized types to enable the protocol based JIT function argument generation.

```python
import cutlass
import cutlass.cute as cute

# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor # Dynamic argument
        self._offset = offset # Dynamic argument

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```

In the example above, the `MyDynamicExpression` implements the `DynamicExpression` protocol and CuTe DSL will generate the JIT function arguments for the JIT kernel `my_kernel` based on the protocol methods.

### 2. Adaptor based protocol implementation for customized types[#](#adaptor-based-protocol-implementation-for-custom-types "Link to this heading")

For the case where directly changing the customized types to implement the protocol is not feasible, CuTe DSL provides adaptor based approach to adapt the customized types for JIT function argument generation.

The JIT function argument adaptor is a callable object that implements the desired protocol methods for the registered customized types. This way, CuTe DSL automatically queries the JIT argument adaptor registry
to generate the JIT function arguments for the given customized types.

```python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # Convert the framework object to a C-ABI compatible object
        # thru its C-ABI interface
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # Return the list of MLIR types the framework object represents
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # Convert the MLIR values back to the framework object
        return MyFrameworkObject(values[0])
```

In this example, the `MyFrameworkObjectAdapter` implements an adaptor class which bridges the CuTe DSL and the 3rd party framework type `MyFrameworkObject`.
The registration is done by just decorating the adaptor with `cutlass.register_jit_arg_adapter` for the customized type. With the registered adaptor,
CuTe DSL will automatically use the adaptor to generate the JIT function arguments for `MyFrameworkObject` typed arguments.

[previous

Control Flow](dsl_control_flow.html "previous page")
[next

Static vs Dynamic layouts](dsl_dynamic_layout.html "next page")

On this page

* [In a nutshell](#in-a-nutshell)
* [Static argument vs. Dynamic argument](#static-argument-vs-dynamic-argument)
* [Type safety](#type-safety)
* [JIT function arguments with customized types](#jit-function-arguments-with-custom-types)
  + [1. Direct protocol implementation in customized types](#direct-protocol-implementation-in-custom-types)
  + [2. Adaptor based protocol implementation for customized types](#adaptor-based-protocol-implementation-for-custom-types)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.