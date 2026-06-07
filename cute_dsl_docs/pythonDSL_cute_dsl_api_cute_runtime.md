

Runtime — NVIDIA CUTLASS Documentation



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
    - [Runtime](#)
  + [cute\_nvgpu](cute_nvgpu.html)
    - [Common](cute_nvgpu_common.html)
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
* [cutlass.cute](cute.html)
* Runtime

# Runtime[#](#runtime "Link to this heading")

Description

## API documentation[#](#module-cutlass.cute.runtime "Link to this heading")

*class* cutlass.cute.runtime.\_Pointer(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.runtime._Pointer "Link to this definition")
:   Bases: `Pointer`

    Runtime representation of a pointer that can inter-operate with various data structures,
    including numpy arrays and device memory.

    Parameters:
    :   * **pointer** (*int* *or* *pointer-like object*) â The pointer to the data
        * **dtype** (*Type*) â Data type of the elements pointed to
        * **mem\_space** (*\_cute\_ir.AddressSpace**,* *optional*) â Memory space where the pointer resides, defaults to generic
        * **assumed\_align** (*int**,* *optional*) â Assumed alignment of input pointer in bytes, defaults to None

    Variables:
    :   * **\_pointer** â The underlying pointer
        * **\_dtype** â Data type of the elements
        * **\_addr\_space** â Memory space of the pointer
        * **\_assumed\_align** â Alignment of the pointer in bytes
        * **\_desc** â C-type descriptor for the pointer
        * **\_c\_pointer** â C-compatible pointer representation

    \_\_init\_\_( : *pointer*, : *dtype*, : *mem\_space: cutlass.\_mlir.dialects.cute.AddressSpace = cutlass.\_mlir.dialects.cute.AddressSpace.generic*, : *assumed\_align=None*, )[#](#cutlass.cute.runtime._Pointer.__init__ "Link to this definition")

    size\_in\_bytes() → int[#](#cutlass.cute.runtime._Pointer.size_in_bytes "Link to this definition")

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*[#](#cutlass.cute.runtime._Pointer.mlir_type "Link to this definition")

    *property* dtype*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.runtime._Pointer.dtype "Link to this definition")

    *property* memspace[#](#cutlass.cute.runtime._Pointer.memspace "Link to this definition")

    align( : *min\_align: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.runtime._Pointer.align "Link to this definition")

*class* cutlass.cute.runtime.\_Tensor(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.runtime._Tensor "Link to this definition")
:   Bases: `Tensor`

    \_\_init\_\_( : *tensor*, : *assumed\_align=None*, : *use\_32bit\_stride=False*, : *\**, : *enable\_tvm\_ffi=False*, )[#](#cutlass.cute.runtime._Tensor.__init__ "Link to this definition")

    load\_dltensor()[#](#cutlass.cute.runtime._Tensor.load_dltensor "Link to this definition")
    :   Lazily load the DLTensorWrapper.

        This function loads the DLTensorWrapper when needed,
        avoiding overhead in the critical path of calling JIT functions.

    mark\_layout\_dynamic(*leading\_dim: int | None = None*)[#](#cutlass.cute.runtime._Tensor.mark_layout_dynamic "Link to this definition")
    :   Marks the tensor layout as dynamic based on the leading dimension.

        Parameters:
        :   **leading\_dim** (*int**,* *optional*) â The leading dimension of the layout, defaults to None

        When `leading_dim` is None, the leading dimension is deduced as follows.

        1. If exactly one dimension has stride 1, that dimension is used.
        2. If multiple dimensions have stride 1 but exactly one of them has size > 1,
           that dimension is used.
        3. If multiple dimensions have stride 1 but none or more than one has size > 1,
           an error is raised.
        4. If no dimension has stride 1, all strides remain dynamic.

        When `leading_dim` is explicitly specified, marks the layout as dynamic while setting the
        stride at `leading_dim` to 1. Also validates that the specified `leading_dim` is consistent
        with the existing layout by checking that the corresponding stride of that dimension is 1.

        Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

        Returns:
        :   The tensor with dynamic layout

        Return type:
        :   [\_Tensor](#cutlass.cute.runtime._Tensor "cutlass.cute.runtime._Tensor")

    mark\_compact\_shape\_dynamic( : *mode: int*, : *stride\_order: tuple[int, ...] | None = None*, : *divisibility: int = 1*, )[#](#cutlass.cute.runtime._Tensor.mark_compact_shape_dynamic "Link to this definition")
    :   Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

        Parameters:
        :   * **mode** (*int*) â The mode of the compact shape, defaults to 0
            * **stride\_order** â Consistent with torch.Tensor.dim\_order. Defaults to None.

        Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
        It starts from the outermost to the innermost dimension.
        :type stride\_order: tuple[int, â¦], optional
        :param divisibility: The divisibility constraint for the compact shape, defaults to 1
        :type divisibility: int, optional
        :return: The tensor with dynamic compact shape
        :rtype: \_Tensor

        If `stride_order` is not provided, the stride ordering will be automatically deduced from the layout.
        Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
        An error is raised if automatic deduction fails.

        If `stride_order` is explicitly specified, it does the consistency check with the layout.

        For example:
        - Layout: (4,2):(1,4) has stride\_order: (1,0) indicates the innermost dimension is 0(4:1), the outermost dimension is 1(2:4)
        - Layout: (5,3,2,4):(3,1,15,30) has stride\_order: (3,2,0,1) indicates the innermost dimension is 1(3:1), the outermost dimension is 3(4:30).

        Using torch.Tensor.dim\_order() to get the stride order of the torch tensor.
        .. code-block:: python
        a = torch.empty(3, 4)
        t = cute.runtime.from\_dlpack(a)
        t = t.mark\_compact\_shape\_dynamic(mode=0, stride\_order=a.dim\_order())

    *property* element\_type*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.runtime._Tensor.element_type "Link to this definition")

    *property* memspace[#](#cutlass.cute.runtime._Tensor.memspace "Link to this definition")

    *property* size\_in\_bytes*: int*[#](#cutlass.cute.runtime._Tensor.size_in_bytes "Link to this definition")

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*[#](#cutlass.cute.runtime._Tensor.mlir_type "Link to this definition")

    *property* iterator[#](#cutlass.cute.runtime._Tensor.iterator "Link to this definition")

    *property* layout[#](#cutlass.cute.runtime._Tensor.layout "Link to this definition")

    *property* shape[#](#cutlass.cute.runtime._Tensor.shape "Link to this definition")

    *property* stride[#](#cutlass.cute.runtime._Tensor.stride "Link to this definition")

    *property* leading\_dim[#](#cutlass.cute.runtime._Tensor.leading_dim "Link to this definition")
    :   Get the leading dimension of this Tensor.

        Returns:
        :   The leading dimension index or indices

        Return type:
        :   int or tuple or None

        The return value depends on the tensorâs stride pattern:

        * If a single leading dimension is found, returns an integer index
        * If nested leading dimensions are found, returns a tuple of indices
        * If no leading dimension is found, returns None

    fill(*value: cutlass.cute.typing.Numeric*)[#](#cutlass.cute.runtime._Tensor.fill "Link to this definition")

    *property* data\_ptr[#](#cutlass.cute.runtime._Tensor.data_ptr "Link to this definition")

    *property* dynamic\_shapes\_mask[#](#cutlass.cute.runtime._Tensor.dynamic_shapes_mask "Link to this definition")
    :   Get the mask of dynamic shapes in the tensor.

    *property* dynamic\_strides\_mask[#](#cutlass.cute.runtime._Tensor.dynamic_strides_mask "Link to this definition")
    :   Get the mask of dynamic strides in the tensor.

cutlass.cute.runtime.\_get\_cute\_type\_str(*inp*)[#](#cutlass.cute.runtime._get_cute_type_str "Link to this definition")

*class* cutlass.cute.runtime.\_FakeTensor(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.runtime._FakeTensor "Link to this definition")
:   Bases: `Tensor`

    Fake Tensor implementation as a placeholder.
    It mimics the interface of Tensor, but does not hold real data or allow indexing.
    Used for compilation or testing situations where only shape/type/layout information is needed.
    All attempts to access or mutate data will raise errors.

    \_\_init\_\_( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *stride: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, : *use\_32bit\_stride: bool = False*, : *compact: bool = False*, )[#](#cutlass.cute.runtime._FakeTensor.__init__ "Link to this definition")

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*[#](#cutlass.cute.runtime._FakeTensor.mlir_type "Link to this definition")

    *property* element\_type*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.runtime._FakeTensor.element_type "Link to this definition")

    *property* memspace[#](#cutlass.cute.runtime._FakeTensor.memspace "Link to this definition")

    *property* iterator[#](#cutlass.cute.runtime._FakeTensor.iterator "Link to this definition")

    *property* shape[#](#cutlass.cute.runtime._FakeTensor.shape "Link to this definition")

    *property* stride[#](#cutlass.cute.runtime._FakeTensor.stride "Link to this definition")

    *property* leading\_dim[#](#cutlass.cute.runtime._FakeTensor.leading_dim "Link to this definition")

    *property* dynamic\_shapes\_mask[#](#cutlass.cute.runtime._FakeTensor.dynamic_shapes_mask "Link to this definition")

    *property* dynamic\_strides\_mask[#](#cutlass.cute.runtime._FakeTensor.dynamic_strides_mask "Link to this definition")

    fill(*value: cutlass.cute.typing.Numeric*)[#](#cutlass.cute.runtime._FakeTensor.fill "Link to this definition")

cutlass.cute.runtime.make\_fake\_compact\_tensor( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *stride\_order: tuple[int, ...] | None = None*, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, : *use\_32bit\_stride: bool = False*, )[#](#cutlass.cute.runtime.make_fake_compact_tensor "Link to this definition")
:   Create a fake tensor with the specified shape, element type, and a compact memory layout.

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) â Data type of the tensor elements.
        * **shape** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) â Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
        * **stride\_order** (*tuple**[**int**,* *...**]**,* *optional*) â Order in which strides (memory layout) are assigned to the tensor dimensions.
          If None, the default layout is left-to-right order (known as column-major order for flatten layout).
          Otherwise, it should be a permutation order of the dimension indices.
          The mode with stride\_order 0 is the fastest changing (leading) dimension, and N-1 is the slowest changing.
        * **memspace** (*AddressSpace**,* *optional*) â Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
        * **assumed\_align** (*int**,* *optional*) â Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.
        * **use\_32bit\_stride** (*bool**,* *optional*) â Whether to use 32-bit stride for dynamic dimensions. If True and the total size of the
          layout (cosize(layout)) fits within int32, then dynamic strides will use 32-bit integers for improved performance.
          Only applies when dimensions are dynamic. Defaults to False.

    Returns:
    :   An instance of a fake tensor with the given properties and compact layout.

    Return type:
    :   [\_FakeTensor](#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

    **Examples:**

    ```python
    @cute.jit
    def foo(x: cute.Tensor):
        ...

    x = make_fake_compact_tensor(
        cutlass.Float32, (100, cute.sym_int32(divisibility=8)), stride_order=(1, 0)
    )

    # Compiled function will take a tensor with the type:
    #   tensor<ptr<f32, generic> o (100,?{div=8}):(?{i32 div=8},1)>
    compiled_foo = cute.compile(foo, x)

    # Default stride order is left-to-right order (0, 1, ..., n-1)
    y = make_fake_compact_tensor(cutlass.Float32, (8, 3, 2)) # y.stride == (1, 8, 24)
    ```

cutlass.cute.runtime.make\_fake\_tensor( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *stride: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, )[#](#cutlass.cute.runtime.make_fake_tensor "Link to this definition")
:   Create a fake tensor with the specified element type, shape, and stride.

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) â Data type of the tensor elements.
        * **shape** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) â Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
        * **stride** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) â Stride of the tensor, consisting of static (int) or dynamic (SymInt) values.
        * **memspace** (*AddressSpace**,* *optional*) â Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
        * **assumed\_align** (*int**,* *optional*) â Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.

    Returns:
    :   An instance of a fake tensor with the given properties.

    Return type:
    :   [\_FakeTensor](#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

*class* cutlass.cute.runtime.\_FakeStream(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)[#](#cutlass.cute.runtime._FakeStream "Link to this definition")
:   Bases: `object`

    A fake stream that can be used as a placeholder for a stream in compilation.

    When use\_tvm\_ffi\_env\_stream is True and the function is compiled with TVM-FFI,
    the argument will be skipped from the function signature and we pass in
    this value through the environment stream obtained from caller context
    (e.g. torch.cuda.current\_stream()).

    \_\_init\_\_(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)[#](#cutlass.cute.runtime._FakeStream.__init__ "Link to this definition")

    use\_tvm\_ffi\_env\_stream*: bool*[#](#cutlass.cute.runtime._FakeStream.use_tvm_ffi_env_stream "Link to this definition")

cutlass.cute.runtime.make\_fake\_stream(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)[#](#cutlass.cute.runtime.make_fake_stream "Link to this definition")
:   Create a fake stream that can be used as a placeholder for a stream in compilation.

    When use\_tvm\_ffi\_env\_stream is True and the function is compiled with TVM-FFI,
    the argument will be skipped from the function signature and we pass in
    this value through the environment stream obtained from caller context
    (e.g. torch.cuda.current\_stream()). This can speedup the calling process
    since we no longer need to do stream query in python.

    Parameters:
    :   **use\_tvm\_ffi\_env\_stream** (*bool*) â Whether to skip this parameter use environment stream instead.

cutlass.cute.runtime.from\_dlpack( : *tensor\_dlpack*, : *assumed\_align=None*, : *use\_32bit\_stride=False*, : *\**, : *enable\_tvm\_ffi=False*, : *force\_tf32=False*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.runtime.from_dlpack "Link to this definition")
:   Convert from tensor object supporting \_\_dlpack\_\_() to a CuTe Tensor.

    Parameters:
    :   * **tensor\_dlpack** (*object*) â Tensor object that supports the DLPack protocol
        * **assumed\_align** (*int**,* *optional*) â Assumed alignment of the tensor (bytes), defaults to None,
          if None, will use the element size bytes as the assumed alignment.
        * **use\_32bit\_stride** (*bool**,* *optional*) â Whether to use 32-bit stride, defaults to False. When True, the dynamic
          stride bitwidth will be set to 32 for small problem size (cosize(layout) <= Int32\_max) for better performance.
          This is only applied when the dimension is dynamic.
        * **enable\_tvm\_ffi** (*bool**,* *optional*) â Whether to enable TVM-FFI, defaults to False. When True, the tensor will be converted to
          a TVM-FFI function compatible tensor.
        * **force\_tf32** (*bool**,* *optional*) â Whether to force the element type to TFloat32 if the element type is Float32.

    Returns:
    :   A CuTe Tensor object

    Return type:
    :   Tensor

    **Examples:**

    ```python
    import torch
    from cutlass.cute.runtime import from_dlpack
    x = torch.randn(100, 100)
    y = from_dlpack(x)
    y.shape
    # (100, 100)
    type(y)
    # <class 'cutlass.cute.Tensor'>
    ```

cutlass.cute.runtime.make\_ptr( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *value: int | \_Pointer*, : *mem\_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic*, : *assumed\_align=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.runtime.make_ptr "Link to this definition")
:   Create a pointer from a memory address

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) â Data type of the pointer elements
        * **value** (*Union**[**int**,* *ctypes.\_Pointer**]*) â Memory address as integer or ctypes pointer
        * **mem\_space** (*AddressSpace**,* *optional*) â Memory address space, defaults to AddressSpace.generic
        * **align\_bytes** (*int**,* *optional*) â Alignment in bytes, defaults to None

    Returns:
    :   A pointer object

    Return type:
    :   Pointer

    ```python
    import numpy as np
    import ctypes

    from cutlass import Float32
    from cutlass.cute.runtime import make_ptr

    # Create a numpy array
    a = np.random.randn(16, 32).astype(np.float32)

    # Get pointer address as integer
    ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Create pointer from address
    y = make_ptr(cutlass.Float32, ptr_address)

    # Check properties
    print(y.element_type)
    print(type(y))  # <class 'cutlass.cute.Pointer'>
    ```

cutlass.cute.runtime.nullptr( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *mem\_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic*, : *assumed\_align=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.runtime.nullptr "Link to this definition")
:   Create a null pointer which is useful for compilation

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) â Data type of the pointer elements
        * **mem\_space** (*AddressSpace**,* *optional*) â Memory address space, defaults to AddressSpace.generic

    Returns:
    :   A null pointer object

    Return type:
    :   Pointer

*class* cutlass.cute.runtime.TensorAdapter(*arg*)[#](#cutlass.cute.runtime.TensorAdapter "Link to this definition")
:   Bases: `object`

    Convert a DLPack protocol supported tensor/array to a cute tensor.

    \_\_init\_\_(*arg*)[#](#cutlass.cute.runtime.TensorAdapter.__init__ "Link to this definition")

cutlass.cute.runtime.find\_runtime\_libraries( : *\**, : *enable\_tvm\_ffi: bool = True*, ) → List[str][#](#cutlass.cute.runtime.find_runtime_libraries "Link to this definition")
:   Find the runtime libraries that needs to be available for loading modules.

    Parameters:
    :   **enable\_tvm\_ffi** (*bool**,* *optional*) â Whether to enable TVM-FFI.

    Returns:
    :   A list of runtime libraries that needs to be available for loading modules.

    Return type:
    :   list

cutlass.cute.runtime.load\_module(*file\_path: str*, *\**, *enable\_tvm\_ffi: bool = False*)[#](#cutlass.cute.runtime.load_module "Link to this definition")
:   Load a module from a file path.

    Parameters:
    :   * **file\_path** (*str*) â The path to the module file
        * **enable\_tvm\_ffi** (*bool**,* *optional*) â Whether to enable TVM-FFI, defaults to True. When True, the module will be loaded as a TVM-FFI module.

    Returns:
    :   A module object

    Return type:
    :   module

[previous

arch](cute_arch.html "previous page")
[next

cutlass.cute.nvgpu](cute_nvgpu.html "next page")

On this page

* [API documentation](#module-cutlass.cute.runtime)
  + [`_Pointer`](#cutlass.cute.runtime._Pointer)
    - [`__init__()`](#cutlass.cute.runtime._Pointer.__init__)
    - [`size_in_bytes()`](#cutlass.cute.runtime._Pointer.size_in_bytes)
    - [`mlir_type`](#cutlass.cute.runtime._Pointer.mlir_type)
    - [`dtype`](#cutlass.cute.runtime._Pointer.dtype)
    - [`memspace`](#cutlass.cute.runtime._Pointer.memspace)
    - [`align()`](#cutlass.cute.runtime._Pointer.align)
  + [`_Tensor`](#cutlass.cute.runtime._Tensor)
    - [`__init__()`](#cutlass.cute.runtime._Tensor.__init__)
    - [`load_dltensor()`](#cutlass.cute.runtime._Tensor.load_dltensor)
    - [`mark_layout_dynamic()`](#cutlass.cute.runtime._Tensor.mark_layout_dynamic)
    - [`mark_compact_shape_dynamic()`](#cutlass.cute.runtime._Tensor.mark_compact_shape_dynamic)
    - [`element_type`](#cutlass.cute.runtime._Tensor.element_type)
    - [`memspace`](#cutlass.cute.runtime._Tensor.memspace)
    - [`size_in_bytes`](#cutlass.cute.runtime._Tensor.size_in_bytes)
    - [`mlir_type`](#cutlass.cute.runtime._Tensor.mlir_type)
    - [`iterator`](#cutlass.cute.runtime._Tensor.iterator)
    - [`layout`](#cutlass.cute.runtime._Tensor.layout)
    - [`shape`](#cutlass.cute.runtime._Tensor.shape)
    - [`stride`](#cutlass.cute.runtime._Tensor.stride)
    - [`leading_dim`](#cutlass.cute.runtime._Tensor.leading_dim)
    - [`fill()`](#cutlass.cute.runtime._Tensor.fill)
    - [`data_ptr`](#cutlass.cute.runtime._Tensor.data_ptr)
    - [`dynamic_shapes_mask`](#cutlass.cute.runtime._Tensor.dynamic_shapes_mask)
    - [`dynamic_strides_mask`](#cutlass.cute.runtime._Tensor.dynamic_strides_mask)
  + [`_get_cute_type_str()`](#cutlass.cute.runtime._get_cute_type_str)
  + [`_FakeTensor`](#cutlass.cute.runtime._FakeTensor)
    - [`__init__()`](#cutlass.cute.runtime._FakeTensor.__init__)
    - [`mlir_type`](#cutlass.cute.runtime._FakeTensor.mlir_type)
    - [`element_type`](#cutlass.cute.runtime._FakeTensor.element_type)
    - [`memspace`](#cutlass.cute.runtime._FakeTensor.memspace)
    - [`iterator`](#cutlass.cute.runtime._FakeTensor.iterator)
    - [`shape`](#cutlass.cute.runtime._FakeTensor.shape)
    - [`stride`](#cutlass.cute.runtime._FakeTensor.stride)
    - [`leading_dim`](#cutlass.cute.runtime._FakeTensor.leading_dim)
    - [`dynamic_shapes_mask`](#cutlass.cute.runtime._FakeTensor.dynamic_shapes_mask)
    - [`dynamic_strides_mask`](#cutlass.cute.runtime._FakeTensor.dynamic_strides_mask)
    - [`fill()`](#cutlass.cute.runtime._FakeTensor.fill)
  + [`make_fake_compact_tensor()`](#cutlass.cute.runtime.make_fake_compact_tensor)
  + [`make_fake_tensor()`](#cutlass.cute.runtime.make_fake_tensor)
  + [`_FakeStream`](#cutlass.cute.runtime._FakeStream)
    - [`__init__()`](#cutlass.cute.runtime._FakeStream.__init__)
    - [`use_tvm_ffi_env_stream`](#cutlass.cute.runtime._FakeStream.use_tvm_ffi_env_stream)
  + [`make_fake_stream()`](#cutlass.cute.runtime.make_fake_stream)
  + [`from_dlpack()`](#cutlass.cute.runtime.from_dlpack)
  + [`make_ptr()`](#cutlass.cute.runtime.make_ptr)
  + [`nullptr()`](#cutlass.cute.runtime.nullptr)
  + [`TensorAdapter`](#cutlass.cute.runtime.TensorAdapter)
    - [`__init__()`](#cutlass.cute.runtime.TensorAdapter.__init__)
  + [`find_runtime_libraries()`](#cutlass.cute.runtime.find_runtime_libraries)
  + [`load_module()`](#cutlass.cute.runtime.load_module)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.