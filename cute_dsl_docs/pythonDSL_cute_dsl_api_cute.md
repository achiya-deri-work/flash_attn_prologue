

cutlass.cute — NVIDIA CUTLASS Documentation



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
  + [cute](#)
    - [arch](cute_arch.html)
    - [Runtime](cute_runtime.html)
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
* cutlass.cute

# cutlass.cute[#](#module-cutlass.cute "Link to this heading")

*class* cutlass.cute.Swizzle(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.Swizzle "Link to this definition")
:   Bases: `Value`

    Swizzle is a transformation that permutes the elements of a layout.

    Swizzles are used to rearrange data elements to improve memory access patterns
    and computational efficiency.

    Swizzle is defined by three parameters:
    - MBase: The number of least-significant bits to keep constant
    - BBits: The number of bits in the mask
    - SShift: The distance to shift the mask

    The mask is applied to the least-significant bits of the layout.

    ```python
    0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                  ^--^ MBase is the number of least-sig bits to keep constant
                     ^-^       ^-^     BBits is the number of bits in the mask
                       ^---------^     SShift is the distance to shift the YYY mask
                                          (pos shifts YYY to the right, neg shifts YYY to the left)

    e.g. Given
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

    the result is
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
    ```

*class* cutlass.cute.struct(*cls*)[#](#cutlass.cute.struct "Link to this definition")
:   Bases: `object`

    Decorator to abstract C structure in Python DSL.

    **Usage:**

    ```python
    # Supports base_dsl scalar int/float elements, array and nested struct:
    @cute.struct
    class complex:
        real : cutlass.Float32
        imag : cutlass.Float32


    @cute.struct
    class StorageA:
        mbarA : cute.struct.MemRange[cutlass.Int64, stage]
        compA : complex
        intA : cutlass.Int16


    # Supports alignment for its elements:
    @cute.struct
    class StorageB:
        a: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, size_a], 1024
        ]
        b: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, size_b], 1024
        ]
        x: cute.struct.Align[cutlass.Int32, 16]
        compA: cute.struct.Align[complex, 16]


    # Statically get size and alignment:
    size = StorageB.__sizeof__()
    align = StorageB.__alignof__()

    # Allocate and referencing elements:
    storage = allocator.allocate(StorageB)

    storage.a[0] ...
    storage.x.ptr ...
    storage.compA.real.ptr ...
    ```

    Parameters:
    :   **cls** â The struct class with annotations.

    Returns:
    :   The decorated struct class.

    *class* \_MemRangeMeta(*name*, *bases*, *dct*)[#](#cutlass.cute.struct._MemRangeMeta "Link to this definition")
    :   Bases: `type`

        A metaclass for creating MemRange classes.

        This metaclass is used to dynamically create MemRange classes with specific
        data types and sizes.

        Variables:
        :   * **\_dtype** â The data type of the MemRange.
            * **\_size** â The size of the MemRange.

        \_dtype*: cutlass.cute.typing.Numeric | None* *= None*[#](#cutlass.cute.struct._MemRangeMeta._dtype "Link to this definition")

        \_size*: int | None* *= None*[#](#cutlass.cute.struct._MemRangeMeta._size "Link to this definition")

        *property* size[#](#cutlass.cute.struct._MemRangeMeta.size "Link to this definition")

        *property* elem\_width[#](#cutlass.cute.struct._MemRangeMeta.elem_width "Link to this definition")

        *property* size\_in\_bytes[#](#cutlass.cute.struct._MemRangeMeta.size_in_bytes "Link to this definition")

    *class* MemRange[#](#cutlass.cute.struct.MemRange "Link to this definition")
    :   Bases: `object`

        Defines a range of memory by MemRange[T, size].

        \_dtype*: cutlass.cute.typing.Numeric | None* *= None*[#](#cutlass.cute.struct.MemRange._dtype "Link to this definition")

        \_size*: int | None* *= None*[#](#cutlass.cute.struct.MemRange._size "Link to this definition")

    *class* \_MemRangeData(*dtype*, *size*, *base*)[#](#cutlass.cute.struct._MemRangeData "Link to this definition")
    :   Bases: `object`

        Represents a range of memory.

        Parameters:
        :   * **dtype** â The data type.
            * **size** â The size of the memory range in bytes.
            * **base** â The base address of the memory range.

        \_\_init\_\_(*dtype*, *size*, *base*)[#](#cutlass.cute.struct._MemRangeData.__init__ "Link to this definition")
        :   Initializes a new memory range.

            Parameters:
            :   * **dtype** â The data type.
                * **size** â Size of the memory range in bytes. A size of **0** is accepted, but in that
                  case the range can only be used for its address (e.g. as a partition marker).
                * **base** â The base address of the memory range.

        data\_ptr( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.struct._MemRangeData.data_ptr "Link to this definition")
        :   Returns start pointer to the data in this memory range.

            Returns:
            :   A pointer to the start of the memory range.

            Raises:
            :   **AssertionError** â If the size of the memory range is negative.

        get\_tensor( : *layout*, : *swizzle=None*, : *dtype=None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.struct._MemRangeData.get_tensor "Link to this definition")
        :   Creates a tensor from the memory range.

            Parameters:
            :   * **layout** â The layout of the tensor.
                * **swizzle** â Optional swizzle pattern.
                * **dtype** â Optional data type; defaults to the memory rangeâs data type if not specified.

            Returns:
            :   A tensor representing the memory range.

            Raises:
            :   * **TypeError** â If the layout is incompatible with the swizzle.
                * **AssertionError** â If the size of the memory range is not greater than zero.

    *class* \_AlignMeta(*name*, *bases*, *dct*)[#](#cutlass.cute.struct._AlignMeta "Link to this definition")
    :   Bases: `type`

        Aligns the given object by setting its alignment attribute.

        Parameters:
        :   * **v** â The object to align. Must be a struct, MemRange, or a scalar type.
            * **align** â The alignment value to set.

        Raises:
        :   **TypeError** â If the object is not a struct, MemRange, or a scalar type.

        Variables:
        :   * **\_dtype** â The data type to be aligned.
            * **\_align** â The alignment of the data type.

        \_dtype*: Any | None* *= None*[#](#cutlass.cute.struct._AlignMeta._dtype "Link to this definition")

        \_align*: int | None* *= None*[#](#cutlass.cute.struct._AlignMeta._align "Link to this definition")

        *property* dtype[#](#cutlass.cute.struct._AlignMeta.dtype "Link to this definition")

        *property* align[#](#cutlass.cute.struct._AlignMeta.align "Link to this definition")

    *class* Align[#](#cutlass.cute.struct.Align "Link to this definition")
    :   Bases: `object`

        Aligns the given type by Align[T, alignment].

        \_dtype*: Any | None* *= None*[#](#cutlass.cute.struct.Align._dtype "Link to this definition")

        \_align*: int | None* *= None*[#](#cutlass.cute.struct.Align._align "Link to this definition")

    *class* \_ScalarData(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.struct._ScalarData "Link to this definition")
    :   Bases: `register_value_caster`

        Represents a scalar value at a given pointer location in memory.

        This class provides utility methods to get a scalar pointer.
        It wraps a pointer to a scalar element and enables element-wise memory operations.

        Variables:
        :   **\_ptr** â The underlying pointer to the scalar value.

        \_\_init\_\_(*ptr*)[#](#cutlass.cute.struct._ScalarData.__init__ "Link to this definition")

        to\_llvm\_ptr( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.\_mlir.ir.Value[#](#cutlass.cute.struct._ScalarData.to_llvm_ptr "Link to this definition")

        *property* ptr*: cutlass.cute.typing.Pointer*[#](#cutlass.cute.struct._ScalarData.ptr "Link to this definition")
        :   Get the underlying pointer.

            Returns:
            :   The pointer to the scalar value.

            Return type:
            :   Pointer

        *property* dtype*: cutlass.cute.typing.Numeric*[#](#cutlass.cute.struct._ScalarData.dtype "Link to this definition")
        :   Get the data type of the scalar value.

            Returns:
            :   The numeric data type of the underlying pointer.

            Return type:
            :   Numeric

        *property* value[#](#cutlass.cute.struct._ScalarData.value "Link to this definition")
        :   Get the raw MLIR value of the underlying pointer.

            Deprecated since version Using: `struct.scalar` as pointer is deprecated.
            Use explicit `struct.scalar.ptr` for pointer instead.

            Returns:
            :   The MLIR value of the underlying pointer.

            Return type:
            :   ir.Value

    *static* \_is\_scalar\_type(*dtype*)[#](#cutlass.cute.struct._is_scalar_type "Link to this definition")
    :   Checks if the given type is a scalar numeric type.

        Parameters:
        :   **dtype** â The type to check.

        Returns:
        :   True if the type is a subclass of Numeric, False otherwise.

    \_\_init\_\_(*cls*)[#](#cutlass.cute.struct.__init__ "Link to this definition")
    :   Initializes a new struct decorator instance.

        Parameters:
        :   **cls** â The class representing the structured data type.

        Raises:
        :   **TypeError** â If the struct is empty.

    size\_in\_bytes() → int[#](#cutlass.cute.struct.size_in_bytes "Link to this definition")
    :   Returns the size of the struct in bytes.

        Returns:
        :   The size of the struct.

    *static* align\_offset(*offset*, *align*)[#](#cutlass.cute.struct.align_offset "Link to this definition")
    :   Return the round-up offset up to the next multiple of align.

cutlass.cute.E(*mode: int | List[int]*) → ScaledBasis[#](#cutlass.cute.E "Link to this definition")
:   Create a unit ScaledBasis element with the specified mode.

    This function creates a ScaledBasis with value 1 and the given mode.
    The mode represents the coordinate axis or dimension in the layout.

    Parameters:
    :   **mode** (*Union**[**int**,* *List**[**int**]**]*) â The mode (dimension) for the basis element, either a single integer or a list of integers

    Returns:
    :   A ScaledBasis with value 1 and the specified mode

    Return type:
    :   ScaledBasis

    Raises:
    :   **TypeError** â If mode is not an integer or a list

    **Examples:**

    ```python
    # Create a basis element for the first dimension (mode 0)
    e0 = E(0)

    # Create a basis element for the second dimension (mode 1)
    e1 = E(1)

    # Create a basis element for a hierarchical dimension
    e_hier = E([0, 1])
    ```

cutlass.cute.get\_divisibility(*x: int | cutlass.cute.typing.Integer*) → int[#](#cutlass.cute.get_divisibility "Link to this definition")

cutlass.cute.is\_static(*x: Any*) → bool[#](#cutlass.cute.is_static "Link to this definition")
:   Check if a value is statically known at compile time.

    In CuTe, static values are those whose values are known at compile time,
    as opposed to dynamic values which are only known at runtime.

    This function checks if a value is static by recursively traversing its type hierarchy
    and checking if all components are static.

    Static values include:
    - Python literals (bool, int, float, None)
    - Static ScaledBasis objects
    - Static ComposedLayout objects
    - Static IR types
    - Tuples containing only static values

    Dynamic values include:
    - Numeric objects (representing runtime values)
    - Dynamic expressions
    - Any tuple containing dynamic values

    Parameters:
    :   **x** (*Any*) â The value to check

    Returns:
    :   True if the value is static, False otherwise

    Return type:
    :   bool

    Raises:
    :   **TypeError** â If an unsupported type is provided

cutlass.cute.has\_underscore(*a: cutlass.cute.typing.XTuple*) → bool[#](#cutlass.cute.has_underscore "Link to this definition")

cutlass.cute.pretty\_str(*arg*) → str[#](#cutlass.cute.pretty_str "Link to this definition")
:   Constructs a concise readable pretty string.

cutlass.cute.printf(*\*args*, *loc=None*, *ip=None*, *end='\n'*) → None[#](#cutlass.cute.printf "Link to this definition")
:   Print one or more values with optional formatting.

    This function provides printf-style formatted printing capabilities. It can print values directly
    or format them using C-style format strings. The function supports printing various types including
    layouts, numeric values, tensors, and other CuTe objects.

    The function accepts either:
    1. A list of values to print directly
    2. A format string followed by values to format

    Parameters:
    :   * **args** (*Any*) â Variable length argument list containing either:
          - One or more values to print directly
          - A format string followed by values to format
        * **loc** (*Optional**[**Location**]*) â Source location information for debugging, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for code generation, defaults to None
        * **end** (*Optional**[**str**]*) â Suffix for the printed value, defaults to newline

    Raises:
    :   * **ValueError** â If no arguments are provided
        * **TypeError** â If an unsupported argument type is passed

    **Examples:**

    Direct printing of values:

    ```python
    a = cute.make_layout(shape=(10, 10), stride=(10, 1))
    b = cutlass.Float32(1.234)
    cute.printf(a, b)  # Prints values directly
    ```

    Formatted printing:

    ```python
    # Using format string with generic format specifiers
    cute.printf("a={}, b={}", a, b)

    # Using format string with C-style format specifiers
    cute.printf("a={}, b=%.2f", a, b)
    ```

cutlass.cute.front(*input*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.front "Link to this definition")
:   Recursively get the first element of input.

    This function traverses a hierarchical structure (like a layout or tensor)
    and returns the first element at the deepest level. Itâs particularly useful
    for accessing the first stride value in a layout to determine properties like
    majorness.

    Parameters:
    :   * **input** (*Union**[**Tensor**,* *Layout**,* *Stride**]*) â The hierarchical structure to traverse
        * **loc** (*source location**,* *optional*) â Source location where itâs called, defaults to None
        * **ip** (*insertion pointer**,* *optional*) â Insertion pointer for IR generation, defaults to None

    Returns:
    :   The first element at the deepest level of the input structure

    Return type:
    :   Union[int, float, bool, ir.Value]

cutlass.cute.is\_major( : *mode*, : *stride: cutlass.cute.typing.Stride*, : *\**, : *loc=None*, : *ip=None*, ) → bool[#](#cutlass.cute.is_major "Link to this definition")
:   Check whether a mode in stride is the major mode.

cutlass.cute.assume(*src*, *divby=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.assume "Link to this definition")

cutlass.cute.make\_swizzle(*b*, *m*, *s*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_swizzle "Link to this definition")

cutlass.cute.static(*value*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.static "Link to this definition")

cutlass.cute.get\_leaves(*value*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.get_leaves "Link to this definition")

cutlass.cute.depth( : *a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, ) → int[#](#cutlass.cute.depth "Link to this definition")
:   Returns the depth (nesting level) of a tuple, layout, or tensor.

    The depth of a tuple is the maximum depth of its elements plus 1.
    For an empty tuple, the depth is 1. For layouts and tensors, the depth
    is determined by the depth of their shape. For non-tuple values (e.g., integers),
    the depth is considered 0.

    Parameters:
    :   **a** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**,* *Any**]*) â The object whose depth is to be determined

    Returns:
    :   The depth of the input object

    Return type:
    :   int

    **Example:**

    ```python
    >>> depth(1)
    0
    >>> depth((1, 2))
    1
    >>> depth(((1, 2), (3, 4)))
    2
    ```

cutlass.cute.rank( : *a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mode: List[int] = []*, ) → int[#](#cutlass.cute.rank "Link to this definition")
:   Returns the rank (dimensionality) of a tuple, layout, or tensor.

    The rank of a tuple is its length. For layouts and tensors, the rank is
    determined by the rank of their shape. For non-tuple values (e.g., integers),
    the rank is considered 1 for convenience.

    Parameters:
    :   **a** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**,* *Any**]*) â The object whose rank is to be determined

    Returns:
    :   The rank of the input object

    Return type:
    :   int

    This function is used in layout algebra to determine the dimensionality
    of tensors and layouts for operations like slicing and evaluation.

cutlass.cute.is\_congruent( : *a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, : *b: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, ) → bool[#](#cutlass.cute.is_congruent "Link to this definition")
:   Returns whether a is congruent to b.

    Congruence is an equivalence relation between hierarchical structures.

    Two objects are congruent if:
    \* They have the same rank, AND
    \* They are both non-tuple values, OR
    \* They are both tuples AND all corresponding elements are congruent.

    Congruence requires type matching at each level â scalar values match with
    scalar values, and tuples match with tuples of the same rank.

    Parameters:
    :   * **a** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â First object to compare
        * **b** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â Second object to compare

    Returns:
    :   True if a and b are congruent, False otherwise

    Return type:
    :   bool

cutlass.cute.is\_weakly\_congruent( : *a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, : *b: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, ) → bool[#](#cutlass.cute.is_weakly_congruent "Link to this definition")
:   Returns whether a is weakly congruent to b.

    Weak congruence is a partial order on hierarchical structures.

    Object X is weakly congruent to object Y if:
    \* X is a non-tuple value, OR
    \* X and Y are both tuples of the same rank AND all corresponding elements are weakly congruent.

    Weak congruence allows scalar values to match with tuples, making it useful
    for determining whether an object has a hierarchical structure âup toâ another.

    Parameters:
    :   * **a** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â First object to compare
        * **b** (*Union**[**XTuple**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â Second object to compare

    Returns:
    :   True if a and b are weakly congruent, False otherwise

    Return type:
    :   bool

cutlass.cute.get(*input*, *mode: List[int]*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.get "Link to this definition")
:   Extract a specific element or sub-layout from a layout or tuple.

    This function recursively traverses the input according to the mode indices,
    extracting the element at the specified path. For layouts, this operation
    corresponds to extracting a specific sub-layout.

    Parameters:
    :   * **input** (*Layout**,* *ComposedLayout**,* *tuple*) â The input layout or tuple to extract from
        * **mode** (*List**[**int**]*) â Indices specifying the path to traverse for extraction
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The extracted element or sub-layout

    Return type:
    :   Layout, ComposedLayout, or element type

    Raises:
    :   * **ValueError** â If any index in mode is out of range
        * **TypeError** â If mode contains non-integer elements or if input has unsupported type

    Postcondition:
    :   `get(t, mode=find(x,t)) == x if find(x,t) != None else True`

    **Examples:**

    ```python
    layout = make_layout(((4, 8), (16, 1), 8), stride=((1, 4), (32, 0), 512))
    sub_layout = get(layout, mode=[0, 1])   # 8:4
    sub_layout = get(layout, mode=[1])      # (16, 1):(32, 0)
    ```

cutlass.cute.select(*input*, *mode: List[int]*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.select "Link to this definition")
:   Select modes from input.

    Parameters:
    :   * **input** (*Layout**,* *ComposedLayout**,* *tuple*) â Input to select from
        * **mode** (*List**[**int**]*) â Indices specifying which dimensions or elements to select
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   A new instance with selected dimensions/elements

    Return type:
    :   Layout, ComposedLayout, tuple

    Raises:
    :   * **ValueError** â If any index in mode is out of range
        * **TypeError** â If the input type is invalid

    **Examples:**

    ```python
    # Select specific dimensions from a layout
    layout = make_layout((4, 8, 16), stride=(32, 4, 1))
    selected = select(layout, mode=[0, 2])  # Select mode 0 and mode 2
    # Result: (4, 16):(32, 1)

    # Select elements from a tuple
    t = (1, 2, 3, 4, 5)
    selected = select(t, mode=[0, 2, 4])  # Select mode 0, mode 2, and mode 4
    # Result: (1, 3, 5)
    ```

cutlass.cute.group\_modes( : *input*, : *begin: int*, : *end: int | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.group_modes "Link to this definition")
:   Group modes of a hierarchical tuple or layout into a single mode.

    This function groups a range of modes from the input object into a single mode,
    creating a hierarchical structure. For tuples, it creates a nested tuple containing
    the specified range of elements. For layouts and other CuTe objects, it creates
    a hierarchical representation where the specified modes are grouped together.

    Parameters:
    :   * **input** (*Layout**,* *ComposedLayout**,* *tuple**,* *Shape**,* *Stride**,* *etc.*) â Input object to group modes from (layout, tuple, etc.)
        * **beg** (*int*) â Beginning index of the range to group (inclusive)
        * **end** (*int*) â Ending index of the range to group (exclusive)
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   A new object with the specified modes grouped

    Return type:
    :   Same type as input with modified structure

    **Examples:**

    ```python
    # Group modes in a tuple
    t = (2, 3, 4, 5)
    grouped = group_modes(t, 1, 3)  # (2, (3, 4), 5)

    # Group modes in a layout
    layout = make_layout((2, 3, 4, 5))
    grouped_layout = group_modes(layout, 1, 3)  # Layout with shape (2, (3, 4), 5)

    # Group modes in a shape
    shape = make_shape(2, 3, 4, 5)
    grouped_shape = group_modes(shape, 0, 2)  # Shape ((2, 3), 4, 5)
    ```

cutlass.cute.slice\_(*src*, *coord: cutlass.cute.typing.Coord*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.slice_ "Link to this definition")
:   Perform a slice operation on a source object using the given coordinate.

    This function implements CuTeâs slicing operation which extracts a subset of elements
    from a source object (tensor, layout, etc.) based on a coordinate pattern. The slice
    operation preserves the structure of the source while selecting specific elements.

    Parameters:
    :   * **src** (*Union**[**Tensor**,* *Layout**,* *IntTuple**,* *Value**]*) â Source object to be sliced (tensor, layout, tuple, etc.)
        * **coord** (*Coord*) â Coordinate pattern specifying which elements to select
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   A new object containing the sliced elements

    Return type:
    :   Union[Tensor, Layout, IntTuple, tuple]

    Raises:
    :   **ValueError** â If the coordinate pattern is incompatible with source

    **Examples:**

    ```python
    # Layout slicing
    layout = make_layout((4,4))

    # Select 1st index of first mode and keep all elements in second mode
    sub_layout = slice_(layout, (1, None))
    ```

    ```python
    # Basic tensor slicing
    tensor = make_tensor(...)           # Create a 2D tensor

    # Select 1st index of first mode and keep all elements in second mode
    sliced = slice_(tensor, (1, None))
    ```

    ```python
    # Select 2nd index of second mode and keep all elements in first mode
    sliced = slice_(tensor, (None, 2))
    ```

    Note

    * None represents keeping all elements in that mode
    * Slicing preserves the layout/structure of the original object
    * Can be used for:
      \* Extracting sub-tensors/sub-layouts
      \* Creating views into data
      \* Selecting specific patterns of elements

cutlass.cute.prepend( : *input*, : *elem*, : *up\_to\_rank: int | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.prepend "Link to this definition")
:   Extend input to rank up\_to\_rank by prepending elem in front of input.

    This function extends the input object by prepending elements to reach a desired rank.
    It supports various CuTe types including shapes, layouts, tensors etc.

    Parameters:
    :   * **input** (*Union**[**Shape**,* *Stride**,* *Coord**,* *IntTuple**,* *Tile**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â Source to be prepended to
        * **elem** (*Union**[**Shape**,* *Stride**,* *Coord**,* *IntTuple**,* *Tile**,* *Layout**]*) â Element to prepend to input
        * **up\_to\_rank** (*Union**[**None**,* *int**]**,* *optional*) â The target rank after extension, defaults to None
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point, defaults to None

    Returns:
    :   The extended result with prepended elements

    Return type:
    :   Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

    Raises:
    :   * **ValueError** â If up\_to\_rank is less than inputâs current rank
        * **TypeError** â If input or elem has unsupported type

    **Examples:**

    ```python
    # Prepend to a Shape
    shape = (4,4)
    prepend(shape, 2)                   # Returns (2,4,4)

    # Prepend to a Layout
    layout = make_layout((8,8))
    prepend(layout, make_layout((2,)))  # Returns (2,8,8):(1,1,8)

    # Prepend with target rank
    coord = (1,1)
    prepend(coord, 0, up_to_rank=4)     # Returns (0,0,1,1)
    ```

cutlass.cute.append( : *input*, : *elem*, : *up\_to\_rank: int | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.append "Link to this definition")
:   Extend input to rank up\_to\_rank by appending elem to the end of input.

    This function extends the input object by appending elements to reach a desired rank.
    It supports various CuTe types including shapes, layouts, tensors etc.

    Parameters:
    :   * **input** (*Union**[**Shape**,* *Stride**,* *Coord**,* *IntTuple**,* *Tile**,* *Layout**,* *ComposedLayout**,* *Tensor**]*) â Source to be appended to
        * **elem** (*Union**[**Shape**,* *Stride**,* *Coord**,* *IntTuple**,* *Tile**,* *Layout**]*) â Element to append to input
        * **up\_to\_rank** (*Union**[**None**,* *int**]**,* *optional*) â The target rank after extension, defaults to None
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point, defaults to None

    Returns:
    :   The extended result with appended elements

    Return type:
    :   Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

    Raises:
    :   * **ValueError** â If up\_to\_rank is less than inputâs current rank
        * **TypeError** â If input or elem has unsupported type

    **Examples:**

    ```python
    # Append to a Shape
    shape = (4,4)
    append(shape, 2)                   # Returns (4,4,2)

    # Append to a Layout
    layout = make_layout((8,8))
    append(layout, make_layout((2,)))  # Returns (8,8,2):(1,8,1)

    # Append with target rank
    coord = (1,1)
    append(coord, 0, up_to_rank=4)     # Returns (1,1,0,0)
    ```

    Note

    * The function preserves the structure of the input while extending it
    * Can be used to extend tensors, layouts, shapes and other CuTe types
    * When up\_to\_rank is specified, fills remaining positions with elem
    * Useful for tensor reshaping and layout transformations

cutlass.cute.prepend\_ones( : *t: cutlass.cute.typing.Tensor*, : *up\_to\_rank: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.prepend_ones "Link to this definition")

cutlass.cute.append\_ones(*t*, *up\_to\_rank: int | None = None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.append_ones "Link to this definition")

cutlass.cute.repeat\_as\_tuple(*x*, *n*) → tuple[#](#cutlass.cute.repeat_as_tuple "Link to this definition")
:   Creates a tuple with x repeated n times.

    This function creates a tuple by repeating the input value x n times.

    Parameters:
    :   * **x** (*Any*) â The value to repeat
        * **n** (*int*) â Number of times to repeat x

    Returns:
    :   A tuple containing x repeated n times

    Return type:
    :   tuple

    **Examples:**

    ```python
    repeat_as_tuple(1, 1)     # Returns (1,)
    repeat_as_tuple(1, 3)     # Returns (1, 1, 1)
    repeat_as_tuple(None, 4)  # Returns (None, None, None, None)
    ```

cutlass.cute.repeat(*x*, *n*)[#](#cutlass.cute.repeat "Link to this definition")
:   Creates an object by repeating x n times.

    This function creates an object by repeating the input value x n times.
    If n=1, returns x directly, otherwise returns a tuple of x repeated n times.

    Parameters:
    :   * **x** (*Any*) â The value to repeat
        * **n** (*int*) â Number of times to repeat x

    Returns:
    :   x if n=1, otherwise a tuple containing x repeated n times

    Return type:
    :   Union[Any, tuple]

    Raises:
    :   **ValueError** â If n is less than 1

    **Examples:**

    ```python
    repeat(1, 1)     # Returns 1
    repeat(1, 3)     # Returns (1, 1, 1)
    repeat(None, 4)  # Returns (None, None, None, None)
    ```

cutlass.cute.repeat\_like(*x*, *target*)[#](#cutlass.cute.repeat_like "Link to this definition")
:   Creates an object congruent to target and filled with x.

    This function recursively creates a nested tuple structure that matches the structure
    of the target, with each leaf node filled with the value x.

    Parameters:
    :   * **x** (*Any*) â The value to fill the resulting structure with
        * **target** (*Union**[**tuple**,* *Any**]*) â The structure to mimic

    Returns:
    :   A structure matching target but filled with x

    Return type:
    :   Union[tuple, Any]

    **Examples:**

    ```python
    repeat_like(0, (1, 2, 3))      # Returns (0, 0, 0)
    repeat_like(1, ((1, 2), 3))    # Returns ((1, 1), 1)
    repeat_like(2, 5)              # Returns 2
    ```

cutlass.cute.flatten(*a*)[#](#cutlass.cute.flatten "Link to this definition")
:   Flattens a CuTe data structure into a simpler form.

    For tuples, this function flattens the structure into a single-level tuple.
    For layouts, it returns a new layout with flattened shape and stride.
    For tensors, it returns a new tensor with flattened layout.
    For other types, it returns the input unchanged.

    Parameters:
    :   **a** (*Union**[**IntTuple**,* *Coord**,* *Shape**,* *Stride**,* *Layout**,* *Tensor**]*) â The structure to flatten

    Returns:
    :   The flattened structure

    Return type:
    :   Union[tuple, Any]

    **Examples:**

    ```python
    flatten((1, 2, 3))                      # Returns (1, 2, 3)
    flatten(((1, 2), (3, 4)))               # Returns (1, 2, 3, 4)
    flatten(5)                              # Returns 5
    flatten(Layout(shape, stride))          # Returns Layout(flatten(shape), flatten(stride))
    flatten(Tensor(layout))                 # Returns Tensor(flatten(layout))
    ```

cutlass.cute.filter\_zeros(*input*, *\**, *target\_profile=None*, *loc=None*, *ip=None*)[#](#cutlass.cute.filter_zeros "Link to this definition")
:   Filter out zeros from a layout or tensor.

    This function removes zero-stride dimensions from a layout or tensor.
    Refer to [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
    for more layout algebra operations.

    Parameters:
    :   * **input** (*Layout* *or* *Tensor*) â The input layout or tensor to filter
        * **target\_profile** (*Stride**,* *optional*) â Target stride profile for the filtered result, defaults to None
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The filtered layout or tensor with zeros removed

    Return type:
    :   Layout or Tensor

    Raises:
    :   **TypeError** â If input is not a Layout or Tensor

cutlass.cute.filter(*input*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.filter "Link to this definition")
:   Filter a layout or tensor.

    This function filters a layout or tensor according to CuTeâs filtering rules.

    Parameters:
    :   * **input** (*Layout* *or* *Tensor*) â The input layout or tensor to filter
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The filtered layout or tensor

    Return type:
    :   Layout or Tensor

    Raises:
    :   **TypeError** â If input is not a Layout or Tensor

cutlass.cute.size( : *a: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Shape | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, : *mode: List[int] = []*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int[#](#cutlass.cute.size "Link to this definition")
:   Return size of domain of layout or tensor.

    Computes the size (number of elements) in the domain of a layout or tensor.
    For layouts, this corresponds to the shape of the coordinate space.
    See [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)
    for more details on layout domains.

    Parameters:
    :   * **a** (*IntTuple**,* *Shape**,* *Layout**,* *ComposedLayout* *or* *Tensor*) â The input object whose size to compute
        * **mode** (*list* *of* *int**,* *optional*) â List of mode(s) for size calculation. If empty, computes total size, defaults to []
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   Static size of layout or tensor if static, otherwise a Value

    Return type:
    :   int or Value

    Raises:
    :   **ValueError** â If mode contains non-integer elements

cutlass.cute.shape\_div( : *lhs: cutlass.cute.typing.Shape*, : *rhs: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Shape[#](#cutlass.cute.shape_div "Link to this definition")
:   Perform element-wise division of shapes.

    This function performs element-wise division between two shapes.

    Parameters:
    :   * **lhs** (*Shape*) â Left-hand side shape
        * **rhs** (*Shape*) â Right-hand side shape
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The result of element-wise division

    Return type:
    :   Shape

cutlass.cute.ceil\_div( : *input: cutlass.cute.typing.Shape*, : *tiler: cutlass.cute.typing.Tiler*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Shape[#](#cutlass.cute.ceil_div "Link to this definition")
:   Compute the ceiling division of a target shape by a tiling specification.

    This function computes the number of tiles required to cover the target domain.
    It is equivalent to the second mode of zipped\_divide(input, tiler).

    Parameters:
    :   * **input** (*Shape*) â A tuple of integers representing the dimensions of the target domain.
        * **tiler** (*Union**[**Layout**,* *Shape**,* *Tile**]*) â The tiling specification.
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions.

    Returns:
    :   A tuple of integers representing the number of tiles required along each dimension,
        i.e. the result of the ceiling division of the input dimensions by the tiler dimensions.

    Return type:
    :   Shape

    Example:

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        input = (10, 6)
        tiler = (3, 4)
        result = cute.ceil_div(input, tiler)
        print(result)  # Outputs: (4, 2)
    ```

cutlass.cute.round\_up( : *a: cutlass.cute.typing.IntTuple*, : *b: cutlass.cute.typing.IntTuple*, ) → cutlass.cute.typing.IntTuple[#](#cutlass.cute.round_up "Link to this definition")
:   Rounds up elements of a using elements of b.

cutlass.cute.make\_layout( : *shape: cutlass.cute.typing.Shape*, : *\**, : *stride: cutlass.cute.typing.Stride | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.make_layout "Link to this definition")
:   Create a CuTe Layout object from shape and optional stride information.

    A Layout in CuTe represents the mapping between logical and physical coordinates of a tensor.
    This function creates a Layout object that defines how tensor elements are arranged in memory.

    Parameters:
    :   * **shape** (*Shape*) â Shape of the layout defining the size of each mode
        * **stride** (*Union**[**Stride**,* *None**]*) â Optional stride values for each mode, defaults to None
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   A new Layout object with the specified shape and stride

    Return type:
    :   Layout

    **Examples:**

    ```python
    # Create a 2D compact left-most layout with shape (4,4)
    layout = make_layout((4,4))                     # compact left-most layout

    # Create a left-most layout with custom strides
    layout = make_layout((4,4), stride=(1,4))       # left-most layout with strides (1,4)

    # Create a layout for a 3D tensor
    layout = make_layout((32,16,8))                 # left-most layout

    # Create a layout with custom strides
    layout = make_layout((2,2,2), stride=(4,1,2))   # layout with strides (4,1,2)
    ```

    Note

    * If stride is not provided, a default compact left-most stride is computed based on the shape
    * The resulting layout maps logical coordinates to physical memory locations
    * The layout object can be used for tensor creation and memory access patterns
    * Strides can be used to implement:
      \* Row-major vs column-major layouts
      \* Padding and alignment
      \* Blocked/tiled memory arrangements
      \* Interleaved data formats
    * Stride is keyword only argument to improve readability, e.g.
      \* make\_layout((3,4), (1,4)) can be confusing with make\_layout(((3,4), (1,4)))
      \* make\_layout((3,4), stride=(1,4)) is more readable

cutlass.cute.make\_identity\_layout( : *shape: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.make_identity_layout "Link to this definition")
:   Create an identity layout with the given shape.

    An identity layout maps logical coordinates directly to themselves without any transformation.
    This is equivalent to a layout with stride (1@0,1@1,â¦,1@(N-1)).

    Parameters:
    :   * **shape** (*Shape*) â The shape of the layout
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   A new identity Layout object with the specified shape

    Return type:
    :   Layout

    **Examples:**

    ```python
    # Create a 2D identity layout with shape (4,4)
    layout = make_identity_layout((4,4))     # stride=(1@0,1@1)

    # Create a 3D identity layout
    layout = make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)
    ```

    Note

    * An identity layout is a special case where each coordinate maps to itself
    * Useful for direct coordinate mapping without any transformation

cutlass.cute.make\_ordered\_layout( : *shape: cutlass.cute.typing.Shape*, : *order: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.make_ordered_layout "Link to this definition")
:   Create a layout with a specific ordering of dimensions.

    This function creates a layout where the dimensions are ordered according to the
    specified order parameter, allowing for custom dimension ordering in the layout.

    Parameters:
    :   * **shape** (*Shape*) â The shape of the layout
        * **order** (*Shape*) â The ordering of dimensions
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   A new Layout object with the specified shape and dimension ordering

    Return type:
    :   Layout

    **Examples:**

    ```python
    # Create a row-major layout
    layout = make_ordered_layout((4,4), order=(1,0))

    # Create a column-major layout
    layout = make_ordered_layout((4,4), order=(0,1))         # stride=(1,4)

    # Create a layout with custom dimension ordering for a 3D tensor
    layout = make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)
    ```

    Note

    * The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
    * For a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
    * The length of order must match the rank of the shape

cutlass.cute.make\_layout\_like( : *input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.make_layout_like "Link to this definition")

cutlass.cute.make\_composed\_layout( : *inner*, : *offset: cutlass.cute.typing.IntTuple*, : *outer: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.ComposedLayout[#](#cutlass.cute.make_composed_layout "Link to this definition")
:   Create a composed layout by composing an inner transformation with an outer layout.

    A composed layout applies a sequence of transformations
    to coordinates. The composition is defined as (inner â offset â outer), where the operations
    are applied from right to left.

    Parameters:
    :   * **inner** (*Union**[**Layout**,* [*Swizzle*](#cutlass.cute.Swizzle "cutlass.cute.Swizzle")*]*) â The inner transformation (can be a Layout or Swizzle)
        * **offset** (*IntTuple*) â An integral offset applied between transformations
        * **outer** (*Layout*) â The outer (right-most) layout that is applied first
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   A new ComposedLayout representing the composition

    Return type:
    :   ComposedLayout

    **Examples:**

    ```python
    # Create a basic layout
    inner = make_layout(...)
    outer = make_layout((4,4), stride=(E(0), E(1)))

    # Create a composed layout with an offset
    composed = make_composed_layout(inner, (2,0), outer)
    ```

    Note

    * The composition applies transformations in the order: outer â offset â inner
    * The stride divisibility condition must be satisfied for valid composition
    * Certain compositions (like Swizzle with scaled basis) are invalid and will raise errors
    * Composed layouts inherit many properties from the outer layout

cutlass.cute.cosize( : *a: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor*, : *mode: List[int] = []*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.cosize "Link to this definition")
:   Return size of codomain of layout or tensor. Return static value if type is static.

    For a layout `L = S:D` where `S` is the shape and `D` is the stride, the codomain size is the
    minimum size needed to store all possible offsets generated by the layout. This is calculated
    by taking the maximum offset plus 1.

    For example, given a layout `L = (4,(3,2)):(2,(8,1))`:
    :   * Shape `S = (4,(3,2))`
        * Stride `D = (2,(8,1))`
        * Maximum offset = `2*(4-1) + 8*(3-1) + 1*(2-1) = 6 + 16 + 1 = 23`
        * Therefore `cosize(L) = 24`

    **Examples:**

    ```python
    L = cute.make_layout((4,(3,2)), stride=(2,(8,1))) # L = (4,(3,2)):(2,(8,1))
    print(cute.cosize(L))  # => 24
    ```

    Parameters:
    :   * **a** (*Union**[**Layout**,* *ComposedLayout**,* *Tensor**]*) â Layout, ComposedLayout, or Tensor object
        * **mode** (*List**[**int**]**,* *optional*) â List of mode(s) for cosize calculation. If empty, calculates over all modes.
          If specified, calculates cosize only for the given modes.
        * **loc** (*optional*) â Location information for diagnostics, defaults to None
        * **ip** (*optional*) â Instruction pointer for diagnostics, defaults to None

    Returns:
    :   Static size of layout or tensor (fast fold) if static, or a dynamic Value

    Return type:
    :   Union[int, Value]

cutlass.cute.size\_in\_bytes( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | None*, : *\**, : *loc=None*, : *ip=None*, ) → int | cutlass.cute.typing.Integer[#](#cutlass.cute.size_in_bytes "Link to this definition")
:   Calculate the size in bytes based on its data type and layout. The result is rounded up to the nearest byte.

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) â The DSL numeric data type
        * **layout** (*Layout**,* *optional*) â The layout of the elements. If None, the function returns 0
        * **loc** (*optional*) â Location information for diagnostics, defaults to None
        * **ip** (*optional*) â Instruction pointer for diagnostics, defaults to None

    Returns:
    :   The total size in bytes. Returns 0 if the layout is None

    Return type:
    :   int

cutlass.cute.coalesce( : *input*, : *\**, : *target\_profile: cutlass.cute.typing.Coord | None = None*, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.coalesce "Link to this definition")

cutlass.cute.crd2idx( : *coord: cutlass.cute.typing.Coord*, : *layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.crd2idx "Link to this definition")
:   Convert a multi-dimensional coordinate into a value using the specified layout.

    This function computes the inner product of the flattened coordinate and stride:

    > index = sum(flatten(coord)[i] \* flatten(stride)[i] for i in range(len(coord)))

    Parameters:
    :   * **coord** (*Coord*) â A tuple or list representing the multi-dimensional coordinate
          (e.g., (i, j) for a 2D layout).
        * **layout** (*Layout* *or* *ComposedLayout*) â A layout object that defines the memory storage layout, including shape and stride,
          used to compute the inner product.
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions.

    Returns:
    :   The result of applying the layout transformation to the provided coordinate.

    Return type:
    :   Any type that the layout maps to

    **Example:**

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        L = cute.make_layout((5, 4), stride=(4, 1))
        idx = cute.crd2idx((2, 3), L)
        # Computed as: 2 * 4 + 3 = 11
        print(idx)
    foo()  # Expected output: 11
    ```

cutlass.cute.idx2crd(*idx*, *shape*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.idx2crd "Link to this definition")
:   Convert a linear index back into a multi-dimensional coordinate using the specified layout.

    Mapping from a linear index to the corresponding multi-dimensional coordinate in the layoutâs coordinate space.
    It essentially âunfoldsâ a linear index into its constituent coordinate components.

    Parameters:
    :   * **idx** (*: int/Integer/Tuple*) â The linear index to convert back to coordinates.
        * **shape** (*Shape*) â Shape of the layout defining the size of each mode
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions.

    Returns:
    :   The result of applying the layout transformation to the provided coordinate.

    Return type:
    :   Coord

    **Examples:**

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        coord = cute.idx2crd(11, (5, 4))
        # idx2crd is always col-major
        # For shape (m, n, l, ...), coord = (idx % m, idx // m % n, idx // m // n % l, ...
        # Computed as: (11 % 5, 11 // 5 % 4) = (1, 2)
        print(coord)

    foo()  # Expected output: (1, 2)
    ```

cutlass.cute.recast\_layout( : *new\_type\_bits: int*, : *old\_type\_bits: int*, : *src\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.recast_layout "Link to this definition")
:   Recast a layout from one data type to another.

    Parameters:
    :   * **new\_type\_bits** (*int*) â The new data type bits
        * **old\_type\_bits** (*int*) â The old data type bits
        * **src\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) â The layout to recast
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions.

    Returns:
    :   The recast layout

    Return type:
    :   Layout or ComposedLayout

    **Example:**

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        # Create a layout
        L = cute.make_layout((2, 3, 4))
        # Recast the layout to a different data type
        L_recast = cute.recast_layout(16, 8, L)
        print(L_recast)
    foo()  # Expected output: (2, 3, 4)
    ```

cutlass.cute.slice\_and\_offset(*coord*, *src*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.slice_and_offset "Link to this definition")

cutlass.cute.recast\_ptr( : *ptr: cutlass.cute.typing.Pointer*, : *swizzle\_=None*, : *dtype: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.recast_ptr "Link to this definition")

cutlass.cute.make\_ptr( : *dtype: Type[cutlass.cute.typing.Numeric] | None*, : *value*, : *mem\_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic*, : *\**, : *assumed\_align=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.cute.make_ptr "Link to this definition")

cutlass.cute.composition( : *lhs*, : *rhs: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape | cutlass.cute.typing.Tile*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.composition "Link to this definition")
:   Compose two layout representations using the CuTe layout algebra.

    Compose a left-hand layout (or tensor) with a right-hand operand into a new layout R, such that
    for every coordinate c in the domain of the right-hand operand, the composed layout satisfies:

    > R(c) = A(B(c))

    where A is the left-hand operand provided as `lhs` and B is the right-hand operand provided as
    `rhs`. In this formulation, B defines the coordinate domain while A applies its transformation to
    Bâs output, and the resulting layout R inherits the stride and shape adjustments from A.

    Satisfies:
    :   cute.shape(cute.composition(lhs, rhs)) is compatible with cute.shape(rhs)

    Parameters:
    :   * **lhs** (*Layout* *or* *Tensor*) â The left-hand operand representing the transformation to be applied.
        * **rhs** (*Layout**,* *Shape**, or* *Tile**, or* *int* *or* *tuple*) â The right-hand operand defining the coordinate domain. If provided as an int or tuple,
          it will be converted to a tile layout.
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions.

    Returns:
    :   A new composed layout R, such that for all coordinates c in the domain of `rhs`,
        R(c) = lhs(rhs(c)).

    Return type:
    :   Layout or Tensor

    **Example:**

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        # Create a layout that maps (i,j) to i*4 + j
        L1 = cute.make_layout((2, 3), stride=(4, 1))
        # Create a layout that maps (i,j) to i*3 + j
        L2 = cute.make_layout((3, 4), stride=(3, 1))
        # Compose L1 and L2
        L3 = cute.composition(L1, L2)
        # L3 now maps coordinates through L2 then L1
    ```

cutlass.cute.complement( : *input: cutlass.cute.typing.Layout*, : *cotarget: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.complement "Link to this definition")
:   Compute the complement layout of the input layout with respect to the cotarget.

    The complement of a layout A with respect to cotarget n is a layout A\* such that
    for every k in Z\_n and c in the domain of A, there exists a unique c\* in the domain
    of A\* where k = A(c) + A\*(c\*).

    This operation is useful for creating layouts that partition a space in complementary ways,
    such as row and column layouts that together cover a matrix.

    Parameters:
    :   * **input** (*Layout*) â The layout to compute the complement of
        * **cotarget** (*Union**[**Layout**,* *Shape**]*) â The target layout or shape that defines the codomain
        * **loc** (*optional*) â Optional location information for IR diagnostics
        * **ip** (*optional*) â Optional instruction pointer or context for underlying IR functions

    Returns:
    :   The complement layout

    Return type:
    :   Layout

    **Example:**

    ```python
    import cutlass.cute as cute
    @cute.jit
    def foo():
        # Create a right-major layout for a 4x4 matrix
        row_layout = cute.make_layout((4, 4), stride=(4, 1))
        # Create a left-major layout that complements the row layout
        col_layout = cute.complement(row_layout, 16)
        # The two layouts are complementary under 16
    ```

cutlass.cute.right\_inverse( : *input: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.right_inverse "Link to this definition")

cutlass.cute.left\_inverse( : *input: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.left_inverse "Link to this definition")

cutlass.cute.logical\_product( : *block*, : *tiler: cutlass.cute.typing.Tile*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.logical_product "Link to this definition")

cutlass.cute.zipped\_product( : *block*, : *tiler: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.zipped_product "Link to this definition")

cutlass.cute.tiled\_product( : *block*, : *tiler: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.tiled_product "Link to this definition")

cutlass.cute.flat\_product( : *block*, : *tiler: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.flat_product "Link to this definition")

cutlass.cute.raked\_product( : *block*, : *tiler: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.raked_product "Link to this definition")

cutlass.cute.blocked\_product( : *block*, : *tiler: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.blocked_product "Link to this definition")

cutlass.cute.logical\_divide( : *target*, : *tiler: cutlass.cute.typing.Tiler*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.logical_divide "Link to this definition")

cutlass.cute.zipped\_divide( : *target*, : *tiler: cutlass.cute.typing.Tiler*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.zipped_divide "Link to this definition")
:   `zipped_divide` is `logical_divide` with Tiler modes and Rest modes gathered together: `(Tiler,Rest)`

    * When Tiler is Layout, this has no effect as `logical_divide` results in the same.
    * When Tiler is `Tile` (nested tuple of `Layout`) or `Shape`, this zips modes into standard form
      `((BLK_A,BLK_B),(a,b,x,y))`

    For example, if `target` has shape `(s, t, r)` and `tiler` has shape `(BLK_A, BLK_B)`,
    then the result will have shape `((BLK_A, BLK_B), (ceil_div(s, BLK_A), ceil_div(t, BLK_B), r))`.

    Parameters:
    :   * **target** (*Layout* *or* *Tensor*) â The layout or tensor to partition.
        * **tiler** (*Tiler*) â The tiling specification (can be a Layout, Shape, Tile).
        * **loc** (*optional*) â Optional MLIR IR location information.
        * **ip** (*optional*) â Optional MLIR IR insertion point.

    Returns:
    :   A zipped (partitioned) version of the target.

    Return type:
    :   Layout or Tensor

    **Example:**

    ```python
    layout = cute.make_layout((128, 64), stride=(64, 1))
    tiler = (8, 8)
    result = cute.zipped_divide(layout, tiler)  # result shape: ((8, 8), (16, 8))
    ```

cutlass.cute.tiled\_divide( : *target*, : *tiler: cutlass.cute.typing.Tiler*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.tiled_divide "Link to this definition")

cutlass.cute.flat\_divide( : *target*, : *tiler: cutlass.cute.typing.Tile*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.flat_divide "Link to this definition")

cutlass.cute.max\_common\_layout( : *a: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *b: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.max_common_layout "Link to this definition")

cutlass.cute.max\_common\_vector( : *a: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *b: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → int[#](#cutlass.cute.max_common_vector "Link to this definition")

cutlass.cute.tile\_to\_shape( : *atom*, : *trg\_shape: cutlass.cute.typing.Shape*, : *order: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.tile_to_shape "Link to this definition")

cutlass.cute.local\_partition( : *target: cutlass.cute.typing.Tensor*, : *tiler: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape*, : *index: int | cutlass.cute.typing.Numeric*, : *proj: cutlass.cute.typing.XTuple = 1*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.local_partition "Link to this definition")

cutlass.cute.local\_tile( : *input: cutlass.cute.typing.Tensor*, : *tiler: cutlass.cute.typing.Tiler*, : *coord: cutlass.cute.typing.Coord*, : *proj: cutlass.cute.typing.XTuple | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.local_tile "Link to this definition")
:   Partition a tensor into tiles using a tiler and extract a single tile at the provided coordinate.

    The `local_tile` operation applies a `zipped_divide` to split the `input` tensor by the `tiler`
    and then slices out a single tile using the provided coord. This is commonly used for extracting block-,
    thread-, or CTA-level tiles for parallel operations.

    \[\text{local\_tile}(input, tiler, coord) = \text{zipped\_divide}(input, tiler)[coord]\]

    This function corresponds to the CUTE/C++ local\_tile utility:
    <https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#local-tile>

    Parameters:
    :   * **input** (*Tensor*) â The input tensor to partition into tiles.
        * **tiler** (*Tiler*) â The tiling specification (can be a Layout, Shape, Tile).
        * **coord** (*Coord*) â The coordinate to select within the remainder (ârestâ) modes after tiling.
          This selects which tile to extract.
        * **proj** (*XTuple**,* *optional*) â (Optional) Projection onto tiling modes; specify to project out unused tiler modes,
          e.g., when working with projections of tilers in multi-mode partitioning.
          Default is None for no projection.
        * **loc** (*Any**,* *optional*) â (Optional) MLIR location, for diagnostic/debugging.
        * **ip** (*Any**,* *optional*) â (Optional) MLIR insertion point, used in IR building context.

    Returns:
    :   A new tensor representing the local tile selected at the given coordinate.

    Return type:
    :   Tensor

    **Examples**

    1. Tiling a 2D tensor and extracting a tile:

       > ```python
       > # input: (16, 24)
       > tensor : cute.Tensor
       > tiler = (2, 4)
       > coord = (1, 1)
       >
       > # output: (8, 6)
       > # - zipped_divide(tensor, tiler)     -> ((2, 4), (8, 6))
       > # - local_tile(tensor, tiler, coord) -> (8, 6)
       > result = cute.local_tile(tensor, tiler=tiler, coord=coord)
       > ```
    2. Using a stride projection for specialized tiling:

       > ```python
       > # input: (16, 24)
       > tensor : cute.Tensor
       > tiler = (2, 2, 4)
       > coord = (0, 1, 1)
       > proj = (1, None, 1)
       >
       > # output: (8, 6)
       > # projected_tiler: (2, 4)
       > # projected_coord: (0, 1)
       > # - zipped_divide(tensor, projected_tiler)               -> ((2, 4), (8, 6))
       > # - local_tile(tensor, projected_tiler, projected_coord) -> (8, 6)
       > result = cute.local_tile(tensor, tiler=tiler, coord=coord, proj=proj)
       > ```

cutlass.cute.make\_layout\_image\_mask( : *lay: cutlass.cute.typing.Layout*, : *coord: cutlass.cute.typing.Coord*, : *mode: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int16[#](#cutlass.cute.make_layout_image_mask "Link to this definition")
:   Makes a 16-bit integer mask of the image of a layout sliced at a given mode
    and accounting for the offset given by the input coordinate for the other modes.

cutlass.cute.leading\_dim( : *shape: cutlass.cute.typing.Shape*, : *stride: cutlass.cute.typing.Stride*, ) → int | Tuple[int, ...] | None[#](#cutlass.cute.leading_dim "Link to this definition")
:   Find the leading dimension of a shape and stride.

    Parameters:
    :   * **shape** (*Shape*) â The shape of the tensor or layout
        * **stride** (*Stride*) â The stride of the tensor or layout

    Returns:
    :   The leading dimension index or indices

    Return type:
    :   Union[int, Tuple[int, â¦], None]

    The return value depends on the stride pattern:

    > * If a single leading dimension is found, returns an integer index
    > * If nested leading dimensions are found, returns a tuple of indices
    > * If no leading dimension is found, returns None

cutlass.cute.make\_layout\_tv( : *thr\_layout: cutlass.cute.typing.Layout*, : *val\_layout: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Shape, cutlass.cute.typing.Layout][#](#cutlass.cute.make_layout_tv "Link to this definition")
:   Create a thread-value layout by repeating the val\_layout over the thr\_layout.

    This function creates a thread-value layout that maps between `(thread_idx, value_idx)`
    coordinates and logical `(M,N)` coordinates. The thread and value layouts must be compact to ensure
    proper partitioning.

    This implements the thread-value partitioning pattern where data is partitioned
    across threads and values within each thread.

    Parameters:
    :   * **thr\_layout** (*Layout*) â Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
        * **val\_layout** (*Layout*) â Layout mapping from `(ValueM,ValueN)` coordinates to value IDs within each thread
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tuple containing `tiler_mn` and `layout_tv`

    Return type:
    :   Tuple[Shape, Layout]

    where:
    :   * `tiler_mn` is tiler and `shape(tiler_mn)` is compatible with `shape(zipped_divide(x, tiler_mn))[0]`
        * `layout_tv`: Thread-value layout mapping (thread\_idx, value\_idx) -> (M,N)

    **Example:**

    The below code creates a TV Layout that maps thread/value coordinates to the logical coordinates in a `(4,6)` tensor:
    :   * *Tiler MN*: `(4,6)`
        * *TV Layout*: `((3,2),(2,2)):((8,2),(4,1))`

    ```python
    thr_layout = cute.make_layout((2, 3), stride=(3, 1))
    val_layout = cute.make_layout((2, 2), stride=(2, 1))
    tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
    ```

    Table 1 TV Layout[#](#id1 "Link to this table")

    |  |  |  |  |  |  |  |
    | --- | --- | --- | --- | --- | --- | --- |
    |  | 0 | 1 | 2 | 3 | 4 | 5 |
    | 0 | T0, V0 | T0, V1 | T1, V0 | T1, V1 | T2, V0 | T2, V1 |
    | 1 | T0, V2 | T0, V3 | T1, V2 | T1, V3 | T2, V2 | T2, V3 |
    | 2 | T3, V0 | T3, V1 | T4, V0 | T4, V1 | T5, V0 | T5, V1 |
    | 3 | T3, V2 | T3, V3 | T4, V2 | T4, V3 | T5, V2 | T5, V3 |

cutlass.cute.get\_nonswizzle\_portion( : *layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.cute.get_nonswizzle_portion "Link to this definition")
:   Extract the non-swizzle portion from a layout.

    For a simple Layout, the entire layout is considered non-swizzled and is returned as-is.
    For a ComposedLayout, the inner layout (non-swizzled portion) is extracted and returned,
    effectively separating the base layout from any swizzle transformation that may be applied.

    Parameters:
    :   * **layout** (*Union**[**Layout**,* *ComposedLayout**]*) â A Layout or ComposedLayout from which to extract the non-swizzle portion.
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional

    Returns:
    :   The non-swizzle portion of the input layout. For Layout objects, returns the layout itself.
        For ComposedLayout objects, returns the outer layout component.

    Return type:
    :   Layout

    Raises:
    :   **TypeError** â If the layout is neither a Layout nor a ComposedLayout.

cutlass.cute.get\_swizzle\_portion( : *layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.\_mlir.ir.register\_value\_caster[#](#cutlass.cute.get_swizzle_portion "Link to this definition")
:   Extract or create the swizzle portion from a layout.

    For a simple Layout (which has no explicit swizzle), a default identity swizzle is created.
    For a ComposedLayout, the outer layout is checked and returned if it is a Swizzle object.
    Otherwise, a default identity swizzle is created. The default identity swizzle has parameters
    (0, 4, 3), which represents a no-op swizzle transformation.

    Parameters:
    :   * **layout** (*Union**[**Layout**,* *ComposedLayout**]*) â A Layout or ComposedLayout from which to extract the swizzle portion.
        * **loc** (*optional*) â Optional location information for IR diagnostics.
        * **ip** (*optional*) â Optional

    Returns:
    :   The swizzle portion of the layout. For Layout objects or ComposedLayout objects without
        a Swizzle outer component, returns a default identity swizzle (0, 4, 3). For ComposedLayout
        objects with a Swizzle outer component, returns that swizzle.

    Return type:
    :   [Swizzle](#cutlass.cute.Swizzle "cutlass.cute.Swizzle")

    Raises:
    :   **TypeError** â If the layout is neither a Layout nor a ComposedLayout.

cutlass.cute.transform\_leaf(*f*, *\*args*)[#](#cutlass.cute.transform_leaf "Link to this definition")
:   Apply a function to the leaf nodes of nested tuple structures.

    This function traverses nested tuple structures in parallel and applies the function f
    to corresponding leaf nodes. All input tuples must have the same nested structure.

    Parameters:
    :   * **f** (*Callable*) â Function to apply to leaf nodes
        * **args** â One or more nested tuple structures with matching profiles

    Returns:
    :   A new nested tuple with the same structure as the inputs, but with leaf values transformed by f

    Raises:
    :   **TypeError** â If the input tuples have different nested structures

    **Example:**

    ```python
    >>> transform_leaf(lambda x: x + 1, (1, 2))
    (2, 3)
    >>> transform_leaf(lambda x, y: x + y, (1, 2), (3, 4))
    (4, 6)
    >>> transform_leaf(lambda x: x * 2, ((1, 2), (3, 4)))
    ((2, 4), (6, 8))
    ```

cutlass.cute.find\_if( : *t: tuple | cutlass.\_mlir.ir.Value | int*, : *pred\_fn: Callable[[tuple | cutlass.\_mlir.ir.Value | int, int], bool]*, : *\**, : *loc=None*, : *ip=None*, ) → int | Tuple[int, ...] | None[#](#cutlass.cute.find_if "Link to this definition")

cutlass.cute.find( : *t: tuple | cutlass.\_mlir.ir.Value | int*, : *x: int*, : *\**, : *loc=None*, : *ip=None*, ) → int | Tuple[int, ...] | None[#](#cutlass.cute.find "Link to this definition")
:   Find the first position of a value `x` in a hierarchical structure `t`.

    Searches for the first occurrence of x in t, optionally excluding positions
    where a comparison value matches. The search can traverse nested structures
    and returns either a single index or a tuple of indices for nested positions.

    Parameters:
    :   * **t** (*Union**[**tuple**,* *ir.Value**,* *int**]*) â The search space
        * **x** (*int*) â The static integer x to search for

    Returns:
    :   Index if found at top level, tuple of indices showing nested position, or None if not found

    Return type:
    :   Union[int, Tuple[int, â¦], None]

cutlass.cute.flatten\_to\_tuple( : *a: cutlass.cute.typing.XTuple*, ) → Tuple[Any, ...][#](#cutlass.cute.flatten_to_tuple "Link to this definition")
:   Flattens a potentially nested tuple structure into a flat tuple.

    This function recursively traverses the input structure and flattens it into
    a single-level tuple, preserving the order of elements.

    Parameters:
    :   **a** (*Union**[**IntTuple**,* *Coord**,* *Shape**,* *Stride**]*) â The structure to flatten

    Returns:
    :   A flattened tuple containing all elements from the input

    Return type:
    :   tuple

    **Examples:**

    ```python
    flatten_to_tuple((1, 2, 3))       # Returns (1, 2, 3)
    flatten_to_tuple(((1, 2), 3))     # Returns (1, 2, 3)
    flatten_to_tuple((1, (2, (3,))))  # Returns (1, 2, 3)
    ```

cutlass.cute.unflatten( : *sequence: Tuple[Any, ...] | List[Any] | Iterable[Any]*, : *profile: cutlass.cute.typing.XTuple*, ) → cutlass.cute.typing.XTuple[#](#cutlass.cute.unflatten "Link to this definition")
:   Unflatten a flat tuple into a nested tuple structure according to a profile.

    This function transforms a flat sequence of elements into a nested tuple structure
    that matches the structure defined by the profile parameter. It traverses the profile
    structure and populates it with elements from the sequence.

    sequence must be long enough to fill the profile. Raises RuntimeError if it is not.

    Parameters:
    :   * **sequence** (*Union**[**Tuple**[**Any**,* *...**]**,* *List**[**Any**]**,* *Iterable**[**Any**]**]*) â A flat sequence of elements to be restructured
        * **profile** (*XTuple*) â A nested tuple structure that defines the shape of the output

    Returns:
    :   A nested tuple with the same structure as profile but containing elements from sequence

    Return type:
    :   XTuple

    **Examples:**

    ```python
    unflatten([1, 2, 3, 4], ((0, 0), (0, 0)))  # Returns ((1, 2), (3, 4))
    ```

cutlass.cute.product( : *a: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.product "Link to this definition")

cutlass.cute.product\_like( : *a: cutlass.cute.typing.IntTuple*, : *target\_profile: cutlass.cute.typing.XTuple*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.IntTuple[#](#cutlass.cute.product_like "Link to this definition")
:   Return product of the given IntTuple or Shape at leaves of target\_profile.

    This function computes products according to the structure defined by target\_profile.

    Parameters:
    :   * **a** (*IntTuple* *or* *Shape*) â The input tuple or shape
        * **target\_profile** (*XTuple*) â The profile that guides how products are computed
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The resulting tuple with products computed according to target\_profile

    Return type:
    :   IntTuple or Shape

    Raises:
    :   * **TypeError** â If inputs have incompatible types
        * **ValueError** â If inputs have incompatible shapes

cutlass.cute.product\_each( : *a: cutlass.cute.typing.IntTuple*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.IntTuple[#](#cutlass.cute.product_each "Link to this definition")

cutlass.cute.elem\_less( : *lhs: cutlass.cute.typing.Shape | cutlass.cute.typing.IntTuple | cutlass.cute.typing.Coord*, : *rhs: cutlass.cute.typing.Shape | cutlass.cute.typing.IntTuple | cutlass.cute.typing.Coord*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.elem_less "Link to this definition")

cutlass.cute.tuple\_cat(*\*tuples*)[#](#cutlass.cute.tuple_cat "Link to this definition")
:   Concatenate multiple tuples into a single tuple.

    This function takes any number of tuples and concatenates them into a single tuple.
    Non-tuple arguments are treated as single-element tuples.

    Parameters:
    :   **tuples** (*tuple* *or* *any*) â Variable number of tuples to concatenate

    Returns:
    :   A single concatenated tuple

    Return type:
    :   tuple

    **Examples:**

    ```python
    >>> tuple_cat((1, 2), (3, 4))
    (1, 2, 3, 4)
    >>> tuple_cat((1,), (2, 3), (4,))
    (1, 2, 3, 4)
    >>> tuple_cat(1, (2, 3))
    (1, 2, 3)
    ```

cutlass.cute.transform\_apply(*\*args*, *f: Callable*, *g: Callable*)[#](#cutlass.cute.transform_apply "Link to this definition")
:   Transform elements of tuple(s) with f, then apply g to all results.

    This function applies f to corresponding elements across input tuple(s),
    then applies g to all transformed results. It mimics the C++ CuTe implementation.

    Supports multiple signatures:
    - transform\_apply(t, f, g): For single tuple, computes g(f(t[0]), f(t[1]), â¦)
    - transform\_apply(t0, t1, f, g): For two tuples, computes g(f(t0[0], t1[0]), f(t0[1], t1[1]), â¦)
    - transform\_apply(t0, t1, t2, â¦, f, g): For multiple tuples of same length

    For non-tuple inputs, f is applied to the input(s) and g is applied to that single result.

    Parameters:
    :   * **args** â One or more tuples (or non-tuples) to transform
        * **f** (*Callable*) â The function to apply to each element (or corresponding elements across tuples)
        * **g** (*Callable*) â The function to apply to all transformed elements
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   The result of applying g to all transformed elements

    Return type:
    :   any

    **Examples:**

    ```python
    >>> transform_apply((1, 2, 3), f=lambda x: x * 2, g=lambda *args: sum(args))
    12  # (1*2 + 2*2 + 3*2) = 12
    >>> transform_apply((1, 2), f=lambda x: (x, x+1), g=tuple_cat)
    (1, 2, 2, 3)
    >>> transform_apply((1, 2), (3, 4), f=lambda x, y: x + y, g=lambda *args: args)
    (4, 6)
    ```

cutlass.cute.filter\_tuple(*\*args*, *f: Callable*)[#](#cutlass.cute.filter_tuple "Link to this definition")
:   Filter and flatten tuple elements by applying a function.

    The function f should return tuples, which are then concatenated together
    to produce the final result. This is useful for filtering and transforming
    tuple structures in a single pass.

    Parameters:
    :   * **t** (*Union**[**tuple**,* *ir.Value**,* *int**]*) â The tuple to filter
        * **f** (*Callable*) â The function to apply to each element of t
        * **loc** (*optional*) â Source location for MLIR, defaults to None
        * **ip** (*optional*) â Insertion point, defaults to None

    Returns:
    :   A concatenated tuple of all results

    Return type:
    :   tuple

    **Examples:**

    ```python
    >>> # Keep only even numbers, wrapped in tuples
    >>> filter_tuple((1, 2, 3, 4), lambda x: (x,) if x % 2 == 0 else ())
    (2, 4)
    >>> # Duplicate each element
    >>> filter_tuple((1, 2, 3), lambda x: (x, x))
    (1, 1, 2, 2, 3, 3)
    ```

*class* cutlass.cute.TensorSSA(*\*args: Any*, *\*\*kwargs: Any*)[#](#cutlass.cute.TensorSSA "Link to this definition")
:   Bases: `ArithValue`

    A class representing thread local data from CuTe Tensor in value semantic and immutable.

    Parameters:
    :   * **value** (*ir.Value*) â Flatten vector as ir.Value holding logic data of SSA Tensor
        * **shape** (*Shape*) â The nested shape in CuTe of the vector
        * **dtype** (*Type**[**Numeric**]*) â Data type of the tensor elements

    Variables:
    :   * **\_shape** â The nested shape in CuTe of the vector
        * **\_dtype** â Data type of the tensor elements

    Raises:
    :   **ValueError** â If shape is not static

    \_\_init\_\_( : *value*, : *shape: cutlass.cute.typing.Shape*, : *dtype: Type[cutlass.cute.typing.Numeric]*, )[#](#cutlass.cute.TensorSSA.__init__ "Link to this definition")
    :   Initialize a new TensorSSA object.

        Parameters:
        :   * **value** (*ir.Value*) â Flatten vector as ir.Value holding logic data of SSA Tensor
            * **shape** (*Shape*) â The nested shape in CuTe of the vector
            * **dtype** (*Type**[**Numeric**]*) â Data type of the tensor elements

        Raises:
        :   **ValueError** â If shape is not static

    *property* dtype*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.TensorSSA.dtype "Link to this definition")

    *property* element\_type*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.TensorSSA.element_type "Link to this definition")

    *property* shape[#](#cutlass.cute.TensorSSA.shape "Link to this definition")

    \_apply\_op( : *op*, : *other: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *flip=False*, : *\**, : *loc*, : *ip*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.TensorSSA._apply_op "Link to this definition")

    \_apply\_op( : *op*, : *other: cutlass.cutlass\_dsl.cutlass\_arith.ArithValue*, : *flip=False*, : *\**, : *loc*, : *ip*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")

    \_apply\_op( : *op*, : *other: int | float | bool*, : *flip=False*, : *\**, : *loc*, : *ip*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")

    apply\_op( : *op*, : *other*, : *flip=False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.TensorSSA.apply_op "Link to this definition")
    :   Apply a binary operation to this tensor and another operand.

        This is a public interface to the internal \_apply\_op method, providing
        a stable API for external users who need to apply custom operations.

        Parameters:
        :   * **op** â The operation function (e.g., operator.add, operator.mul, etc.)
            * **other** â The other operand (TensorSSA, ArithValue, or scalar)
            * **flip** â Whether to flip the operands (for right-hand operations)
            * **loc** â MLIR location (optional)
            * **ip** â MLIR insertion point (optional)

        Returns:
        :   The result of the operation

        Return type:
        :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

        Example

        ```python
        >>> tensor1 = cute.Tensor(...)
        >>> tensor2 = cute.Tensor(...)
        >>> result = tensor1.apply_op(operator.add, tensor2)
        >>> # Equivalent to: tensor1 + tensor2
        ```

    broadcast\_to( : *target\_shape: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.TensorSSA.broadcast_to "Link to this definition")
    :   Broadcast the tensor to the target shape.

    \_flatten\_shape\_and\_coord(*crd*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TensorSSA._flatten_shape_and_coord "Link to this definition")

    \_build\_result( : *res\_vect*, : *res\_shp*, : *\**, : *row\_major=False*, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.TensorSSA._build_result "Link to this definition")

    reshape( : *shape: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.TensorSSA.reshape "Link to this definition")
    :   Reshape the tensor to a new shape.

        Parameters:
        :   **shape** (*Shape*) â The new shape to reshape to.

        Returns:
        :   A new tensor with the same elements but with the new shape.

        Return type:
        :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

        Raises:
        :   * **NotImplementedError** â If dynamic size is not supported
            * **ValueError** â If the new shape is not compatible with the current shape

    to( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.TensorSSA.to "Link to this definition")
    :   Convert the tensor to a different numeric type.

        Parameters:
        :   **dtype** (*Type**[**Numeric**]*) â The target numeric type to cast to.

        Returns:
        :   A new tensor with the same shape but with elements cast to the target type.

        Return type:
        :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

        Raises:
        :   * **TypeError** â If dtype is not a subclass of Numeric.
            * **NotImplementedError** â If dtype is an unsigned integer type.

    ir\_value(*\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TensorSSA.ir_value "Link to this definition")

    ir\_value\_int8(*\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TensorSSA.ir_value_int8 "Link to this definition")
    :   Returns int8 ir value of Boolean tensor.
        When we need to store Boolean tensor ssa, use ir\_value\_int8().

        Parameters:
        :   * **loc** (*Optional**[**Location**]**,* *optional*) â Source location information, defaults to None
            * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point for MLIR operations, defaults to None

        Returns:
        :   The int8 value of this Boolean

        Return type:
        :   ir.Value

    reduce( : *op*, : *init\_val*, : *reduction\_profile: cutlass.cute.typing.Coord*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.TensorSSA.reduce "Link to this definition")
    :   Perform reduce on selected modes with given predefined reduction op.

        Parameters:
        :   * **op** (*operator*) â The reduction operator to use (operator.add or operator.mul)
            * **init\_val** (*numeric*) â The initial value for the reduction
            * **reduction\_profile** (*Coord*) â Specifies which dimensions to reduce. Dimensions marked with None are kept.

        Returns:
        :   The reduced tensor

        Return type:
        :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

        **Examples:**

        ```python
        reduce(f32 o (4,))
          => f32

        reduce(f32 o (4, 5))
          => f32
        reduce(f32 o (4, (5, 4)), reduction_profile=(None, 1))
          => f32 o (4,)
        reduce(f32 o (4, (5, 4)), reduction_profile=(None, (None, 1)))
          => f32 o (4, (5,))
        ```

cutlass.cute.make\_tensor( : *iterator*, : *layout: cutlass.cute.typing.Shape | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.make_tensor "Link to this definition")
:   Creates a tensor by composing an engine (iterator/pointer) with a layout.

    A tensor is defined as T = E â L, where E is an engine (array, pointer, or counting iterator)
    and L is a layout that maps logical coordinates to physical offsets. The tensor
    evaluates coordinates by applying the layout mapping and dereferencing the engine
    at the resulting offset.

    Parameters:
    :   * **iterator** (*Union**[**Pointer**,* *IntTuple**,* *ir.Value**]*) â Engine component that provides data access capabilities. Can be:
          - A pointer (Pointer type)
          - An integer or integer tuple for coordinate tensors
          - A shared memory descriptor (SmemDescType)
        * **layout** (*Union**[**Shape**,* *Layout**,* *ComposedLayout**]*) â Layout component that defines the mapping from logical coordinates to
          physical offsets. Can be:
          - A shape tuple that will be converted to a layout
          - A Layout object
          - A ComposedLayout object (must be a normal layout)
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR operation tracking, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for MLIR operation, defaults to None

    Returns:
    :   A tensor object representing the composition E â L

    Return type:
    :   Tensor

    Raises:
    :   * **TypeError** â If iterator type is not a supported type
        * **ValueError** â If layout is a composed layout with customized inner functions

    **Examples:**

    ```python
    # Create a tensor with row-major layout from a pointer
    ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
    layout = make_layout((64, 128), stride=(128, 1))
    tensor = make_tensor(ptr, layout)

    # Create a tensor with hierarchical layout in shared memory
    smem_ptr = make_ptr(Float16, base_ptr, AddressSpace.smem)
    layout = make_layout(((128, 8), (1, 4, 1)), stride=((32, 1), (0, 8, 4096)))
    tensor = make_tensor(smem_ptr, layout)

    # Create a coordinate tensor
    layout = make_layout(2, stride=16 * E(0))
    tensor = make_tensor(5, layout)  # coordinate tensor with iterator starting at 5
    ```

    Notes

    * The engine (iterator) must support random access operations
    * Common engine types include raw pointers, arrays, and random-access iterators
    * The layout defines both the shape (logical dimensions) and stride (physical mapping)
    * Supports both direct coordinate evaluation T(c) and partial evaluation (slicing)
    * ComposedLayouts must be ânormalâ layouts (no inner functions)
    * For coordinate tensors, the iterator is converted to a counting sequence

cutlass.cute.make\_identity\_tensor( : *shape: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.make_identity_tensor "Link to this definition")
:   Creates an identity tensor with the given shape.

    An identity tensor maps each coordinate to itself, effectively creating a counting
    sequence within the shapeâs bounds. This is useful for generating coordinate indices
    or creating reference tensors for layout transformations.

    Parameters:
    :   * **shape** (*Shape*) â The shape defining the tensorâs dimensions. Can be a simple integer
          sequence or a hierarchical structure ((m,n),(p,q))
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR operation tracking, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for MLIR operation, defaults to None

    Returns:
    :   A tensor that maps each coordinate to itself

    Return type:
    :   Tensor

    **Examples:**

    ```python
    # Create a simple 1D coord tensor
    tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

    # Create a 2D coord tensor
    tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

    # Create hierarchical coord tensor
    tensor = make_identity_tensor(((2,1),3))
    # [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
    ```

    Notes

    * The shape parameter follows CuTeâs IntTuple concept
    * Coordinates are ordered colexicographically
    * Useful for generating reference coordinates in layout transformations

cutlass.cute.make\_fragment( : *layout\_or\_shape: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.make_fragment "Link to this definition")

cutlass.cute.make\_fragment\_like(*src*, *dtype=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_fragment_like "Link to this definition")

cutlass.cute.make\_rmem\_tensor\_like( : *src: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor | [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *dtype: Type[cutlass.cute.typing.Numeric] | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.make_rmem_tensor_like "Link to this definition")
:   Creates a tensor in register memory with the same shape as the input layout but
    :   compact col-major strides. This is equivalent to calling make\_rmem\_tensor(make\_layout\_like(tensor)).

    This function allocates a tensor in register memory (rmem) usually on stack with
    with the compact layout like the source. The tensor will have elements of the
    specified numeric data type or the same as the source.

    Parameters:
    :   * **src** (*Union**[**Layout**,* *ComposedLayout**,* *Tensor**]*) â The source layout or tensor whose shape will be matched
        * **dtype** (*Type**[**Numeric**]**,* *optional*) â The element type for the fragment tensor, defaults to None
        * **loc** (*Location**,* *optional*) â Source location for MLIR operations, defaults to None
        * **ip** (*InsertionPoint**,* *optional*) â Insertion point for MLIR operations, defaults to None

    Returns:
    :   A new layout or fragment tensor with matching shape

    Return type:
    :   Union[Layout, Tensor]

    **Examples:**

    Creating a rmem tensor from a tensor:

    ```python
    smem_tensor = cute.make_tensor(smem_ptr, layout)
    rmem_tensor = cute.make_rmem_tensor_like(smem_tensor, cutlass.Float32)
    # frag_tensor will be a register-backed tensor with the same shape
    ```

    Creating a fragment with a different element type:

    ```python
    tensor = cute.make_tensor(gmem_ptr, layout)
    rmem_bool_tensor = cute.make_rmem_tensor_like(tensor, cutlass.Boolean)
    # bool_frag will be a register-backed tensor with Boolean elements
    ```

    **Notes**

    * When used with a Tensor, if a type is provided, it will create a new
      fragment tensor with that element type.
    * For layouts with ScaledBasis strides, the function creates a fragment
      from the shape only.
    * This function is commonly used in GEMM and other tensor operations to
      create register storage for intermediate results.

cutlass.cute.make\_rmem\_tensor( : *layout\_or\_shape: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.make_rmem_tensor "Link to this definition")
:   Creates a tensor in register memory with the specified layout/shape and data type.

    This function allocates a tensor in register memory (rmem) usually on stack with
    either a provided layout or creates a new layout from the given shape. The tensor
    will have elements of the specified numeric data type.

    Parameters:
    :   * **layout\_or\_shape** (*Union**[**Layout**,* *Shape**]*) â Either a Layout object defining the tensorâs memory organization,
          or a Shape defining its dimensions
        * **dtype** (*Type**[**Numeric**]*) â The data type for tensor elements (must be a Numeric type)
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR operation tracking, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for MLIR operation, defaults to None

    Returns:
    :   A tensor allocated in register memory

    Return type:
    :   Tensor

    **Examples:**

    ```python
    # Create rmem tensor with explicit layout
    layout = make_layout((128, 32))
    tensor = make_rmem_tensor(layout, cutlass.Float16)

    # Create rmem tensor directly from shape
    tensor = make_rmem_tensor((64, 64), cutlass.Float32)
    ```

    Notes

    * Uses 32-byte alignment to support .128 load/store operations
    * Boolean types are stored as 8-bit integers
    * Handles both direct shapes and Layout objects

cutlass.cute.recast\_tensor( : *src: cutlass.cute.typing.Tensor*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *swizzle\_=None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.recast_tensor "Link to this definition")
:   Recast a tensor to a different data type by changing the element interpretation.

    This function reinterprets the memory of a tensor with a different element type,
    adjusting both the iterator pointer type and the layout to maintain consistency.

    Parameters:
    :   * **src** (*Tensor*) â The source tensor to recast
        * **dtype** (*Type**[**Numeric**]*) â The target data type for tensor elements
        * **swizzle** (*Optional**,* *unused*) â Optional swizzle parameter (reserved for future use), defaults to None
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR operation tracking, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for MLIR operation, defaults to None

    Returns:
    :   A new tensor with the same memory but reinterpreted as dtype

    Return type:
    :   Tensor

    Raises:
    :   **TypeError** â If dtype is not a subclass of Numeric

    **Examples:**

    ```python
    # Create a Float32 tensor
    tensor_f32 = make_rmem_tensor((4, 8), Float32)

    # Recast to Int32 to manipulate bits
    tensor_i32 = recast_tensor(tensor_f32, Int32)

    # Both tensors share the same memory, but interpret it differently
    ```

cutlass.cute.domain\_offset( : *coord: cutlass.cute.typing.Coord*, : *tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.domain_offset "Link to this definition")
:   Offset the tensor domain by the given coordinate.

    This function creates a new tensor by offsetting the iterator/pointer of the input tensor
    by the amount corresponding to the given coordinate in its layout.

    Parameters:
    :   * **coord** (*Coord*) â The coordinate offset to apply
        * **tensor** (*Tensor*) â The source tensor to offset
        * **loc** (*Optional**[**Location**]*) â Source location for MLIR operation tracking, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for MLIR operation, defaults to None

    Returns:
    :   A new tensor with the offset iterator

    Return type:
    :   Tensor

    Raises:
    :   **ValueError** â If the tensor type doesnât support domain offsetting

    **Examples:**

    ```python
    # Create a tensor with a row-major layout
    ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
    layout = make_layout((64, 128), stride=(128, 1))
    tensor = make_tensor(ptr, layout)

    # Offset by coordinate (3, 5)
    offset_tensor = domain_offset((3, 5), tensor)
    # offset_tensor now points to element at (3, 5)
    ```

cutlass.cute.print\_tensor( : *tensor: cutlass.cute.typing.Tensor | [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *\**, : *verbose: bool = False*, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.print_tensor "Link to this definition")
:   Print content of the tensor in human readable format.

    Outputs the tensor data in a structured format showing both metadata
    and the actual data values. The output includes tensor type information,
    layout details, and a formatted array representation of the values.

    Parameters:
    :   * **tensor** (*Tensor*) â The tensor to print
        * **verbose** (*bool*) â If True, includes additional debug information in the output
        * **loc** (*source location**,* *optional*) â Source location where itâs called, defaults to None
        * **ip** (*insertion pointer**,* *optional*) â Insertion pointer for IR generation, defaults to None

    Raises:
    :   **NotImplementedError** â If the tensor type doesnât support trivial dereferencing

    **Example output:**

    ```python
    tensor(raw_ptr<@..., Float32, generic, align(4)> o (8,5):(5,1), data=
           [[-0.4326, -0.5434,  0.1238,  0.7132,  0.8042],
            [-0.8462,  0.9871,  0.4389,  0.7298,  0.6948],
            [ 0.3426,  0.5856,  0.1541,  0.2923,  0.6976],
            [-0.1649,  0.8811,  0.1788,  0.1404,  0.2568],
            [-0.2944,  0.8593,  0.4171,  0.8998,  0.1766],
            [ 0.8814,  0.7919,  0.7390,  0.4566,  0.1576],
            [ 0.9159,  0.7577,  0.6918,  0.0754,  0.0591],
            [ 0.6551,  0.1626,  0.1189,  0.0292,  0.8655]])
    ```

cutlass.cute.full( : *shape*, : *fill\_value*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.full "Link to this definition")
:   Return a new TensorSSA of given shape and type, filled with fill\_value.

    Parameters:
    :   * **shape** (*tuple*) â Shape of the new tensor.
        * **fill\_value** (*scalar*) â Value to fill the tensor with.
        * **dtype** (*Type**[**Numeric**]*) â Data type of the tensor.

    Returns:
    :   Tensor of fill\_value with the specified shape and dtype.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.full\_like( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Tensor*, : *fill\_value*, : *dtype: Type[cutlass.cute.typing.Numeric] | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.full_like "Link to this definition")
:   Return a full TensorSSA with the same shape and type as a given array.

    Parameters:
    :   * **a** (*array\_like*) â The shape and data-type of a define these same attributes of the returned array.
        * **fill\_value** (*array\_like*) â Fill value.
        * **dtype** (*Union**[**None**,* *Type**[**Numeric**]**]**,* *optional*) â Overrides the data type of the result, defaults to None

    Returns:
    :   Tensor of fill\_value with the same shape and type as a.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

    See also

    [`empty_like()`](#cutlass.cute.empty_like "cutlass.cute.empty_like"): Return an empty array with shape and type of input.
    [`ones_like()`](#cutlass.cute.ones_like "cutlass.cute.ones_like"): Return an array of ones with shape and type of input.
    [`zeros_like()`](#cutlass.cute.zeros_like "cutlass.cute.zeros_like"): Return an array of zeros with shape and type of input.
    [`full()`](#cutlass.cute.full "cutlass.cute.full"): Return a new array of given shape filled with value.

    **Examples:**

    ```python
    frg = cute.make_rmem_tensor((2, 3), Float32)
    a = frg.load()
    b = cute.full_like(a, 1.0)
    ```

cutlass.cute.empty\_like(*a*, *dtype=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.empty_like "Link to this definition")
:   Return a new TensorSSA with the same shape and type as a given array, without initializing entries.

    Parameters:
    :   * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â The shape and data-type of a define these same attributes of the returned array.
        * **dtype** (*Type**[**Numeric**]**,* *optional*) â Overrides the data type of the result, defaults to None

    Returns:
    :   Uninitialized tensor with the same shape and type (unless overridden) as a.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.ones\_like(*a*, *dtype=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.ones_like "Link to this definition")
:   Return a TensorSSA of ones with the same shape and type as a given array.

    Parameters:
    :   * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â The shape and data-type of a define these same attributes of the returned array.
        * **dtype** (*Type**[**Numeric**]**,* *optional*) â Overrides the data type of the result, defaults to None

    Returns:
    :   Tensor of ones with the same shape and type (unless overridden) as a.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.zeros\_like(*a*, *dtype=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.zeros_like "Link to this definition")
:   Return a TensorSSA of zeros with the same shape and type as a given array.

    Parameters:
    :   * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â The shape and data-type of a define these same attributes of the returned array.
        * **dtype** (*Type**[**Numeric**]**,* *optional*) â Overrides the data type of the result, defaults to None

    Returns:
    :   Tensor of zeros with the same shape and type (unless overridden) as a.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.where( : *cond: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *x: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *y: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")[#](#cutlass.cute.where "Link to this definition")
:   Return elements chosen from x or y depending on condition; will auto broadcast x or y if needed.

    Parameters:
    :   * **cond** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â Where True, yield x, where False, yield y.
        * **x** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Values from which to choose when condition is True.
        * **y** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Values from which to choose when condition is False.

    Returns:
    :   A tensor with elements from x where condition is True, and elements from y where condition is False.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.any\_( : *x: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.any_ "Link to this definition")
:   Test whether any tensor element evaluates to True.

    Parameters:
    :   **x** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â Input tensor.

    Returns:
    :   Returns a TensorSSA scalar containing True if any element of x is True, False otherwise.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

cutlass.cute.all\_( : *x: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean[#](#cutlass.cute.all_ "Link to this definition")
:   Test whether all tensor elements evaluate to True.

    Parameters:
    :   **x** ([*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) â Input tensor.

    Returns:
    :   Returns a TensorSSA scalar containing True if all elements of x are True, False otherwise.

    Return type:
    :   [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

*class* cutlass.cute.Atom(*op: Op*, *trait: Trait*)[#](#cutlass.cute.Atom "Link to this definition")
:   Bases: `ABC`

    Atom base class.

    An Atom is the composition of

    * a MMA or Copy Operation;
    * an internal MMA or Copy Trait.

    An Operation is a pure Python class that is used to model a specific MMA or Copy instruction.
    The Trait wraps the underlying IR Value and provides access to the metadata of the instruction
    encoded using CuTe Layouts. When the Trait can be constructed straighforwardly from an
    Operation, the `make_mma_atom` or `make_copy_atom` API should be used. There are cases where
    constructing the metadata is not trivial and requires more information, for example to determine
    the number of bytes copied per TMA instruction (âthe TMA vector lengthâ). In such cases,
    dedicated helper functions are provided with an appropriate API such that the Atom is
    constructed internally in an optimal fashion for the user.

    \_\_init\_\_( : *op: Op*, : *trait: Trait*, ) → None[#](#cutlass.cute.Atom.__init__ "Link to this definition")

    *property* op*: Op*[#](#cutlass.cute.Atom.op "Link to this definition")

    *property* type[#](#cutlass.cute.Atom.type "Link to this definition")

    set(*modifier*, *value*, *\**, *loc=None*, *ip=None*) → None[#](#cutlass.cute.Atom.set "Link to this definition")
    :   Sets runtime fields of the Atom.

        Some Atoms have runtime state, for example a tcgen05 MMA Atom

        ```python
        tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
        tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
        ```

        The `set` method provides a way to the user to modify such runtime state. Modifiable
        fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
        instance internally validates the field as well as the value provided by the user to set
        the field to.

    get(*field*, *\**, *loc=None*, *ip=None*) → Any[#](#cutlass.cute.Atom.get "Link to this definition")
    :   Gets runtime fields of the Atom.

        Some Atoms have runtime state, for example a tcgen05 MMA Atom

        ```python
        tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
        accum = tiled_mma.get(cute.nvgpu.tcgen05.Field.ACCUMULATE)
        ```

        The `get` method provides a way to the user to access such runtime state. Modifiable
        fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
        instance internally validates the field as well as the value provided by the user to set
        the field to.

    with\_(*\**, *loc=None*, *ip=None*, *\*\*kwargs*) → [Atom](#cutlass.cute.Atom "cutlass.cute.atom.Atom")[#](#cutlass.cute.Atom.with_ "Link to this definition")
    :   Returns a new Atom with the new Operation and Trait with the given runtime state. The runtime state
        is provided as keyword arguments and it is Atom-specific.

        ```python
        tiled_copy = cute.make_tiled_copy(tma_copy_op)
        new_tiled_copy = tiled_copy.with_(tma_bar_ptr=tma_bar_ptr, cache_policy=cute.CacheEvictionPriority.EVICT_LAST)
        ```

        The `with_` method provides a way to the user to modify such runtime state or create an executable Atom
        (e.g. an Executable TMA Load Atom).

    \_unpack(*\**, *loc=None*, *ip=None*, *\*\*kwargs*) → cutlass.\_mlir.ir.Value[#](#cutlass.cute.Atom._unpack "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.Atom._abc_impl "Link to this definition")

*class* cutlass.cute.MmaAtom(*op: Op*, *trait: Trait*)[#](#cutlass.cute.MmaAtom "Link to this definition")
:   Bases: [`Atom`](#cutlass.cute.Atom "cutlass.cute.atom.Atom")

    The MMA Atom class.

    *property* thr\_id[#](#cutlass.cute.MmaAtom.thr_id "Link to this definition")

    *property* shape\_mnk[#](#cutlass.cute.MmaAtom.shape_mnk "Link to this definition")

    *property* tv\_layout\_A[#](#cutlass.cute.MmaAtom.tv_layout_A "Link to this definition")

    *property* tv\_layout\_B[#](#cutlass.cute.MmaAtom.tv_layout_B "Link to this definition")

    *property* tv\_layout\_C[#](#cutlass.cute.MmaAtom.tv_layout_C "Link to this definition")

    make\_fragment\_A(*input*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.MmaAtom.make_fragment_A "Link to this definition")

    make\_fragment\_B(*input*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.MmaAtom.make_fragment_B "Link to this definition")

    make\_fragment\_C(*input*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.MmaAtom.make_fragment_C "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.MmaAtom._abc_impl "Link to this definition")

*class* cutlass.cute.CopyAtom(*op: Op*, *trait: Trait*)[#](#cutlass.cute.CopyAtom "Link to this definition")
:   Bases: [`Atom`](#cutlass.cute.Atom "cutlass.cute.atom.Atom")

    The Copy Atom class.

    *property* value\_type*: Type[cutlass.cute.typing.Numeric]*[#](#cutlass.cute.CopyAtom.value_type "Link to this definition")

    *property* thr\_id*: cutlass.cute.typing.Layout*[#](#cutlass.cute.CopyAtom.thr_id "Link to this definition")

    *property* layout\_src\_tv*: cutlass.cute.typing.Layout*[#](#cutlass.cute.CopyAtom.layout_src_tv "Link to this definition")

    *property* layout\_dst\_tv*: cutlass.cute.typing.Layout*[#](#cutlass.cute.CopyAtom.layout_dst_tv "Link to this definition")

    *property* smem\_layout[#](#cutlass.cute.CopyAtom.smem_layout "Link to this definition")
    :   Convenience property to access the SMEM layout for TMA copy atoms.

        This is a shortcut for `atom.op.smem_layout` that checks if the operation
        is a TMA operation and provides a clearer error message if not.

        Returns:
        :   The SMEM layout

        Return type:
        :   Layout or ComposedLayout

        Raises:
        :   * **TypeError** â If the operation is not a TMA operation
            * **ValueError** â If the SMEM layout is not set

        Example

        ```python
        >>> layout = tma_atom.smem_layout  # Instead of tma_atom.op.smem_layout
        ```

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.CopyAtom._abc_impl "Link to this definition")

*class* cutlass.cute.TiledCopy(*op: Op*, *trait: Trait*)[#](#cutlass.cute.TiledCopy "Link to this definition")
:   Bases: [`CopyAtom`](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")

    The tiled Copy class.

    *property* layout\_tv\_tiled*: cutlass.cute.typing.Layout*[#](#cutlass.cute.TiledCopy.layout_tv_tiled "Link to this definition")

    *property* tiler\_mn*: cutlass.cute.typing.Tile*[#](#cutlass.cute.TiledCopy.tiler_mn "Link to this definition")

    *property* layout\_src\_tv\_tiled*: cutlass.cute.typing.Layout*[#](#cutlass.cute.TiledCopy.layout_src_tv_tiled "Link to this definition")

    *property* layout\_dst\_tv\_tiled*: cutlass.cute.typing.Layout*[#](#cutlass.cute.TiledCopy.layout_dst_tv_tiled "Link to this definition")

    *property* size*: int*[#](#cutlass.cute.TiledCopy.size "Link to this definition")

    get\_slice( : *thr\_idx: int | cutlass.cute.typing.Int32*, ) → [ThrCopy](#cutlass.cute.ThrCopy "cutlass.cute.atom.ThrCopy")[#](#cutlass.cute.TiledCopy.get_slice "Link to this definition")

    retile(*src*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TiledCopy.retile "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.TiledCopy._abc_impl "Link to this definition")

*class* cutlass.cute.TiledMma(*op: Op*, *trait: Trait*)[#](#cutlass.cute.TiledMma "Link to this definition")
:   Bases: [`MmaAtom`](#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")

    The tiled MMA class.

    *property* tv\_layout\_A\_tiled[#](#cutlass.cute.TiledMma.tv_layout_A_tiled "Link to this definition")

    *property* tv\_layout\_B\_tiled[#](#cutlass.cute.TiledMma.tv_layout_B_tiled "Link to this definition")

    *property* tv\_layout\_C\_tiled[#](#cutlass.cute.TiledMma.tv_layout_C_tiled "Link to this definition")

    *property* permutation\_mnk[#](#cutlass.cute.TiledMma.permutation_mnk "Link to this definition")

    *property* thr\_layout\_vmnk[#](#cutlass.cute.TiledMma.thr_layout_vmnk "Link to this definition")

    *property* size*: int*[#](#cutlass.cute.TiledMma.size "Link to this definition")

    get\_tile\_size(*mode\_idx: int*) → cutlass.cute.typing.Shape[#](#cutlass.cute.TiledMma.get_tile_size "Link to this definition")

    get\_slice( : *thr\_idx: int | cutlass.cute.typing.Int32*, ) → [ThrMma](#cutlass.cute.ThrMma "cutlass.cute.atom.ThrMma")[#](#cutlass.cute.TiledMma.get_slice "Link to this definition")

    \_partition\_shape(*operand\_id*, *shape*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TiledMma._partition_shape "Link to this definition")

    partition\_shape\_A(*shape\_mk*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TiledMma.partition_shape_A "Link to this definition")

    partition\_shape\_B(*shape\_nk*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TiledMma.partition_shape_B "Link to this definition")

    partition\_shape\_C(*shape\_mn*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.TiledMma.partition_shape_C "Link to this definition")

    \_thrfrg( : *operand\_id*, : *input: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout[#](#cutlass.cute.TiledMma._thrfrg "Link to this definition")

    \_thrfrg( : *operand\_id*, : *input: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor

    \_thrfrg\_A( : *input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor[#](#cutlass.cute.TiledMma._thrfrg_A "Link to this definition")

    \_thrfrg\_B( : *input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor[#](#cutlass.cute.TiledMma._thrfrg_B "Link to this definition")

    \_thrfrg\_C( : *input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor[#](#cutlass.cute.TiledMma._thrfrg_C "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.TiledMma._abc_impl "Link to this definition")

*class* cutlass.cute.ThrMma( : *op: Op*, : *trait: Trait*, : *thr\_idx: int | cutlass.cute.typing.Int32*, )[#](#cutlass.cute.ThrMma "Link to this definition")
:   Bases: [`TiledMma`](#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")

    The thread MMA class for modeling a thread-slice of a tiled MMA.

    \_\_init\_\_( : *op: Op*, : *trait: Trait*, : *thr\_idx: int | cutlass.cute.typing.Int32*, ) → None[#](#cutlass.cute.ThrMma.__init__ "Link to this definition")

    *property* thr\_idx[#](#cutlass.cute.ThrMma.thr_idx "Link to this definition")

    partition\_A( : *input\_mk: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.ThrMma.partition_A "Link to this definition")

    partition\_B( : *input\_nk: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.ThrMma.partition_B "Link to this definition")

    partition\_C( : *input\_mn: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.ThrMma.partition_C "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.ThrMma._abc_impl "Link to this definition")

*class* cutlass.cute.ThrCopy( : *op: Op*, : *trait: Trait*, : *thr\_idx: int | cutlass.cute.typing.Int32*, )[#](#cutlass.cute.ThrCopy "Link to this definition")
:   Bases: [`TiledCopy`](#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")

    The thread Copy class for modeling a thread-slice of a tiled Copy.

    \_\_init\_\_( : *op: Op*, : *trait: Trait*, : *thr\_idx: int | cutlass.cute.typing.Int32*, ) → None[#](#cutlass.cute.ThrCopy.__init__ "Link to this definition")

    *property* thr\_idx[#](#cutlass.cute.ThrCopy.thr_idx "Link to this definition")

    partition\_S( : *src: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.ThrCopy.partition_S "Link to this definition")

    partition\_D( : *dst: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor[#](#cutlass.cute.ThrCopy.partition_D "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.cute.ThrCopy._abc_impl "Link to this definition")

cutlass.cute.make\_atom(*ty*, *values=None*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_atom "Link to this definition")
:   This is a wrapper around the \_cute\_ir.make\_atom operation, providing default value for the values argument.

cutlass.cute.make\_mma\_atom( : *op: MmaOp*, : *\**, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → [MmaAtom](#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")[#](#cutlass.cute.make_mma_atom "Link to this definition")
:   Makes an MMA Atom from an MMA Operation.

    This function creates an MMA Atom from a given MMA Operation. Arbitrary kw arguments can be
    provided for Op-specific additional parameters. They are not used as of today.

    Parameters:
    :   **op** (*MmaOp*) â The MMA Operation to construct an Atom for

    Returns:
    :   The MMA Atom

    Return type:
    :   [MmaAtom](#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")

cutlass.cute.make\_tiled\_mma( : *op\_or\_atom: Op | [MmaAtom](#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")*, : *atom\_layout\_mnk=(1, 1, 1)*, : *permutation\_mnk=None*, : *\**, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → [TiledMma](#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")[#](#cutlass.cute.make_tiled_mma "Link to this definition")
:   Makes a tiled MMA from an MMA Operation or an MMA Atom.

    Parameters:
    :   * **op\_or\_atom** (*Union**[**Op**,* [*MmaAtom*](#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")*]*) â The MMA Operation or Atom
        * **atom\_layout\_mnk** (*Layout*) â A Layout describing the tiling of Atom across threads
        * **permutation\_mnk** (*Tiler*) â A permutation Tiler describing the tiling of Atom across values including any permutation of such tiling

    Returns:
    :   The resulting tiled MMA

    Return type:
    :   [TiledMma](#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

cutlass.cute.make\_copy\_atom( : *op: CopyOp*, : *copy\_internal\_type: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.cute.make_copy_atom "Link to this definition")
:   Makes a Copy Atom from a Copy Operation.

    This function creates a Copy Atom from a given Copy Operation. Arbitrary kw arguments can be
    provided for Op-specific additional parameters.

    Example:

    ```python
    op = cute.nvgpu.CopyUniversalOp()
    atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
    ```

    Parameters:
    :   * **op** (*CopyOp*) â The Copy Operation to construct an Atom for
        * **copy\_internal\_type** (*Type**[**Numeric**]*) â An internal data type used to construct the source/destination layouts in unit of tensor elements

    Returns:
    :   The Copy Atom

    Return type:
    :   [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

cutlass.cute.make\_tiled\_copy\_tv( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *thr\_layout: cutlass.cute.typing.Layout*, : *val\_layout: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")[#](#cutlass.cute.make_tiled_copy_tv "Link to this definition")
:   Create a tiled copy given separate thread and value layouts.

    A TV partitioner is inferred based on the input layouts. The input thread layout
    must be compact.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **thr\_layout** (*Layout*) â Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
        * **val\_layout** (*Layout*) â Layout mapping from `(ValueM,ValueN)` coordinates to value IDs
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy(*atom*, *layout\_tv*, *tiler\_mn*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy "Link to this definition")
:   Create a tiled type given a TV partitioner and tiler.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom, e.g. smit\_copy and simt\_async\_copy, tma\_load, etc.
        * **layout\_tv** (*Layout*) â Thread-value layout
        * **tiler\_mn** (*Tiler*) â Tile size
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_S(*atom*, *tiled\_copy*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy_S "Link to this definition")
:   Create a tiled copy out of the copy\_atom that matches the Src-Layout of tiled\_copy.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **tiled\_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) â Tiled copy
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_D(*atom*, *tiled\_copy*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy_D "Link to this definition")
:   Create a tiled copy out of the copy\_atom that matches the Dst-Layout of tiled\_copy.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **tiled\_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) â Tiled copy
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_A(*atom*, *tiled\_mma*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy_A "Link to this definition")
:   Create a tiled copy out of the copy\_atom that matches the A-Layout of tiled\_mma.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â Tiled MMA
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_B(*atom*, *tiled\_mma*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy_B "Link to this definition")
:   Create a tiled copy out of the copy\_atom that matches the B-Layout of tiled\_mma.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â Tiled MMA
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_C(*atom*, *tiled\_mma*, *\**, *loc=None*, *ip=None*)[#](#cutlass.cute.make_tiled_copy_C "Link to this definition")
:   Create a tiled copy out of the copy\_atom that matches the C-Layout of tiled\_mma.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â Tiled MMA
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for the partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

cutlass.cute.make\_tiled\_copy\_C\_atom( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *mma: [TiledMma](#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.cute.make_tiled_copy_C_atom "Link to this definition")
:   Create the smallest tiled copy that can retile LayoutC\_TV for use with pipelined epilogues with subtiled stores.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom
        * **mma** ([*TiledMma*](#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â Tiled MMA
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Returns:
    :   A tiled copy for partitioner

    Return type:
    :   [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

    Raises:
    :   **ValueError** â If the number value of CopyAtomâs source layout is greater than the size of TiledMmaâs LayoutC\_TV

cutlass.cute.make\_cotiled\_copy( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *atom\_layout\_tv: cutlass.cute.typing.Layout*, : *data\_layout: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")[#](#cutlass.cute.make_cotiled_copy "Link to this definition")
:   Produce a TiledCopy from thread and value offset maps.
    The TV Layout maps threads and values to the codomain of the data\_layout.
    It is verified that the intended codomain is valid within data\_layout.
    Useful when threads and values donât care about owning specific coordinates, but
    care more about the vector-width and offsets between them.

    Parameters:
    :   * **atom** (*copy atom**,* *e.g. simt\_copy and simt\_async\_copy**,* *tgen05.st**,* *etc.*)
        * **atom\_layout\_tv** (*(**tid**,* *vid**)* *-> data addr*)
        * **data\_layout** (*data coord -> data addr*)
        * **loc** (*source location for mlir* *(**optional**)*)
        * **ip** (*insertion point* *(**optional**)*)

    Returns:
    :   A tuple of A tiled copy and atom

    Return type:
    :   tiled\_copy

cutlass.cute.copy\_atom\_call( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *src: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *dst: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *\**, : *pred: cutlass.cute.typing.Tensor | None = None*, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → None[#](#cutlass.cute.copy_atom_call "Link to this definition")
:   Execute a single copy atom operation.

    The copy\_atom\_call operation executes a copy atom with the given operands.
    Source and destination tensors have layout profile `(V)`.

    The `V-mode` represents either:

    * A singular mode directly consumable by the provided Copy Atom
    * A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,

    For src/dst layout like `(V, Rest...)`, the layout profile of `pred` must match `(Rest...)`.

    > * Certain Atoms may require additional operation-specific keyword arguments.
    > * Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
    >   for future releases.

    Both `src` and `dst` operands are variadic, containing a variable number of tensors:

    * For regular copy, `src` and `dst` each contain a single tensor.
    * For copy with auxiliary operands, they contain the main tensor followed by
      auxiliary tensors. For example:

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom specifying the transfer operation
        * **src** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â Source tensor(s) with layout profile `(V)`. Can be a single Tensor
          or a list/tuple of Tensors for operations with auxiliary source operands.
        * **dst** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â Destination tensor(s) with layout profile `(V)`. Can be a single Tensor
          or a list/tuple of Tensors for operations with auxiliary destination operands.
        * **pred** (*Optional**[**Tensor**]**,* *optional*) â Optional predication tensor for conditional transfers, defaults to None
        * **loc** (*Any**,* *optional*) â Source location information, defaults to None
        * **ip** (*Any**,* *optional*) â Insertion point, defaults to None
        * **kwargs** (*Dict**[**str**,* *Any**]*) â Additional copy atom specific arguments

    Raises:
    :   **TypeError** â If source and destination element type bit widths differ

    Returns:
    :   None

    Return type:
    :   None

    **Examples**:

    ```python
    # Regular copy atom operation
    cute.copy_atom_call(copy_atom, src, dst)

    # Predicated copy atom operation
    cute.copy_atom_call(copy_atom, src, dst, pred=pred)
    ```

cutlass.cute.mma\_atom\_call( : *atom: [MmaAtom](#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")*, : *d: cutlass.cute.typing.Tensor*, : *a: cutlass.cute.typing.Tensor*, : *b: cutlass.cute.typing.Tensor*, : *c: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → None[#](#cutlass.cute.mma_atom_call "Link to this definition")
:   Execute a single MMA atom operation.

    The mma\_atom\_call operation executes an MMA atom with the given operands.
    This performs a matrix multiplication and accumulation operation:
    D = A \* B + C

    Note: The tensors âdâ, âaâ, âbâ, and âcâ must only have a single fragment.

    Parameters:
    :   * **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")) â The MMA atom to execute
        * **d** (*Tensor*) â Destination tensor (output accumulator)
        * **a** (*Tensor*) â First source tensor (matrix A)
        * **b** (*Tensor*) â Second source tensor (matrix B)
        * **c** (*Tensor*) â Third source tensor (input accumulator C)
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

    Examples:

    ```python
    # Call an MMA atom operation
    cute.mma_atom_call(mma_atom, d_tensor, a_tensor, b_tensor, c_tensor)
    ```

cutlass.cute.gemm( : *atom: [MmaAtom](#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")*, : *d: cutlass.cute.typing.Tensor*, : *a: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *b: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *c: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → None[#](#cutlass.cute.gemm "Link to this definition")
:   The GEMM algorithm.

    Computes `D <- A * B + C` where `C` and `D` can alias. Note that some MMA Atoms (e.g.
    warpgroup-wide or tcgen05 MMAs) require manually setting an âaccumulateâ boolean field.

    All tensors must be partitioned according to the provided MMA Atom.

    For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
    election internally. Manual thread selection is not required in such cases.

    Following dispatch rules are supported:

    * Dispatch [1]: (V) x (V) => (V) => (V,1,1) x (V,1,1) => (V,1,1)
    * Dispatch [2]: (M) x (N) => (M,N) => (1,M,1) x (1,N,1) => (1,M,N)
    * Dispatch [3]: (M,K) x (N,K) => (M,N) => (1,M,K) x (1,N,K) => (1,M,N)
    * Dispatch [4]: (V,M) x (V,N) => (V,M,N) => (V,M,1) x (V,N,1) => (V,M,N)
    * Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)

    Operand flexibility:
    - a and b can be a single Tensor (regular GEMM) or a sequence [operand, scale\_factor] for block-scaled GEMM.

    Parameters:
    :   * **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")) â MMA atom
        * **d** (*Tensor*) â Destination tensor
        * **a** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â First source tensor or sequence for advanced modes (e.g., [a, sfa])
        * **b** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â Second source tensor or sequence for advanced modes (e.g., [b, sfb])
        * **c** (*Tensor*) â Third source tensor
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point for MLIR, defaults to None
        * **kwargs** (*dict*) â Additional keyword arguments

    Returns:
    :   None

    Return type:
    :   None

cutlass.cute.copy( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *src: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *dst: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, : *\**, : *pred: cutlass.cute.typing.Tensor | None = None*, : *loc=None*, : *ip=None*, : *\*\*kwargs*, ) → None[#](#cutlass.cute.copy "Link to this definition")
:   Facilitates data transfer between two tensors conforming to layout profile `(V, Rest...)`.

    Parameters:
    :   * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) â Copy atom specifying the transfer operation
        * **src** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â Source tensor or list of tensors with layout profile `(V, Rest...)`
        * **dst** (*Union**[**Tensor**,* *List**[**Tensor**]**,* *Tuple**[**Tensor**,* *...**]**]*) â Destination tensor or list of tensors with layout profile `(V, Rest...)`
        * **pred** (*Optional**[**Tensor**]**,* *optional*) â Optional predication tensor for conditional transfers, defaults to None
        * **loc** (*Any**,* *optional*) â Source location information, defaults to None
        * **ip** (*Any**,* *optional*) â Insertion point, defaults to None
        * **kwargs** (*Dict**[**str**,* *Any**]*) â Additional copy atom specific arguments

    Raises:
    :   * **TypeError** â If source and destination element type bit widths differ
        * **ValueError** â If source and destination ranks differ
        * **ValueError** â If source and destination mode-1 sizes differ
        * **NotImplementedError** â If `V-mode` rank exceeds 2

    Returns:
    :   None

    Return type:
    :   None

    The `V-mode` represents either:

    * A singular mode directly consumable by the provided Copy Atom
    * A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,
      and src/dst layout like `((V, Rest...), Rest...)`

    The algorithm recursively processes the `V-mode`, decomposing it until reaching the minimum granularity
    compatible with the provided Copy Atomâs requirements.

    Source and destination tensors must be partitioned in accordance with the Copy Atom specifications.
    Post-partitioning, both tensors will exhibit a `(V, Rest...)` layout profile.

    The operands src and dst are variadic, each containing a variable number of tensors:

    * For regular copy, src and dst contain single source and destination tensors respectively.
    * For copy with auxiliary operands, src and dst contain the primary tensors followed by
      their respective auxiliary tensors.

    **Precondition:** The size of mode 1 must be equal for both source and destination tensors:
    `size(src, mode=[1]) == size(dst, mode=[1])`

    **Examples**:

    TMA copy operation with multicast functionality:

    ```python
    cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask, cache_policy=policy)
    ```

    Optional predication is supported through an additional tensor parameter. For partitioned tensors with
    logical profile `((ATOM_V,ATOM_REST),REST,...)`, the predication tensor must maintain profile
    compatibility with `(ATOM_REST,REST,...)`.

    For Copy Atoms requiring single-threaded execution, thread election is managed automatically by the
    copy operation. External thread selection mechanisms are not necessary.

    Note

    * Certain Atoms may require additional operation-specific keyword arguments.
    * Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
      for future releases.

cutlass.cute.basic\_copy( : *src: cutlass.cute.typing.Tensor*, : *dst: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.basic_copy "Link to this definition")
:   Performs a basic element-wise copy.

    This functions **assumes** the following pre-conditions:
    1. size(src) == size(dst)

    When the src and dst shapes are static, the pre-conditions are actually verified and the
    element-wise loop is fully unrolled.

    Parameters:
    :   * **src** (*Tensor*) â Source tensor
        * **dst** (*Tensor*) â Destination tensor
        * **loc** (*Optional**[**Location**]**,* *optional*) â Source location for MLIR, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]**,* *optional*) â Insertion point, defaults to None

cutlass.cute.basic\_copy\_if( : *pred: cutlass.cute.typing.Tensor*, : *src: cutlass.cute.typing.Tensor*, : *dst: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.basic_copy_if "Link to this definition")
:   Performs a basic predicated element-wise copy.

    This functions **assumes** the following pre-conditions:
    1. size(src) == size(dst)
    2. size(src) == size(pred)

    When all shapes are static, the pre-conditions are actually verified and the element-wise loop
    is fully unrolled.

cutlass.cute.autovec\_copy( : *src: cutlass.cute.typing.Tensor*, : *dst: cutlass.cute.typing.Tensor*, : *\**, : *l1c\_evict\_priority: [CacheEvictionPriority](cute_nvgpu_common.html#cutlass.cute.nvgpu.CacheEvictionPriority "cutlass.cute.nvgpu.common.CacheEvictionPriority") = cutlass.\_mlir.dialects.cute.CacheEvictionPriority.EVICT\_NORMAL*, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.autovec_copy "Link to this definition")
:   Auto-vectorization SIMT copy policy.

    Given a source and destination tensors that are statically shaped, this policy figures out the
    largest safe vector width that the copy instruction can take and performs the copy.

cutlass.cute.prefetch( : *atom: [CopyAtom](#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *src: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.cute.prefetch "Link to this definition")
:   The Prefetch algorithm.

    The âprefetchâ expects source tensors to be partitioned according to the provided Copy Atom.
    Prefetch is used for loading tensors from global memory to L2.

    Prefetch accepts Copy Atom but not all are allowed. Currently, only supports TMA prefetch.

    ```python
    cute.prefetch(tma_prefetch, src)
    ```

    For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
    election internally. Manual thread selection is not required in such cases.

cutlass.cute.acos( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.acos "Link to this definition")
:   Compute element-wise arc cosine of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the arc cosine of each element in input tensor

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = acos(y)  # Compute arc cosine
    ```

cutlass.cute.asin( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.asin "Link to this definition")
:   Compute element-wise arc sine of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the arc sine of each element in input tensor

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = asin(y)  # Compute arc sine
    ```

cutlass.cute.atan( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.atan "Link to this definition")
:   Compute element-wise arc tangent of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the arc tangent of each element in input tensor

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = atan(y)  # Compute arc tangent
    ```

cutlass.cute.atan2( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *b: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.atan2 "Link to this definition")
:   Compute element-wise arc tangent of two tensors.

    Computes atan2(a, b) element-wise. The function atan2(a, b) is the angle in radians
    between the positive x-axis and the point given by the coordinates (b, a).

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â First input tensor (y-coordinates)
        * **b** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Second input tensor (x-coordinates)
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the arc tangent of a/b element-wise

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    y = cute.make_rmem_tensor(ptr1, layout).load()  # y coordinates
    x = cute.make_rmem_tensor(ptr2, layout).load()  # x coordinates
    theta = atan2(y, x)  # Compute angles
    ```

cutlass.cute.cos( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.cos "Link to this definition")
:   Compute element-wise cosine of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor (in radians)
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the cosine of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = cos(y)  # Compute cosine
    ```

cutlass.cute.erf( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.erf "Link to this definition")
:   Compute element-wise error function of the input tensor.

    The error function is defined as:
    erf(x) = 2/âÏ â«[0 to x] exp(-tÂ²) dt

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the error function value for each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = erf(y)  # Compute error function
    ```

cutlass.cute.exp( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.exp "Link to this definition")
:   Compute element-wise exponential of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the exponential of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = exp(y)  # Compute exponential
    ```

cutlass.cute.exp2( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.exp2 "Link to this definition")
:   Compute element-wise base-2 exponential of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing 2 raised to the power of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = exp2(y)  # Compute 2^x
    ```

cutlass.cute.log( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.log "Link to this definition")
:   Compute element-wise natural logarithm of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the natural logarithm of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = log(y)  # Compute natural logarithm
    ```

cutlass.cute.log10( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.log10 "Link to this definition")
:   Compute element-wise base-10 logarithm of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the base-10 logarithm of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = log10(y)  # Compute log base 10
    ```

cutlass.cute.log2( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.log2 "Link to this definition")
:   Compute element-wise base-2 logarithm of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the base-2 logarithm of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = log2(y)  # Compute log base 2
    ```

cutlass.cute.rsqrt( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.rsqrt "Link to this definition")
:   Compute element-wise reciprocal square root of the input tensor.

    Computes 1/âx element-wise.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the reciprocal square root of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = rsqrt(y)  # Compute 1/âx
    ```

cutlass.cute.sin( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.sin "Link to this definition")
:   Compute element-wise sine of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor (in radians)
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the sine of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = sin(y)  # Compute sine
    ```

cutlass.cute.sqrt( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.sqrt "Link to this definition")
:   Compute element-wise square root of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the square root of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = sqrt(y)  # Compute square root
    ```

cutlass.cute.tan( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.tan "Link to this definition")
:   Compute element-wise tangent of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor (in radians)
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the tangent of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = tan(y)  # Compute tangent
    ```

cutlass.cute.tanh( : *a: [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric*, : *fastmath: bool = False*, : *\**, : *loc=None*, : *ip=None*, ) → [TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.tensor.TensorSSA") | cutlass.cute.typing.Numeric[#](#cutlass.cute.tanh "Link to this definition")
:   Compute element-wise hyperbolic tangent of the input tensor.

    Parameters:
    :   * **a** (*Union**[*[*TensorSSA*](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")*,* *Numeric**]*) â Input tensor
        * **fastmath** (*bool**,* *optional*) â Enable fast math optimizations, defaults to False
        * **loc** (*Optional**[**Location**]*) â Source location information, defaults to None
        * **ip** (*Optional**[**InsertionPoint**]*) â Insertion point for IR generation, defaults to None

    Returns:
    :   Tensor containing the hyperbolic tangent of each element

    Return type:
    :   Union[[TensorSSA](#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

    Example:

    ```python
    x = cute.make_rmem_tensor(layout)  # Create tensor
    y = x.load()  # Load values
    z = tanh(y)  # Compute hyperbolic tangent
    ```

*class* cutlass.cute.ffi(*\**, *name: str*, *params\_types: list = []*, *return\_type=None*)[#](#cutlass.cute.ffi "Link to this definition")
:   Bases: `object`

    Foreign Function Interface (FFI) wrapper for external function invocation in the CUTLASS Python DSL.

    This class enables calling external MLIR function prototypes from Python code, handling type conversion,
    prototype registration, and dynamic insertion of function symbols into MLIR modules as needed.

    Parameters:
    :   * **name** (*str*) â Name of the external function. This will be used as the symbol name when calling or registering a prototype in the MLIR module.
        * **params\_types** (*list**,* *optional*) â List of argument types for the external function. These can be CUTLASS numeric types, numeric meta types, or types convertible via get\_mlir\_types.
        * **return\_type** (*optional*) â The return type of the external function. If not specified, the function is assumed to have no return value.

    \_\_call\_\_(*\*args*)[#](#cutlass.cute.ffi.__call__ "Link to this definition")
    :   Calls the external function with the given arguments, ensuring argument and result types match the prototype.

    \_\_init\_\_(*\**, *name: str*, *params\_types: list = []*, *return\_type=None*)[#](#cutlass.cute.ffi.__init__ "Link to this definition")

    \_get\_prototype\_region(*current\_op*)[#](#cutlass.cute.ffi._get_prototype_region "Link to this definition")
    :   Helper method to determine the appropriate MLIR module and region for inserting a function prototype.

        This method recursively traverses the current operationâs parent hierarchy to find the correct module
        and region where the function prototype should be inserted. It supports both builtin.module and gpu.module.
        :param current\_op: The current operation to check.
        :type current\_op: Operation

        Returns:
        :   A tuple containing the module operation and the insertion region.

        Return type:
        :   tuple

    *static* \_to\_mlir\_types(*args*)[#](#cutlass.cute.ffi._to_mlir_types "Link to this definition")
    :   Helper method to convert a list of arguments to their corresponding MLIR types.

        This method converts CUTLASS numeric types, numeric meta types, and types convertible via get\_mlir\_types
        to their corresponding MLIR types.
        :param args: The list of arguments to convert to MLIR types.
        :type args: list

        Returns:
        :   A list of MLIR types.

        Return type:
        :   list

    *static* \_type\_check(*callee*, *exec\_types*, *returns\_types*)[#](#cutlass.cute.ffi._type_check "Link to this definition")
    :   Helper method to check if the function prototype types match the expected types.

        This method compares the input and output types of the function prototype with the provided expected types.
        :param callee: The function prototype operation to check.
        :type callee: func.FuncOp
        :param exec\_types: The expected input types.
        :type exec\_types: list
        :param returns\_types: The expected output types.
        :type returns\_types: list

    \_create\_prototype\_in\_region(*op*, *region*, *exec\_args*)[#](#cutlass.cute.ffi._create_prototype_in_region "Link to this definition")
    :   Helper method to create or retrieve a function prototype in the current module.

        This method checks if a function prototype with the given name already exists in the symbol table of the current module.
        If it does, it checks if the prototypeâs types match the expected types. If it does not, it raises an error.
        If it does not exist, it creates a new function prototype and inserts it into the current region.
        :param op: The module operation to check.
        :type op: Operation
        :param region: The region to insert the function prototype into.
        :type region: Region
        :param exec\_args: The arguments to pass to the function prototype.
        :type exec\_args: list

[previous

Changelog for CuTe DSL API changes](changelog.html "previous page")
[next

arch](cute_arch.html "next page")

On this page

* [`Swizzle`](#cutlass.cute.Swizzle)
* [`struct`](#cutlass.cute.struct)
  + [`_MemRangeMeta`](#cutlass.cute.struct._MemRangeMeta)
    - [`_dtype`](#cutlass.cute.struct._MemRangeMeta._dtype)
    - [`_size`](#cutlass.cute.struct._MemRangeMeta._size)
    - [`size`](#cutlass.cute.struct._MemRangeMeta.size)
    - [`elem_width`](#cutlass.cute.struct._MemRangeMeta.elem_width)
    - [`size_in_bytes`](#cutlass.cute.struct._MemRangeMeta.size_in_bytes)
  + [`MemRange`](#cutlass.cute.struct.MemRange)
    - [`_dtype`](#cutlass.cute.struct.MemRange._dtype)
    - [`_size`](#cutlass.cute.struct.MemRange._size)
  + [`_MemRangeData`](#cutlass.cute.struct._MemRangeData)
    - [`__init__()`](#cutlass.cute.struct._MemRangeData.__init__)
    - [`data_ptr()`](#cutlass.cute.struct._MemRangeData.data_ptr)
    - [`get_tensor()`](#cutlass.cute.struct._MemRangeData.get_tensor)
  + [`_AlignMeta`](#cutlass.cute.struct._AlignMeta)
    - [`_dtype`](#cutlass.cute.struct._AlignMeta._dtype)
    - [`_align`](#cutlass.cute.struct._AlignMeta._align)
    - [`dtype`](#cutlass.cute.struct._AlignMeta.dtype)
    - [`align`](#cutlass.cute.struct._AlignMeta.align)
  + [`Align`](#cutlass.cute.struct.Align)
    - [`_dtype`](#cutlass.cute.struct.Align._dtype)
    - [`_align`](#cutlass.cute.struct.Align._align)
  + [`_ScalarData`](#cutlass.cute.struct._ScalarData)
    - [`__init__()`](#cutlass.cute.struct._ScalarData.__init__)
    - [`to_llvm_ptr()`](#cutlass.cute.struct._ScalarData.to_llvm_ptr)
    - [`ptr`](#cutlass.cute.struct._ScalarData.ptr)
    - [`dtype`](#cutlass.cute.struct._ScalarData.dtype)
    - [`value`](#cutlass.cute.struct._ScalarData.value)
  + [`_is_scalar_type()`](#cutlass.cute.struct._is_scalar_type)
  + [`__init__()`](#cutlass.cute.struct.__init__)
  + [`size_in_bytes()`](#cutlass.cute.struct.size_in_bytes)
  + [`align_offset()`](#cutlass.cute.struct.align_offset)
* [`E()`](#cutlass.cute.E)
* [`get_divisibility()`](#cutlass.cute.get_divisibility)
* [`is_static()`](#cutlass.cute.is_static)
* [`has_underscore()`](#cutlass.cute.has_underscore)
* [`pretty_str()`](#cutlass.cute.pretty_str)
* [`printf()`](#cutlass.cute.printf)
* [`front()`](#cutlass.cute.front)
* [`is_major()`](#cutlass.cute.is_major)
* [`assume()`](#cutlass.cute.assume)
* [`make_swizzle()`](#cutlass.cute.make_swizzle)
* [`static()`](#cutlass.cute.static)
* [`get_leaves()`](#cutlass.cute.get_leaves)
* [`depth()`](#cutlass.cute.depth)
* [`rank()`](#cutlass.cute.rank)
* [`is_congruent()`](#cutlass.cute.is_congruent)
* [`is_weakly_congruent()`](#cutlass.cute.is_weakly_congruent)
* [`get()`](#cutlass.cute.get)
* [`select()`](#cutlass.cute.select)
* [`group_modes()`](#cutlass.cute.group_modes)
* [`slice_()`](#cutlass.cute.slice_)
* [`prepend()`](#cutlass.cute.prepend)
* [`append()`](#cutlass.cute.append)
* [`prepend_ones()`](#cutlass.cute.prepend_ones)
* [`append_ones()`](#cutlass.cute.append_ones)
* [`repeat_as_tuple()`](#cutlass.cute.repeat_as_tuple)
* [`repeat()`](#cutlass.cute.repeat)
* [`repeat_like()`](#cutlass.cute.repeat_like)
* [`flatten()`](#cutlass.cute.flatten)
* [`filter_zeros()`](#cutlass.cute.filter_zeros)
* [`filter()`](#cutlass.cute.filter)
* [`size()`](#cutlass.cute.size)
* [`shape_div()`](#cutlass.cute.shape_div)
* [`ceil_div()`](#cutlass.cute.ceil_div)
* [`round_up()`](#cutlass.cute.round_up)
* [`make_layout()`](#cutlass.cute.make_layout)
* [`make_identity_layout()`](#cutlass.cute.make_identity_layout)
* [`make_ordered_layout()`](#cutlass.cute.make_ordered_layout)
* [`make_layout_like()`](#cutlass.cute.make_layout_like)
* [`make_composed_layout()`](#cutlass.cute.make_composed_layout)
* [`cosize()`](#cutlass.cute.cosize)
* [`size_in_bytes()`](#cutlass.cute.size_in_bytes)
* [`coalesce()`](#cutlass.cute.coalesce)
* [`crd2idx()`](#cutlass.cute.crd2idx)
* [`idx2crd()`](#cutlass.cute.idx2crd)
* [`recast_layout()`](#cutlass.cute.recast_layout)
* [`slice_and_offset()`](#cutlass.cute.slice_and_offset)
* [`recast_ptr()`](#cutlass.cute.recast_ptr)
* [`make_ptr()`](#cutlass.cute.make_ptr)
* [`composition()`](#cutlass.cute.composition)
* [`complement()`](#cutlass.cute.complement)
* [`right_inverse()`](#cutlass.cute.right_inverse)
* [`left_inverse()`](#cutlass.cute.left_inverse)
* [`logical_product()`](#cutlass.cute.logical_product)
* [`zipped_product()`](#cutlass.cute.zipped_product)
* [`tiled_product()`](#cutlass.cute.tiled_product)
* [`flat_product()`](#cutlass.cute.flat_product)
* [`raked_product()`](#cutlass.cute.raked_product)
* [`blocked_product()`](#cutlass.cute.blocked_product)
* [`logical_divide()`](#cutlass.cute.logical_divide)
* [`zipped_divide()`](#cutlass.cute.zipped_divide)
* [`tiled_divide()`](#cutlass.cute.tiled_divide)
* [`flat_divide()`](#cutlass.cute.flat_divide)
* [`max_common_layout()`](#cutlass.cute.max_common_layout)
* [`max_common_vector()`](#cutlass.cute.max_common_vector)
* [`tile_to_shape()`](#cutlass.cute.tile_to_shape)
* [`local_partition()`](#cutlass.cute.local_partition)
* [`local_tile()`](#cutlass.cute.local_tile)
* [`make_layout_image_mask()`](#cutlass.cute.make_layout_image_mask)
* [`leading_dim()`](#cutlass.cute.leading_dim)
* [`make_layout_tv()`](#cutlass.cute.make_layout_tv)
* [`get_nonswizzle_portion()`](#cutlass.cute.get_nonswizzle_portion)
* [`get_swizzle_portion()`](#cutlass.cute.get_swizzle_portion)
* [`transform_leaf()`](#cutlass.cute.transform_leaf)
* [`find_if()`](#cutlass.cute.find_if)
* [`find()`](#cutlass.cute.find)
* [`flatten_to_tuple()`](#cutlass.cute.flatten_to_tuple)
* [`unflatten()`](#cutlass.cute.unflatten)
* [`product()`](#cutlass.cute.product)
* [`product_like()`](#cutlass.cute.product_like)
* [`product_each()`](#cutlass.cute.product_each)
* [`elem_less()`](#cutlass.cute.elem_less)
* [`tuple_cat()`](#cutlass.cute.tuple_cat)
* [`transform_apply()`](#cutlass.cute.transform_apply)
* [`filter_tuple()`](#cutlass.cute.filter_tuple)
* [`TensorSSA`](#cutlass.cute.TensorSSA)
  + [`__init__()`](#cutlass.cute.TensorSSA.__init__)
  + [`dtype`](#cutlass.cute.TensorSSA.dtype)
  + [`element_type`](#cutlass.cute.TensorSSA.element_type)
  + [`shape`](#cutlass.cute.TensorSSA.shape)
  + [`_apply_op()`](#cutlass.cute.TensorSSA._apply_op)
  + [`apply_op()`](#cutlass.cute.TensorSSA.apply_op)
  + [`broadcast_to()`](#cutlass.cute.TensorSSA.broadcast_to)
  + [`_flatten_shape_and_coord()`](#cutlass.cute.TensorSSA._flatten_shape_and_coord)
  + [`_build_result()`](#cutlass.cute.TensorSSA._build_result)
  + [`reshape()`](#cutlass.cute.TensorSSA.reshape)
  + [`to()`](#cutlass.cute.TensorSSA.to)
  + [`ir_value()`](#cutlass.cute.TensorSSA.ir_value)
  + [`ir_value_int8()`](#cutlass.cute.TensorSSA.ir_value_int8)
  + [`reduce()`](#cutlass.cute.TensorSSA.reduce)
* [`make_tensor()`](#cutlass.cute.make_tensor)
* [`make_identity_tensor()`](#cutlass.cute.make_identity_tensor)
* [`make_fragment()`](#cutlass.cute.make_fragment)
* [`make_fragment_like()`](#cutlass.cute.make_fragment_like)
* [`make_rmem_tensor_like()`](#cutlass.cute.make_rmem_tensor_like)
* [`make_rmem_tensor()`](#cutlass.cute.make_rmem_tensor)
* [`recast_tensor()`](#cutlass.cute.recast_tensor)
* [`domain_offset()`](#cutlass.cute.domain_offset)
* [`print_tensor()`](#cutlass.cute.print_tensor)
* [`full()`](#cutlass.cute.full)
* [`full_like()`](#cutlass.cute.full_like)
* [`empty_like()`](#cutlass.cute.empty_like)
* [`ones_like()`](#cutlass.cute.ones_like)
* [`zeros_like()`](#cutlass.cute.zeros_like)
* [`where()`](#cutlass.cute.where)
* [`any_()`](#cutlass.cute.any_)
* [`all_()`](#cutlass.cute.all_)
* [`Atom`](#cutlass.cute.Atom)
  + [`__init__()`](#cutlass.cute.Atom.__init__)
  + [`op`](#cutlass.cute.Atom.op)
  + [`type`](#cutlass.cute.Atom.type)
  + [`set()`](#cutlass.cute.Atom.set)
  + [`get()`](#cutlass.cute.Atom.get)
  + [`with_()`](#cutlass.cute.Atom.with_)
  + [`_unpack()`](#cutlass.cute.Atom._unpack)
  + [`_abc_impl`](#cutlass.cute.Atom._abc_impl)
* [`MmaAtom`](#cutlass.cute.MmaAtom)
  + [`thr_id`](#cutlass.cute.MmaAtom.thr_id)
  + [`shape_mnk`](#cutlass.cute.MmaAtom.shape_mnk)
  + [`tv_layout_A`](#cutlass.cute.MmaAtom.tv_layout_A)
  + [`tv_layout_B`](#cutlass.cute.MmaAtom.tv_layout_B)
  + [`tv_layout_C`](#cutlass.cute.MmaAtom.tv_layout_C)
  + [`make_fragment_A()`](#cutlass.cute.MmaAtom.make_fragment_A)
  + [`make_fragment_B()`](#cutlass.cute.MmaAtom.make_fragment_B)
  + [`make_fragment_C()`](#cutlass.cute.MmaAtom.make_fragment_C)
  + [`_abc_impl`](#cutlass.cute.MmaAtom._abc_impl)
* [`CopyAtom`](#cutlass.cute.CopyAtom)
  + [`value_type`](#cutlass.cute.CopyAtom.value_type)
  + [`thr_id`](#cutlass.cute.CopyAtom.thr_id)
  + [`layout_src_tv`](#cutlass.cute.CopyAtom.layout_src_tv)
  + [`layout_dst_tv`](#cutlass.cute.CopyAtom.layout_dst_tv)
  + [`smem_layout`](#cutlass.cute.CopyAtom.smem_layout)
  + [`_abc_impl`](#cutlass.cute.CopyAtom._abc_impl)
* [`TiledCopy`](#cutlass.cute.TiledCopy)
  + [`layout_tv_tiled`](#cutlass.cute.TiledCopy.layout_tv_tiled)
  + [`tiler_mn`](#cutlass.cute.TiledCopy.tiler_mn)
  + [`layout_src_tv_tiled`](#cutlass.cute.TiledCopy.layout_src_tv_tiled)
  + [`layout_dst_tv_tiled`](#cutlass.cute.TiledCopy.layout_dst_tv_tiled)
  + [`size`](#cutlass.cute.TiledCopy.size)
  + [`get_slice()`](#cutlass.cute.TiledCopy.get_slice)
  + [`retile()`](#cutlass.cute.TiledCopy.retile)
  + [`_abc_impl`](#cutlass.cute.TiledCopy._abc_impl)
* [`TiledMma`](#cutlass.cute.TiledMma)
  + [`tv_layout_A_tiled`](#cutlass.cute.TiledMma.tv_layout_A_tiled)
  + [`tv_layout_B_tiled`](#cutlass.cute.TiledMma.tv_layout_B_tiled)
  + [`tv_layout_C_tiled`](#cutlass.cute.TiledMma.tv_layout_C_tiled)
  + [`permutation_mnk`](#cutlass.cute.TiledMma.permutation_mnk)
  + [`thr_layout_vmnk`](#cutlass.cute.TiledMma.thr_layout_vmnk)
  + [`size`](#cutlass.cute.TiledMma.size)
  + [`get_tile_size()`](#cutlass.cute.TiledMma.get_tile_size)
  + [`get_slice()`](#cutlass.cute.TiledMma.get_slice)
  + [`_partition_shape()`](#cutlass.cute.TiledMma._partition_shape)
  + [`partition_shape_A()`](#cutlass.cute.TiledMma.partition_shape_A)
  + [`partition_shape_B()`](#cutlass.cute.TiledMma.partition_shape_B)
  + [`partition_shape_C()`](#cutlass.cute.TiledMma.partition_shape_C)
  + [`_thrfrg()`](#cutlass.cute.TiledMma._thrfrg)
  + [`_thrfrg_A()`](#cutlass.cute.TiledMma._thrfrg_A)
  + [`_thrfrg_B()`](#cutlass.cute.TiledMma._thrfrg_B)
  + [`_thrfrg_C()`](#cutlass.cute.TiledMma._thrfrg_C)
  + [`_abc_impl`](#cutlass.cute.TiledMma._abc_impl)
* [`ThrMma`](#cutlass.cute.ThrMma)
  + [`__init__()`](#cutlass.cute.ThrMma.__init__)
  + [`thr_idx`](#cutlass.cute.ThrMma.thr_idx)
  + [`partition_A()`](#cutlass.cute.ThrMma.partition_A)
  + [`partition_B()`](#cutlass.cute.ThrMma.partition_B)
  + [`partition_C()`](#cutlass.cute.ThrMma.partition_C)
  + [`_abc_impl`](#cutlass.cute.ThrMma._abc_impl)
* [`ThrCopy`](#cutlass.cute.ThrCopy)
  + [`__init__()`](#cutlass.cute.ThrCopy.__init__)
  + [`thr_idx`](#cutlass.cute.ThrCopy.thr_idx)
  + [`partition_S()`](#cutlass.cute.ThrCopy.partition_S)
  + [`partition_D()`](#cutlass.cute.ThrCopy.partition_D)
  + [`_abc_impl`](#cutlass.cute.ThrCopy._abc_impl)
* [`make_atom()`](#cutlass.cute.make_atom)
* [`make_mma_atom()`](#cutlass.cute.make_mma_atom)
* [`make_tiled_mma()`](#cutlass.cute.make_tiled_mma)
* [`make_copy_atom()`](#cutlass.cute.make_copy_atom)
* [`make_tiled_copy_tv()`](#cutlass.cute.make_tiled_copy_tv)
* [`make_tiled_copy()`](#cutlass.cute.make_tiled_copy)
* [`make_tiled_copy_S()`](#cutlass.cute.make_tiled_copy_S)
* [`make_tiled_copy_D()`](#cutlass.cute.make_tiled_copy_D)
* [`make_tiled_copy_A()`](#cutlass.cute.make_tiled_copy_A)
* [`make_tiled_copy_B()`](#cutlass.cute.make_tiled_copy_B)
* [`make_tiled_copy_C()`](#cutlass.cute.make_tiled_copy_C)
* [`make_tiled_copy_C_atom()`](#cutlass.cute.make_tiled_copy_C_atom)
* [`make_cotiled_copy()`](#cutlass.cute.make_cotiled_copy)
* [`copy_atom_call()`](#cutlass.cute.copy_atom_call)
* [`mma_atom_call()`](#cutlass.cute.mma_atom_call)
* [`gemm()`](#cutlass.cute.gemm)
* [`copy()`](#cutlass.cute.copy)
* [`basic_copy()`](#cutlass.cute.basic_copy)
* [`basic_copy_if()`](#cutlass.cute.basic_copy_if)
* [`autovec_copy()`](#cutlass.cute.autovec_copy)
* [`prefetch()`](#cutlass.cute.prefetch)
* [`acos()`](#cutlass.cute.acos)
* [`asin()`](#cutlass.cute.asin)
* [`atan()`](#cutlass.cute.atan)
* [`atan2()`](#cutlass.cute.atan2)
* [`cos()`](#cutlass.cute.cos)
* [`erf()`](#cutlass.cute.erf)
* [`exp()`](#cutlass.cute.exp)
* [`exp2()`](#cutlass.cute.exp2)
* [`log()`](#cutlass.cute.log)
* [`log10()`](#cutlass.cute.log10)
* [`log2()`](#cutlass.cute.log2)
* [`rsqrt()`](#cutlass.cute.rsqrt)
* [`sin()`](#cutlass.cute.sin)
* [`sqrt()`](#cutlass.cute.sqrt)
* [`tan()`](#cutlass.cute.tan)
* [`tanh()`](#cutlass.cute.tanh)
* [`ffi`](#cutlass.cute.ffi)
  + [`__call__()`](#cutlass.cute.ffi.__call__)
  + [`__init__()`](#cutlass.cute.ffi.__init__)
  + [`_get_prototype_region()`](#cutlass.cute.ffi._get_prototype_region)
  + [`_to_mlir_types()`](#cutlass.cute.ffi._to_mlir_types)
  + [`_type_check()`](#cutlass.cute.ffi._type_check)
  + [`_create_prototype_in_region()`](#cutlass.cute.ffi._create_prototype_in_region)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.