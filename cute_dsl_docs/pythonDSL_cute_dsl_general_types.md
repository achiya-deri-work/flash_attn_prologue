

Types — NVIDIA CUTLASS Documentation



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
  + [JIT Types](#)
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
* Types

# Types[#](#id1 "Link to this heading")

## Overview[#](#overview "Link to this heading")

CuTe DSL provides a set of core types that form the foundation of tensor layout algebra and GPU programming. These types enable precise control over memory layout, data representation, and tensor operations. This document covers the key types available in `cutlass.cute.core`.

## Core Numeric Types[#](#core-numeric-types "Link to this heading")

### IntValue[#](#intvalue "Link to this heading")

`IntValue` is an internal representation of constrained integer types with divisibility information. It serves as a proxy for constrained integer types in the CuTe IR, automatically tracking divisibility constraints that are crucial for layout operations.

**Key Features:**

* Inherits from `ArithValue` with extensions for divisibility tracking
* Automatically emits `cute.get_scalars` operations in the IR
* Supports arithmetic operations that propagate divisibility information
* Used internally for type-safe integer operations in layout algebra

**API Methods:**

* `get_typed_value()` - Returns the value as an IntTupleType
* `get_divisibility()` - Returns the divisibility constraint of the value
* `divisibility` - Property that returns the divisibility constraint

**Supported Operations:**

The `IntValue` type supports standard arithmetic operations with divisibility tracking:

```python
# Addition, subtraction, multiplication, division, and modulo
result = int_val1 + int_val2
result = int_val1 - int_val2
result = int_val1 * int_val2
result = int_val1 // int_val2
result = int_val1 % int_val2
```

**String Representation:**

```python
# IntValue with divisibility 1
str(int_val)  # Returns "?"

# IntValue with divisibility 4
str(int_val)  # Returns "?{div=4}"
```

### Ratio[#](#ratio "Link to this heading")

`Ratio` represents a rational number as a ratio of two integers. It is used in CuTe to represent exact fractional values that arise in tensor layout operations, particularly in composition operations where divisibility conditions may not be satisfied.

**Constructor:**

```python
ratio = cute.Ratio(numerator, denominator)
```

param numerator:
:   The numerator of the ratio

type numerator:
:   int

param denominator:
:   The denominator of the ratio

type denominator:
:   int

raises TypeError:
:   If numerator or denominator are not integers

**Methods:**

* `is_integral()` - Returns `True` if the ratio represents an integer value (numerator divisible by denominator)
* `reduced()` - Returns a new Ratio with numerator and denominator reduced to lowest terms
* `to(dtype)` - Converts the ratio to another type (Ratio, float, or int)

**Arithmetic Operations:**

```python
# Multiplication with another ratio
ratio1 = cute.Ratio(1, 2)
ratio2 = cute.Ratio(3, 4)
result = ratio1 * ratio2  # Returns Ratio(3, 8)

# Multiplication with integer
ratio = cute.Ratio(2, 3)
result = ratio * 5  # Returns Ratio(10, 3)
result = 5 * ratio  # Returns Ratio(10, 3)
```

**Type Conversion:**

```python
ratio = cute.Ratio(3, 2)

# Convert to float
float_val = ratio.to(float)  # Returns 1.5

# Convert to int (floor division)
int_val = ratio.to(int)  # Returns 1
```

## Layout Algebra Types[#](#layout-algebra-types "Link to this heading")

### ScaledBasis[#](#scaledbasis "Link to this heading")

`ScaledBasis` represents a scaled basis element in CuTeâs layout algebra. It consists of a scale value and a mode that identifies which basis element in the layout algebra is being referenced. ScaledBasis elements are fundamental to CuTeâs coordinate system representation.

**Constructor:**

```python
sb = cute.ScaledBasis(value, mode)
```

param value:
:   The scale value

type value:
:   Union[int, Integer, Ratio, ir.Value]

param mode:
:   The mode identifying the basis element

type mode:
:   Union[int, List[int]]

raises TypeError:
:   If mode is not an integer or list of integers

**Examples:**

```python
# Create a scaled basis with integer scale and mode
sb1 = cute.ScaledBasis(2, 0)  # 2 * E(0)

# Create a scaled basis with a Ratio scale
sb2 = cute.ScaledBasis(cute.Ratio(1, 2), 1)  # (1/2) * E(1)

# Create a scaled basis with a list of modes
sb3 = cute.ScaledBasis(4, [0, 1])  # 4 * E([0, 1])

# Scaled basis elements are commonly used in layout strides
layout = cute.make_layout((4, 8), stride=(cute.ScaledBasis(2, 0), cute.ScaledBasis(1, 1)))

# This creates a layout with strides (2@0, 1@1) representing
# a coordinate system where each dimension has its own basis

# Example: Mapping coordinates to indices using the layout
coord = (2, 3)
idx = cute.crd2idx(coord, layout)  # Maps (2, 3) to (4, 3)
```

**Properties:**

* `value` - Get the scale value
* `mode` - Get the mode as a list of integers
* `is_static()` - Returns `True` if the value is statically known

**Methods:**

* `to(dtype)` - Convert to another type (ScaledBasis or internal \_ScaledBasis)

**Operations:**

```python
# Right multiplication by a scale factor
sb = cute.ScaledBasis(2, 0)
result = 3 * sb  # Creates ScaledBasis(6, 0)
```

**Utility Function:**

```python
# Create a basis element with unit scale
basis = cute.E(mode)  # Equivalent to ScaledBasis(1, mode)
```

### Swizzle[#](#swizzle "Link to this heading")

`Swizzle` is a transformation that permutes the elements of a layout. Swizzles are used to rearrange data elements to improve memory access patterns and computational efficiency, particularly for avoiding bank conflicts in shared memory.

**Swizzle Parameters:**

A swizzle is defined by three parameters:

* **MBase**: The number of least-significant bits to keep constant
* **BBits**: The number of bits in the mask
* **SShift**: The distance to shift the mask

**Bit Pattern:**

```python
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase (least-sig bits kept constant)
                 ^-^       ^-^     BBits (number of bits in mask)
                   ^---------^     SShift (distance to shift YYY)
                                      (positive: right, negative: left)

Given:    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
Result:   0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx
          where AA = ZZ xor YY
```

**Usage:**

Swizzles are typically created using CuTeâs swizzle factory functions and composed with layouts to create optimized memory access patterns.

### Layout[#](#layout "Link to this heading")

`Layout` is CuTeâs core abstraction for representing tensor layouts. A Layout maps from a logical coordinate space to an index space, defined by a pair of (Shape, Stride). Layouts present a common interface to multidimensional array access that abstracts away the details of how array elements are organized in memory.

**Key Concepts:**

* **Shape**: Defines the abstract dimensions of the Layout
* **Stride**: Defines how coordinates within the Shape map to linear indices
* **Hierarchical Structure**: CuTe layouts are inherently hierarchical, constructed from smaller nested layouts

**Properties:**

* `shape` - An IntTuple representing the dimensions of the layout
* `stride` - An IntTuple representing the strides of the layout
* `max_alignment` - The maximum alignment of the layout in bytes

**Examples:**

```python
# Creating a layout with shape (4,8) and default stride (column major)
layout = cute.make_layout((4, 8))

# Creating a layout with explicit shape and stride (row major)
layout = cute.make_layout((4, 8), stride=(8, 1))

# Accessing layout properties
shape = layout.shape      # Returns (4, 8)
stride = layout.stride    # Returns (8, 1)

# Mapping a coordinate to an index: (2, 3) -> 2 * 8 + 3 * 1 = 19
idx = cute.crd2idx((2, 3), layout)
```

**Layout Operations:**

Layouts support a rich algebra of operations:

* **Concatenation**: Combining layouts along dimensions
* **Coalescence**: Merging adjacent modes
* **Composition**: Composing layouts with functions or other layouts
* **Complement**: Computing the complement space
* **Inversion**: Inverting the layout mapping

**String Representation:**

```python
layout = cute.make_layout((4, 8), stride=(1, 4))
print(layout)  # Prints "shape:stride" format, e.g., "(4,8):(1,4)"
```

### ComposedLayout[#](#composedlayout "Link to this heading")

`ComposedLayout` represents a composition of layouts and transformations. It is a generalization of normal layouts that can support arbitrary function mappings from coordinate to coordinate as an inner layout.

**Structure:**

A ComposedLayout consists of three components:

* **inner**: The inner transformation (Swizzle or Layout)
* **offset**: An offset applied to coordinates
* **outer**: The outer layout

**Properties:**

* `inner` - Returns the inner transformation (Union[Swizzle, Layout])
* `offset` - Returns the offset as an IntTuple
* `outer` - Returns the outer layout
* `shape` - Returns the shape of the composed layout
* `max_alignment` - Returns the maximum alignment
* `is_normal` - Returns `True` if this is a normal layout (not a general composition)

**Examples:**

```python
# ComposedLayouts are typically created through composition operations
# For example, composing a layout with a swizzle
layout = cute.make_layout((8, 8))
swizzle = cute.make_swizzle(...)
composed = cute.composition(swizzle, layout)

# Accessing components
inner = composed.inner      # Returns the swizzle
outer = composed.outer      # Returns the layout
offset = composed.offset    # Returns the offset
```

**String Representation:**

```python
print(composed)  # Prints "inner o offset o outer" format
```

## Memory and Pointer Types[#](#memory-and-pointer-types "Link to this heading")

### Pointer[#](#pointer "Link to this heading")

`Pointer` represents a memory address with specific properties. Pointers are a fundamental type of iterator/engine that support random-access operations. They can be offset by elements of a layoutâs codomain and dereferenced to produce values.

**Properties:**

* `dtype` - The type of value this pointer points to
* `type` - The MLIR type of the pointer
* `memspace` - The memory space where the pointer data resides (e.g., `gmem`, `smem`, `rmem`)
* `alignment` - The alignment of the pointer in bytes
* `max_alignment` - The maximum alignment of the pointer in bytes

**Operations:**

```python
# Pointer arithmetic
ptr2 = ptr + offset     # Offset pointer forward
ptr3 = offset + ptr     # Offset pointer forward (commutative)
ptr4 = ptr - offset     # Offset pointer backward

# Convert pointer to integer
int_addr = ptr.toint()

# Align pointer to specified byte boundary
aligned_ptr = ptr.align(16)  # Align to 16-byte boundary
```

**Tensor Composition:**

When composed with a layout, a pointer forms a tensor: `T = E â L`, where `E` is the pointer (engine) and `L` is the layout. The tensor evaluates the layout by mapping a coordinate `c` to the codomain, offsets the pointer accordingly, and dereferences the result:

```python
T(c) = (E â L)(c) = *(E + L(c))
```

**Methods:**

* `llvm_ptr` - Get the LLVM pointer representation (low-level use only)
* `align(min_align)` - Align pointer to specified byte alignment (must be power of 2)
* `toint()` - Convert pointer to integer address (Int64 for gmem/generic, Int32 otherwise)

**Examples:**

```python
# Create a pointer from a tensor's data
ptr = tensor.data()

# Offset the pointer
offset_ptr = ptr + 16

# Check pointer properties
print(f"Memory space: {ptr.memspace}")
print(f"Alignment: {ptr.alignment}")
print(f"Data type: {ptr.dtype}")
```

## Structured Data Types[#](#structured-data-types "Link to this heading")

### struct[#](#struct "Link to this heading")

The `struct` decorator abstracts C structures in Python DSL. It allows you to define structured data types with precise control over layout, alignment, and nesting.

**Supported Elements:**

* Base DSL scalar int/float elements
* Arrays (MemRange)
* Nested structures
* Aligned elements

**Basic Usage:**

```python
# Define a simple struct
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32

# Define a struct with arrays and nested structures
@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16
```

**Alignment Control:**

```python
# Define a struct with explicit alignment
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
```

**Static Queries:**

```python
# Get size and alignment at compile time
size = StorageB.__sizeof__()
align = StorageB.__alignof__()
```

**Allocation and Access:**

```python
# Allocate and reference elements
storage = allocator.allocate(StorageB)

# Access struct members
storage.a[0] = ...
storage.x = ...
... = storage.compA.real.ptr
... = storage.x.ptr.load()
```

**Methods:**

* `__sizeof__()` - Returns the size of the struct in bytes
* `__alignof__()` - Returns the alignment of the struct in bytes
* `size_in_bytes()` - Returns the size of the struct in bytes

#### struct.MemRange[#](#struct-memrange "Link to this heading")

`MemRange` defines a contiguous range of memory with a specific element type and size.

**Syntax:**

```python
cute.struct.MemRange[dtype, size]
```

param dtype:
:   The data type (must be a DSL scalar type)

type dtype:
:   Type[Numeric]

param size:
:   The number of elements in the range

type size:
:   int

**Properties:**

* `size` - Number of elements in the range
* `elem_width` - Width of each element in bits
* `size_in_bytes` - Total size in bytes

**Methods:**

* `data_ptr()` - Returns a pointer to the start of the memory range
* `get_tensor(layout, swizzle=None, dtype=None)` - Creates a tensor from the memory range
* `__getitem__(index)` - Returns the element at the specified index

**Examples:**

```python
@cute.struct
class Buffer:
    data : cute.struct.MemRange[cutlass.Float32, 128]

# Allocate buffer
buf = allocator.allocate(Buffer)

# Get pointer to data
ptr = buf.data.data_ptr()

# Access individual elements
element = buf.data[5]

# Create tensor from memory range
layout = cute.make_layout((8, 16))
tensor = buf.data.get_tensor(layout)
```

#### struct.Align[#](#struct-align "Link to this heading")

`Align` specifies explicit alignment requirements for struct members.

**Syntax:**

```python
cute.struct.Align[dtype, alignment]
```

param dtype:
:   The type to align (scalar, MemRange, or struct)

type dtype:
:   Type

param alignment:
:   The alignment in bytes (must be > 0)

type alignment:
:   int

**Properties:**

* `dtype` - The data type being aligned
* `align` - The alignment value

**Examples:**

```python
@cute.struct
class AlignedStorage:
    # Align scalar to 16 bytes
    counter: cute.struct.Align[cutlass.Int32, 16]

    # Align array to 1024 bytes
    buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 256], 1024
    ]
```

### union[#](#union "Link to this heading")

The `union` decorator abstracts C unions in Python DSL. Similar to `struct`, but all members start at offset 0, and the size is the maximum size of all members.

**Layout Characteristics:**

* All objects start at offset 0
* Alignment is the maximum alignment of all objects
* Size is the maximum size of all objects

**Usage:**

```python
# Define a union with scalar elements
@cute.union
class value_union:
    as_int : cutlass.Int32
    as_float : cutlass.Float32

# Allocate union
val = allocator.allocate(value_union)

# Access different interpretations of same memory
val.as_int = 42
float_val = val.as_float.ptr.load()  # Interpret same bits as float
```

**Methods:**

Same as `struct`:

* `__sizeof__()` - Returns the size of the union in bytes
* `__alignof__()` - Returns the alignment of the union in bytes

## Deprecated Types[#](#deprecated-types "Link to this heading")

### ThrMma[#](#thrmma "Link to this heading")

Deprecated since version ``cute.core.ThrMma``: is deprecated, use `cute.ThrMma` instead

### ThrCopy[#](#thrcopy "Link to this heading")

Deprecated since version ``cute.core.ThrCopy``: is deprecated, use `cute.ThrCopy` instead

## Type Hierarchies and Relationships[#](#type-hierarchies-and-relationships "Link to this heading")

**Type Protocol Support:**

Many CuTe types implement standard Python protocols for integration:

* `__str__()` - String representation for debugging
* `__eq__()` / `__ne__()` - Equality comparison
* `__getitem__()` - Indexing operations
* `__add__()` / `__sub__()` / `__mul__()` / `__floordiv__()` / `__mod__()` - Arithmetic

**MLIR Integration:**

Internal types like `IntValue`, `Layout`, `Pointer`, and `ComposedLayout` are registered as MLIR value casters, enabling seamless integration with the underlying compiler infrastructure.

## Best Practices[#](#best-practices "Link to this heading")

**Choosing Between Static and Dynamic:**

* Use static values (Python `int`) when dimensions are known at compile time for maximum optimization
* Use dynamic values (`IntValue`) when dimensions must be determined at runtime
* Refer to [Static vs Dynamic layouts](dsl_dynamic_layout.html) for detailed guidance on static vs dynamic layouts

**Memory Alignment:**

* Always specify alignment requirements for shared memory structures to avoid bank conflicts
* Use `struct.Align` to enforce alignment constraints
* Check `max_alignment` properties to verify pointer and layout alignment

**Layout Operations:**

* Prefer built-in layout operations (`make_layout`, `composition`, etc.) over manual construction
* Use `ScaledBasis` for explicit control over stride modes in multi-modal layouts
* Leverage `ComposedLayout` for complex transformations like swizzling

**Type Safety:**

* Use type annotations in `@jit` and `@kernel` functions
* Let the DSL infer types when possible for cleaner code
* Check `dtype` and `memspace` properties when working with pointers

## See Also[#](#see-also "Link to this heading")

* [Introduction](dsl_introduction.html) - Introduction to CuTe DSL decorators and calling conventions
* [Control Flow](dsl_control_flow.html) - Control flow with static and dynamic values
* [Static vs Dynamic layouts](dsl_dynamic_layout.html) - Working with static and dynamic layouts
* [Integration with Frameworks](framework_integration.html) - Integration with deep learning frameworks
* [Debugging](debugging.html) - Debugging techniques for CuTe DSL programs

[previous

JIT Compilation Options](dsl_jit_compilation_options.html "previous page")
[next

Integration with Frameworks](framework_integration.html "next page")

On this page

* [Overview](#overview)
* [Core Numeric Types](#core-numeric-types)
  + [IntValue](#intvalue)
  + [Ratio](#ratio)
* [Layout Algebra Types](#layout-algebra-types)
  + [ScaledBasis](#scaledbasis)
  + [Swizzle](#swizzle)
  + [Layout](#layout)
  + [ComposedLayout](#composedlayout)
* [Memory and Pointer Types](#memory-and-pointer-types)
  + [Pointer](#pointer)
* [Structured Data Types](#structured-data-types)
  + [struct](#struct)
    - [struct.MemRange](#struct-memrange)
    - [struct.Align](#struct-align)
  + [union](#union)
* [Deprecated Types](#deprecated-types)
  + [ThrMma](#thrmma)
  + [ThrCopy](#thrcopy)
* [Type Hierarchies and Relationships](#type-hierarchies-and-relationships)
* [Best Practices](#best-practices)
* [See Also](#see-also)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.