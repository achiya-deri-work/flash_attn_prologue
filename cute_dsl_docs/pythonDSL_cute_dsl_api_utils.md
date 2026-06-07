

cutlass.utils — NVIDIA CUTLASS Documentation



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
    - [tcgen05 submodule](cute_nvgpu_tcgen05.html)
  + [pipeline](pipeline.html)
  + [utils](#)
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
* cutlass.utils

# cutlass.utils[#](#cutlass-utils "Link to this heading")

The `cutlass.utils` module contains utilities for developing kernels with CuTe DSL.

cutlass.utils.get\_smem\_capacity\_in\_bytes( : *compute\_capability: str | None = None*, ) → int[#](#cutlass.utils.get_smem_capacity_in_bytes "Link to this definition")
:   Get the shared memory capacity in bytes for a given compute capability.

    Returns the maximum shared memory capacity in bytes available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*Optional**[**str**]*) â The compute capability string (e.g. â70â, â75â, â80â)

    Returns:
    :   The shared memory capacity in bytes

    Return type:
    :   int

    Raises:
    :   **ValueError** â If the compute capability is not supported

*class* cutlass.utils.SmemAllocator[#](#cutlass.utils.SmemAllocator "Link to this definition")
:   Bases: `object`

    A helper class for managing shared memory allocation on GPU.

    This class manages shared memory and provides APIs for allocation of raw bytes,
    numeric types, arrays, and tensors with specified layouts and alignments.

    Note

    * The base pointer is aligned to 1024 bytes upon initialization.
    * SmemAllocator will automatically calculate the usage upon kernel launch.
    * There is no need to explicitly specify shared memory size in kernel launch.
    * Currently only supports static layouts. Dynamic layouts are not supported.

    **Examples**:

    ```python
    smem = SmemAllocator()

    # Allocate raw bytes
    buf_ptr = smem.allocate(100)  # 100 bytes

    # Allocate numeric type
    int8_ptr = smem.allocate(Int8)  # 1 byte

    # Define a struct
    @cute.struct
    class SharedStorage:
        alpha: cutlass.Float32
        x: cutlass.Int32

    # Allocate struct
    struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

    # use of struct members
    struct_ptr.alpha = 1.0
    struct_ptr.x = 2
    x_ptr = struct_ptr.x.ptr

    # Allocate array
    int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

    # Allocate tensor
    layout = cute.make_layout((16, 16))
    tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes
    ```

    *static* capacity\_in\_bytes( : *compute\_capability: str | None = None*, ) → int[#](#cutlass.utils.SmemAllocator.capacity_in_bytes "Link to this definition")
    :   Get the shared memory capacity in bytes for a given compute capability.

        Returns the maximum shared memory capacity in bytes available for the specified
        GPU compute capability.

        Parameters:
        :   **compute\_capability** (*Optional**[**str**]*) â The compute capability string (e.g. â70â, â75â, â80â)

        Returns:
        :   The shared memory capacity in bytes

        Return type:
        :   int

        Raises:
        :   **ValueError** â If the compute capability is not supported

    \_\_init\_\_(*\**, *loc=None*, *ip=None*)[#](#cutlass.utils.SmemAllocator.__init__ "Link to this definition")
    :   Initialize a new SmemAllocator instance.

        Creates a new shared memory allocator with a base pointer aligned to 1024 bytes.
        Tracks the allocator instance for memory management.

        Parameters:
        :   * **loc** (*Optional**[**ir.Location**]*) â Source location information for debugging, defaults to None
            * **ip** (*Optional**[**ir.InsertionPoint**]*) â Insertion point for MLIR operations, defaults to None

    allocate( : *size\_or\_type: int*, : *byte\_alignment: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.utils.SmemAllocator.allocate "Link to this definition")

    allocate( : *size\_or\_type: Type[cutlass.cutlass\_dsl.Numeric]*, : *byte\_alignment: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer

    allocate( : *size\_or\_type: [struct](cute.html#cutlass.cute.struct "cutlass.cute.core.struct")*, : *byte\_alignment: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer
    :   Allocate a block of memory with specified size and alignment.

        This method allocates a block of shared memory with the specified size and alignment requirements.
        It supports allocating raw bytes, numeric types(as scalar value), and struct types.

        Parameters:
        :   * **size\_or\_type** (*Union**[**int**,* *Type**[**Numeric**]**,* [*cute.struct*](cute.html#cutlass.cute.struct "cutlass.cute.struct")*]*) â The allocation specification, which can be:
              - An integer specifying the number of bytes to allocate
              - A Numeric type (e.g., Int8, Float32) to allocate space for one element
              - A struct type to allocate space for the entire struct
            * **byte\_alignment** (*int**,* *optional*) â The minimum byte alignment requirement for the allocation, defaults to 1
            * **loc** (*Optional**[**ir.Location**]*) â Source location information for debugging, defaults to None
            * **ip** (*Optional**[**ir.InsertionPoint**]*) â Insertion point for MLIR operations, defaults to None

        Returns:
        :   For raw bytes and numeric types, returns a pointer to the allocated memory.
            For struct types, returns an initialized struct instance at the allocated location.

        Return type:
        :   cute.Pointer

        Raises:
        :   * **ValueError** â If size is negative or alignment is less than 1
            * **TypeError** â If size\_or\_type is not an integer, Numeric type, or struct
            * **RuntimeError** â If allocation would exceed available shared memory

    allocate\_array( : *element\_type: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_elems: int = 1*, : *\**, : *byte\_alignment: int = 1*, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.SmemAllocator.allocate_array "Link to this definition")
    :   Allocate an array of elements in shared memory.

        Parameters:
        :   * **element\_type** (*Type**[**Numeric**]*) â The type of elements to allocate
            * **num\_elems** (*int**,* *optional*) â Number of elements to allocate, defaults to 1

        Returns:
        :   Pointer to the start of the allocated array

        Return type:
        :   cute.Pointer

        Raises:
        :   * **ValueError** â If num\_elems is less than 1
            * **TypeError** â If element\_type is not a Numeric type

    allocate\_tensor( : *element\_type: Type[cutlass.cutlass\_dsl.Numeric]*, : *layout: int | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *byte\_alignment: int = 1*, : *swizzle: cutlass.\_mlir.ir.register\_value\_caster | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.SmemAllocator.allocate_tensor "Link to this definition")
    :   Allocate a tensor in shared memory.

        Note: Currently only supports static layouts. Dynamic layouts are not supported.

        Parameters:
        :   * **element\_type** (*Type**[**Numeric**]*) â The type of elements in the tensor
            * **layout** (*Union**[**int**,* *cute.Layout**,* *cute.ComposedLayout**]*) â The layout specification for the tensor. Must be a static layout.
            * **byte\_alignment** (*int**,* *optional*) â The byte alignment requirement, defaults to 1
            * **swizzle** ([*cute.Swizzle*](cute.html#cutlass.cute.Swizzle "cutlass.cute.Swizzle")*,* *optional*) â Swizzle for position-dependent swizzling, defaults to None

        Returns:
        :   The allocated tensor with specified properties

        Return type:
        :   cute.Tensor

        Raises:
        :   * **TypeError** â If element\_type is not a Numeric type or if swizzle conflicts with layout
            * **ValueError** â If allocation is not byte-aligned
            * **NotImplementedError** â If dynamic layout is specified

*class* cutlass.utils.TmemAllocator( : *alloc\_result\_dst\_smem\_ptr: cutlass.cute.typing.Pointer*, : *barrier\_for\_retrieve: [NamedBarrier](pipeline.html#cutlass.pipeline.NamedBarrier "cutlass.pipeline.helpers.NamedBarrier")*, : *allocator\_warp\_id: int = 0*, : *is\_two\_cta: bool = False*, : *num\_allocated\_columns: int = 0*, : *two\_cta\_tmem\_dealloc\_mbar\_ptr: cutlass.cute.typing.Pointer | None = None*, )[#](#cutlass.utils.TmemAllocator "Link to this definition")
:   Bases: `object`

    A class for managing tensor memory allocation on GPUs.

    This class manages allocation/deallocation of tensor memory, including the mbarrier
    synchronization for two cta use case.

    Variables:
    :   * **\_alloc\_result\_dst\_smem\_ptr** â The smem pointer that holds the base address of allocated tensor memory.
        * **\_barrier\_for\_retrieve** â The barrier for retrieving tensor memory ptr.
        * **\_allocator\_warp\_id** â The warp id of the allocator warp.
        * **\_is\_two\_cta** â Whether the allocator is for two cta.
        * **\_num\_allocated\_columns** â The number of columns allocated in the tensor memory.
        * **\_two\_cta\_tmem\_dealloc\_mbar\_ptr** â The mbarrier pointer required when deallocating tensor memory for two cta.
        * **\_arch** â The architecture of the GPU.

    \_\_init\_\_( : *alloc\_result\_dst\_smem\_ptr: cutlass.cute.typing.Pointer*, : *barrier\_for\_retrieve: [NamedBarrier](pipeline.html#cutlass.pipeline.NamedBarrier "cutlass.pipeline.helpers.NamedBarrier")*, : *allocator\_warp\_id: int = 0*, : *is\_two\_cta: bool = False*, : *num\_allocated\_columns: int = 0*, : *two\_cta\_tmem\_dealloc\_mbar\_ptr: cutlass.cute.typing.Pointer | None = None*, : *\**, : *arch: str = 'sm\_100'*, : *dealloc\_mbarrier\_initialized: bool = False*, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.TmemAllocator.__init__ "Link to this definition")
    :   Initialize a TmemAllocator instance for managing tensor memory on Blackwell GPUs.

        This initializer sets up the allocatorâs state, including the shared memory (smem) pointer
        holding the base address of the allocated tensor memory, barrier synchronization for
        retrieving the tensor memory pointer, allocator warp ID, whether the allocator is being used
        for a 2-SM configuration, number of allocated columns in tensor
        memory, and the optional mbarrier pointer for deallocation in the 2-SM case.

        If is\_two\_cta is set to True, this will initialize the mbarrier pointer required for tensor
        memory deallocation across two CTAs.

        Parameters:
        :   * **alloc\_result\_dst\_smem\_ptr** (*cute.Pointer*) â The shared memory pointer that holds the base address of allocated tensor memory.
            * **barrier\_for\_retrieve** ([*pipeline.NamedBarrier*](pipeline.html#cutlass.pipeline.NamedBarrier "cutlass.pipeline.NamedBarrier")) â The named barrier for retrieving the tensor memory pointer.
            * **allocator\_warp\_id** (*int**,* *optional*) â The warp ID of the allocator warp, defaults to 0.
            * **is\_two\_cta** (*bool**,* *optional*) â Whether the allocator should coordinate two CTAs, defaults to False.
            * **num\_allocated\_columns** (*int**,* *optional*) â The number of columns allocated in tensor memory, defaults to 0.
            * **two\_cta\_tmem\_dealloc\_mbar\_ptr** (*cute.Pointer**,* *optional*) â The mbarrier pointer required for two-CTA tensor memory deallocation, optional.
            * **loc** (*Any**,* *optional*) â Optional codegen location for debugging and error reporting.
            * **ip** (*Any**,* *optional*) â Optional insertion point for codegen.

        Raises:
        :   **AssertionError** â If two\_cta\_tmem\_dealloc\_mbar\_ptr is None while is\_two\_cta is True.

    check\_valid\_num\_columns(*num\_columns: int*)[#](#cutlass.utils.TmemAllocator.check_valid_num_columns "Link to this definition")
    :   Check if the number of columns is valid.

        This method checks if the number of columns is valid.
        It checks if the number of columns is larger than 0, smaller than max capacity, a multiple of 32, and a power of two.

    wait\_for\_alloc(*\**, *loc=None*, *ip=None*)[#](#cutlass.utils.TmemAllocator.wait_for_alloc "Link to this definition")
    :   Wait for the allocator warp to finish allocation.

        This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.

    retrieve\_ptr( : *dtype: Type[cutlass.cutlass\_dsl.Numeric] = cutlass.cutlass\_dsl.Float32*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.utils.TmemAllocator.retrieve_ptr "Link to this definition")
    :   Retrieve the pointer to the allocated tensor memory.

        This method can be called by all warps after allocation has been performed
        by the allocator warp.

cutlass.utils.get\_num\_tmem\_alloc\_cols( : *tmem\_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor]*, : *rounding=True*, : *\**, : *arch: str = 'sm\_100'*, : *loc=None*, : *ip=None*, ) → int[#](#cutlass.utils.get_num_tmem_alloc_cols "Link to this definition")
:   Get the total number of TMEM allocation columns for the given TMEM tensors.

    Parameters:
    :   * **tmem\_tensors** (*Union**[**cute.Tensor**,* *List**[**cute.Tensor**]**]*) â The TMEM tensors to get the number of allocation columns for.
        * **rounding** (*bool*) â Whether to round up the number of allocation columns to the nearest power of 2.
        * **arch** (*str*) â The architecture of the GPU.

    Returns:
    :   The total number of TMEM allocation columns.

    Return type:
    :   int

    Raises:
    :   **ValueError** â If the number of TMEM allocation columns exceeds the maximum capacity or is less than 32.

*class* cutlass.utils.LayoutEnum(*value*)[#](#cutlass.utils.LayoutEnum "Link to this definition")
:   Bases: `Enum`

    An enumeration.

    ROW\_MAJOR *= 'row\_major'*[#](#cutlass.utils.LayoutEnum.ROW_MAJOR "Link to this definition")

    COL\_MAJOR *= 'col\_major'*[#](#cutlass.utils.LayoutEnum.COL_MAJOR "Link to this definition")

    mma\_major\_mode()[#](#cutlass.utils.LayoutEnum.mma_major_mode "Link to this definition")

    sm90\_mma\_major\_mode()[#](#cutlass.utils.LayoutEnum.sm90_mma_major_mode "Link to this definition")

    is\_k\_major\_a()[#](#cutlass.utils.LayoutEnum.is_k_major_a "Link to this definition")

    is\_m\_major\_a()[#](#cutlass.utils.LayoutEnum.is_m_major_a "Link to this definition")

    is\_n\_major\_b()[#](#cutlass.utils.LayoutEnum.is_n_major_b "Link to this definition")

    is\_k\_major\_b()[#](#cutlass.utils.LayoutEnum.is_k_major_b "Link to this definition")

    is\_n\_major\_c()[#](#cutlass.utils.LayoutEnum.is_n_major_c "Link to this definition")

    is\_m\_major\_c()[#](#cutlass.utils.LayoutEnum.is_m_major_c "Link to this definition")

    *static* from\_tensor( : *tensor: cutlass.cute.typing.Tensor*, ) → [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")[#](#cutlass.utils.LayoutEnum.from_tensor "Link to this definition")

*class* cutlass.utils.WorkTileInfo( : *tile\_idx: cutlass.cute.typing.Coord*, : *is\_valid\_tile: cutlass.cutlass\_dsl.Boolean*, )[#](#cutlass.utils.WorkTileInfo "Link to this definition")
:   Bases: `object`

    A class to represent information about a work tile.

    Variables:
    :   * **tile\_idx** â The index of the tile.
        * **is\_valid\_tile** â Whether the tile is valid.

    \_\_init\_\_( : *tile\_idx: cutlass.cute.typing.Coord*, : *is\_valid\_tile: cutlass.cutlass\_dsl.Boolean*, )[#](#cutlass.utils.WorkTileInfo.__init__ "Link to this definition")

    *property* is\_valid\_tile*: cutlass.cutlass\_dsl.Boolean*[#](#cutlass.utils.WorkTileInfo.is_valid_tile "Link to this definition")
    :   Check latest tile returned by the scheduler is valid or not. Any scheduling
        requests after all tasks completed will return an invalid tile.

        Returns:
        :   The validity of the tile.

        Return type:
        :   Boolean

    *property* tile\_idx*: cutlass.cute.typing.Coord*[#](#cutlass.utils.WorkTileInfo.tile_idx "Link to this definition")
    :   Get the index of the tile.

        Returns:
        :   The index of the tile.

        Return type:
        :   cute.Coord

*class* cutlass.utils.PersistentTileSchedulerParams[#](#cutlass.utils.PersistentTileSchedulerParams "Link to this definition")
:   Bases: `object`

    A class to represent parameters for a persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    Variables:
    :   * **cluster\_shape\_mn** â Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
        * **problem\_layout\_ncluster\_mnl** â Layout of the problem in terms of
          number of clusters in (m, n, l) dimensions.

    \_\_init\_\_( : *problem\_shape\_ntile\_mnl: cutlass.cute.typing.Shape*, : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *swizzle\_size: int = 1*, : *raster\_along\_m: bool = True*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.PersistentTileSchedulerParams.__init__ "Link to this definition")
    :   Initializes the PersistentTileSchedulerParams with the given parameters.

        Parameters:
        :   * **problem\_shape\_ntile\_mnl** (*cute.Shape*) â The shape of the problem in terms of
              number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
            * **cluster\_shape\_mnk** (*cute.Shape*) â The shape of the cluster in (m, n) dimensions.
            * **swizzle\_size** (*int*) â Swizzling size in the unit of cluster. 1 means no swizzle
            * **raster\_along\_m** (*bool*) â Rasterization order of clusters. Only used when swizzle\_size > 1.
              True means along M, false means along N.

        Raises:
        :   **ValueError** â If cluster\_shape\_k is not 1.

    get\_grid\_shape( : *max\_active\_clusters: cutlass.cutlass\_dsl.Int32*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer][#](#cutlass.utils.PersistentTileSchedulerParams.get_grid_shape "Link to this definition")
    :   Computes the grid shape based on the maximum active clusters allowed.

        Parameters:
        :   **max\_active\_clusters** (*Int32*) â The maximum number of active clusters that
            can run in one wave.

        Returns:
        :   A tuple containing the grid shape in (m, n, persistent\_clusters).
            - m: self.cluster\_shape\_m.
            - n: self.cluster\_shape\_n.
            - persistent\_clusters: Number of persistent clusters that can run.

*class* cutlass.utils.StaticPersistentTileScheduler( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: cutlass.cutlass\_dsl.Int32*, : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, )[#](#cutlass.utils.StaticPersistentTileScheduler "Link to this definition")
:   Bases: `object`

    A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

    Variables:
    :   * **params** â Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
        * **num\_persistent\_clusters** â Number of persistent clusters that can be launched
        * **cta\_id\_in\_cluster** â ID of the CTA within its cluster
        * **\_num\_tiles\_executed** â Counter for executed tiles
        * **\_current\_work\_linear\_idx** â Current cluster index

    \_\_init\_\_( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: cutlass.cutlass\_dsl.Int32*, : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, )[#](#cutlass.utils.StaticPersistentTileScheduler.__init__ "Link to this definition")
    :   Initializes the StaticPersistentTileScheduler with the given parameters.

        Parameters:
        :   * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
            * **num\_persistent\_clusters** (*Int32*) â Number of persistent clusters that can be launched.
            * **current\_work\_linear\_idx** (*Int32*) â Current cluster index.
            * **cta\_id\_in\_cluster** (*cute.Coord*) â ID of the CTA within its cluster.
            * **num\_tiles\_executed** (*Int32*) â Counter for executed tiles.

    *static* create( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *block\_idx: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *grid\_dim: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.StaticPersistentTileScheduler.create "Link to this definition")
    :   Initialize the static persistent tile scheduler.

        Parameters:
        :   * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Parameters for the persistent
              tile scheduler.
            * **block\_idx** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d block index in the format (bidx, bidy, bidz).
            * **grid\_dim** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d grid dimensions for kernel launch.

        Returns:
        :   A StaticPersistentTileScheduler object.

        Return type:
        :   [StaticPersistentTileScheduler](#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.StaticPersistentTileScheduler")

    *static* get\_grid\_shape( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *max\_active\_clusters: cutlass.cutlass\_dsl.Int32*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer][#](#cutlass.utils.StaticPersistentTileScheduler.get_grid_shape "Link to this definition")
    :   Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        Parameters:
        :   * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Parameters for grid shape calculation.
            * **max\_active\_clusters** (*Int32*) â Maximum active clusters allowed.

        Returns:
        :   The calculated 3d grid shape.

        Return type:
        :   Tuple[Integer, Integer, Integer]

    \_get\_current\_work\_for\_linear\_idx( : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.StaticPersistentTileScheduler._get_current_work_for_linear_idx "Link to this definition")
    :   Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

        Parameters:
        :   **current\_work\_linear\_idx** (*Int32*) â The linear index of the current work.

        Returns:
        :   An object containing information about the current tile coordinates
            and validity status.

        Return type:
        :   [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

    \_get\_cluster\_work\_idx\_with\_fastdivmod( : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cutlass\_dsl.Int32, cutlass.cutlass\_dsl.Int32, cutlass.cutlass\_dsl.Int32][#](#cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod "Link to this definition")
    :   FastDivmod optimized CLUSTER coordinate calculation.

        CRITICAL: This should mimic problem\_layout\_ncluster\_mnl.get\_hier\_coord()
        which returns CLUSTER coordinates, not tile coordinates!

        Parameters:
        :   **current\_work\_linear\_idx** (*Int32*) â Linear index in the work space

        Returns:
        :   Cluster coordinates (m, n, l) or None if FastDivmod not available

        Return type:
        :   Tuple[Int32, Int32, Int32] or None

    get\_current\_work( : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.StaticPersistentTileScheduler.get_current_work "Link to this definition")

    initial\_work\_tile\_info( : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.StaticPersistentTileScheduler.initial_work_tile_info "Link to this definition")

    advance\_to\_next\_work( : *\**, : *advance\_count: int = 1*, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.StaticPersistentTileScheduler.advance_to_next_work "Link to this definition")

    *property* num\_tiles\_executed*: cutlass.cutlass\_dsl.Int32*[#](#cutlass.utils.StaticPersistentTileScheduler.num_tiles_executed "Link to this definition")

*class* cutlass.utils.StaticPersistentRuntimeTileScheduler( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: cutlass.cutlass\_dsl.Int32*, : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, : *inner\_mode: int = 1*, )[#](#cutlass.utils.StaticPersistentRuntimeTileScheduler "Link to this definition")
:   Bases: [`StaticPersistentTileScheduler`](#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler")

    A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
    This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

    Variables:
    :   * **params** â Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
        * **num\_persistent\_clusters** â Number of persistent clusters that can be launched
        * **cta\_id\_in\_cluster** â ID of the CTA within its cluster
        * **\_num\_tiles\_executed** â Counter for executed tiles
        * **\_current\_work\_linear\_idx** â Current cluster index

    \_\_init\_\_( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: cutlass.cutlass\_dsl.Int32*, : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, : *inner\_mode: int = 1*, )[#](#cutlass.utils.StaticPersistentRuntimeTileScheduler.__init__ "Link to this definition")
    :   Initializes the StaticPersistentTileScheduler with the given parameters.

        Parameters:
        :   * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
            * **num\_persistent\_clusters** (*Int32*) â Number of persistent clusters that can be launched.
            * **current\_work\_linear\_idx** (*Int32*) â Current cluster index.
            * **cta\_id\_in\_cluster** (*cute.Coord*) â ID of the CTA within its cluster.
            * **num\_tiles\_executed** (*Int32*) â Counter for executed tiles.
            * **inner\_mode** (*int*) â The inner mode along which the linear index will be decomposed first.

    *static* create( : *params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *block\_idx: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *grid\_dim: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *inner\_mode: int = 1*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.StaticPersistentRuntimeTileScheduler.create "Link to this definition")
    :   Initialize the static persistent tile scheduler.

        Parameters:
        :   * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Parameters for the persistent
              tile scheduler.
            * **block\_idx** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d block index in the format (bidx, bidy, bidz).
            * **grid\_dim** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d grid dimensions for kernel launch.
            * **inner\_mode** (*int*) â The inner mode along which the linear index will be decomposed first.

        Returns:
        :   A StaticPersistentRuntimeTileScheduler object.

        Return type:
        :   [StaticPersistentRuntimeTileScheduler](#cutlass.utils.StaticPersistentRuntimeTileScheduler "cutlass.utils.StaticPersistentRuntimeTileScheduler")

    \_get\_current\_work\_for\_linear\_idx( : *current\_work\_linear\_idx: cutlass.cutlass\_dsl.Int32*, : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.StaticPersistentRuntimeTileScheduler._get_current_work_for_linear_idx "Link to this definition")
    :   Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

        Parameters:
        :   **current\_work\_linear\_idx** (*Int32*) â The linear index of the current work.

        Returns:
        :   An object containing information about the current tile coordinates
            and validity status.

        Return type:
        :   [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

*class* cutlass.utils.TensorMapUpdateMode(*value*)[#](#cutlass.utils.TensorMapUpdateMode "Link to this definition")
:   Bases: `Enum`

    Enum class defining tensor map update modes.

    Modes:
    GMEM: Update tensormap in global memory
    SMEM: Load tensormap from global memory to shared memory,
    update it in shared memory, then store back to global memory

    GMEM *= 1*[#](#cutlass.utils.TensorMapUpdateMode.GMEM "Link to this definition")

    SMEM *= 2*[#](#cutlass.utils.TensorMapUpdateMode.SMEM "Link to this definition")

*class* cutlass.utils.TensorMapManager( : *tensormap\_update\_mode: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*, : *bytes\_per\_tensormap: int*, )[#](#cutlass.utils.TensorMapManager "Link to this definition")
:   Bases: `object`

    Manages TensorMap operations including initialization and updates.
    Provides utilities to convert tensormap pointer to across different memory spaces.

    tensormap\_update\_mode*: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*[#](#cutlass.utils.TensorMapManager.tensormap_update_mode "Link to this definition")

    bytes\_per\_tensormap*: int*[#](#cutlass.utils.TensorMapManager.bytes_per_tensormap "Link to this definition")

    get\_tensormap\_ptr( : *ptr: cutlass.cute.typing.Pointer*, : *address\_space=cutlass.\_mlir.dialects.cute.AddressSpace.gmem*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.utils.TensorMapManager.get_tensormap_ptr "Link to this definition")

    fence\_tensormap\_initialization( : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.utils.TensorMapManager.fence_tensormap_initialization "Link to this definition")

    fence\_tensormap\_update( : *tensormap\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.utils.TensorMapManager.fence_tensormap_update "Link to this definition")

    \_\_init\_\_( : *tensormap\_update\_mode: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*, : *bytes\_per\_tensormap: int*, ) → None[#](#cutlass.utils.TensorMapManager.__init__ "Link to this definition")

*class* cutlass.utils.GroupSearchResult( : *group\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_idx\_m: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_idx\_n: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_m: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_n: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_k: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_count\_k: cutlass.cutlass\_dsl.Int32*, )[#](#cutlass.utils.GroupSearchResult "Link to this definition")
:   Bases: `object`

    The result of the group search for grouped gemm.

    Parameters:
    :   * **group\_idx** (*Int32*) â The result group index
        * **cta\_tile\_idx\_m** (*Int32*) â CTA tile index along M dimension after rasterization
        * **cta\_tile\_idx\_n** (*Int32*) â CTA tile index along N dimension after rasterization
        * **problem\_shape\_m** (*Int32*) â The M dimension of the gemm problem
        * **problem\_shape\_n** (*Int32*) â The N dimension of the gemm problem
        * **problem\_shape\_k** (*Int32*) â The K dimension of the gemm problem
        * **cta\_tile\_count\_k** (*Int32*) â Number of tiles along K dimension

    \_\_init\_\_( : *group\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_idx\_m: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_idx\_n: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_m: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_n: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_k: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_count\_k: cutlass.cutlass\_dsl.Int32*, ) → None[#](#cutlass.utils.GroupSearchResult.__init__ "Link to this definition")

*class* cutlass.utils.GroupedGemmGroupSearchState( : *start\_group\_idx: cutlass.cutlass\_dsl.Int32*, : *tile\_count\_prev\_group: cutlass.cutlass\_dsl.Int32*, : *tile\_count\_searched: cutlass.cutlass\_dsl.Int32*, : *found: cutlass.cutlass\_dsl.Boolean*, )[#](#cutlass.utils.GroupedGemmGroupSearchState "Link to this definition")
:   Bases: `object`

    The state of group index search for grouped gemm.

    The state will be initialized once and updated in every round of group index search.

    Parameters:
    :   * **start\_group\_idx** (*Int32*) â The group idx to start the search with
        * **tile\_count\_prev\_group** (*Int32*) â Number of tiles before the matched group
        * **tile\_count\_searched** (*Int32*) â Number of tiles we have searched. When the matched group is found,
          it records the number of tiles including the matched group

    \_\_init\_\_( : *start\_group\_idx: cutlass.cutlass\_dsl.Int32*, : *tile\_count\_prev\_group: cutlass.cutlass\_dsl.Int32*, : *tile\_count\_searched: cutlass.cutlass\_dsl.Int32*, : *found: cutlass.cutlass\_dsl.Boolean*, ) → None[#](#cutlass.utils.GroupedGemmGroupSearchState.__init__ "Link to this definition")

cutlass.utils.create\_initial\_search\_state() → [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")[#](#cutlass.utils.create_initial_search_state "Link to this definition")
:   Create an initial search state for grouped gemm.

    Returns:
    :   A new search state with initial values

    Return type:
    :   [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

*class* cutlass.utils.GroupedGemmTileSchedulerHelper(*\*\*kwargs*)[#](#cutlass.utils.GroupedGemmTileSchedulerHelper "Link to this definition")
:   Bases: `object`

    A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA tile index for grouped gemm.

    Parameters:
    :   * **group\_count** (*int*) â Number of groups in current grouped gemm problem
        * **tile\_sched\_params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) â Parameter used to create the tile scheduler this helper works with
        * **cluster\_tile\_shape\_mnk** (*tuple**[**int**,* *int**,* *int**]*) â The shape of cluster tile as (m, n, k)
        * **search\_state** ([*GroupedGemmGroupSearchState*](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")) â The initial search state

    \_\_init\_\_( : *group\_count: int*, : *tile\_sched\_params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *cluster\_tile\_shape\_mnk: tuple[int, int, int]*, : *search\_state: [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")*, ) → None[#](#cutlass.utils.GroupedGemmTileSchedulerHelper.__init__ "Link to this definition")

    delinearize\_z( : *cta\_tile\_coord: tuple*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, ) → [GroupSearchResult](#cutlass.utils.GroupSearchResult "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupSearchResult")[#](#cutlass.utils.GroupedGemmTileSchedulerHelper.delinearize_z "Link to this definition")
    :   Delinearize the linear z index and return GroupSearchResult.

        This function should be used by warps that need to know the CTA tile index on M and N dimensions.

        Parameters:
        :   * **cta\_tile\_coord** (*tuple* *of* *Int32*) â The raw CTA coordinate from tile scheduler
            * **problem\_shape\_mnkl** (*cute.Tensor*) â Tensor containing gemm problem size (M, N, K, L) for each group

        Returns:
        :   The search result containing group index and tile coordinates

        Return type:
        :   [GroupSearchResult](#cutlass.utils.GroupSearchResult "cutlass.utils.GroupSearchResult")

    search\_cluster\_tile\_count\_k( : *cta\_tile\_coord: tuple*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, ) → Tuple[cutlass.cutlass\_dsl.Int32, cutlass.cutlass\_dsl.Int32][#](#cutlass.utils.GroupedGemmTileSchedulerHelper.search_cluster_tile_count_k "Link to this definition")
    :   Search the matched group for given linear index and compute the number of tiles along K dimension for the matched group.

        This function should be used by warps that are only interested in the number of tiles along K dimension.

        Parameters:
        :   * **cta\_tile\_coord** (*tuple* *of* *Int32*) â The raw CTA coordinate from tile scheduler
            * **problem\_shape\_mnkl** (*cute.Tensor*) â Tensor containing gemm problem size (M, N, K, L) for all groups

        Returns:
        :   A tuple containing cluster count along K dimension and the group index

        Return type:
        :   Tuple[Int32, Int32]

    \_prefix\_sum( : *value\_per\_thread: cutlass.cutlass\_dsl.Int32*, ) → cutlass.cutlass\_dsl.Int32[#](#cutlass.utils.GroupedGemmTileSchedulerHelper._prefix_sum "Link to this definition")
    :   Perform prefix sum within a full warp.

        Parameters:
        :   **value\_per\_thread** (*Int32*) â The value for this thread to contribute to the prefix sum

        Returns:
        :   The prefix sum result for this thread

        Return type:
        :   Int32

    \_get\_problem\_for\_group( : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *group\_idx: cutlass.cutlass\_dsl.Int32*, ) → cutlass.cute.typing.Tensor[#](#cutlass.utils.GroupedGemmTileSchedulerHelper._get_problem_for_group "Link to this definition")
    :   Load gemm problem (m,n,k,l) for the specified group from global memory to register.

        Parameters:
        :   * **problem\_shape\_mnkl** (*cute.Tensor*) â Tensor in global memory with layout (group\_count, 4):(4, 1)
            * **group\_idx** (*Int32*) â The index of the group to load

        Returns:
        :   The problem shape tensor for the specified group

        Return type:
        :   cute.Tensor

    \_get\_cluster\_tile\_count\_mn( : *problem\_shape: cutlass.cute.typing.Tensor*, ) → cutlass.cutlass\_dsl.Int32[#](#cutlass.utils.GroupedGemmTileSchedulerHelper._get_cluster_tile_count_mn "Link to this definition")
    :   Compute total cluster count.

        Parameters:
        :   **problem\_shape** (*cute.Tensor*) â Tensor containing problem shape (m, n, k, l)

        Returns:
        :   The total cluster tile count for M and N dimensions

        Return type:
        :   Int32

    \_compute\_cta\_tile\_coord( : *cluster\_tile\_idx: cutlass.cutlass\_dsl.Int32*, : *cta\_tile\_coord\_in\_cluster: tuple*, : *cluster\_tile\_count\_m: cutlass.cutlass\_dsl.Int32*, : *cluster\_tile\_count\_n: cutlass.cutlass\_dsl.Int32*, ) → tuple[#](#cutlass.utils.GroupedGemmTileSchedulerHelper._compute_cta_tile_coord "Link to this definition")
    :   Compute CTA tile indices along M and N dimensions based on the linear index within a group.

        It uses the AlongM mode to decompose the linear index onto M and N dimensions.

        Parameters:
        :   * **cluster\_tile\_idx** (*Int32*) â The linear index within a group
            * **cta\_tile\_coord\_in\_cluster** (*tuple* *of* *Int32*) â CTA indices along M and N dimensions within a cluster
            * **cluster\_tile\_count\_m** (*Int32*) â The number of clusters along M dimension of the matched group
            * **cluster\_tile\_count\_n** (*Int32*) â The number of clusters along N dimension of the matched group

        Returns:
        :   A tuple containing CTA tile indices along M and N dimensions

        Return type:
        :   tuple of (Int32, Int32)

    \_group\_search( : *linear\_idx: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *init\_group\_idx: cutlass.cutlass\_dsl.Int32*, : *init\_tile\_count\_searched: cutlass.cutlass\_dsl.Int32*, ) → [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")[#](#cutlass.utils.GroupedGemmTileSchedulerHelper._group_search "Link to this definition")
    :   Search which group the linear index belongs to.

        Parameters:
        :   * **linear\_idx** (*Int32*) â The linear index to be decomposed
            * **problem\_shape\_mnkl** (*cute.Tensor*) â Tensor containing gemm problem size (M, N, K, L) for all groups
            * **init\_group\_idx** (*Int32*) â The group idx to start the search with
            * **init\_tile\_count\_searched** (*Int32*) â The number of tiles we have searched

        Returns:
        :   The updated search state

        Return type:
        :   [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

    \_group\_search\_and\_load\_problem\_shape( : *linear\_idx: cutlass.cutlass\_dsl.Int32*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *start\_group\_idx: cutlass.cutlass\_dsl.Int32*, : *tile\_count\_searched: cutlass.cutlass\_dsl.Int32*, ) → Tuple[cutlass.cutlass\_dsl.Int32, cutlass.cute.typing.Tensor][#](#cutlass.utils.GroupedGemmTileSchedulerHelper._group_search_and_load_problem_shape "Link to this definition")
    :   Perform group search and load problem shape for the matched group.

        Parameters:
        :   * **linear\_idx** (*Int32*) â The linear index to be decomposed
            * **problem\_shape\_mnkl** (*cute.Tensor*) â Tensor containing gemm problem size (M, N, K, L) for all groups
            * **start\_group\_idx** (*Int32*) â The group idx to start the search with
            * **tile\_count\_searched** (*Int32*) â The number of tiles we have searched

        Returns:
        :   A tuple containing the final group index and the problem shape tensor

        Return type:
        :   Tuple[Int32, cute.Tensor]

*class* cutlass.utils.HardwareInfo(*device\_id: int = 0*)[#](#cutlass.utils.HardwareInfo "Link to this definition")
:   Bases: `object`

    device\_id: CUDA device ID to get the hardware info.

    \_\_init\_\_(*device\_id: int = 0*)[#](#cutlass.utils.HardwareInfo.__init__ "Link to this definition")

    get\_max\_active\_clusters( : *cluster\_size: int*, : *stream: cuda.bindings.driver.CUstream | None = None*, ) → int[#](#cutlass.utils.HardwareInfo.get_max_active_clusters "Link to this definition")
    :   Get the maximum number of active clusters for a given cluster size.

        When a stream from a green context is provided, the occupancy calculation
        will reflect the reduced SM partition of the green context.

        Parameters:
        :   * **cluster\_size** (*int*) â Number of blocks per cluster (must be between 1 and 32)
            * **stream** (*driver.CUstream**,* *optional*) â Optional CUDA stream handle. If provided (especially from a green context),
              the occupancy calculation reflects the streamâs SM partition.

        Returns:
        :   Maximum number of active clusters

        Return type:
        :   int

    get\_l2\_cache\_size\_in\_bytes() → int[#](#cutlass.utils.HardwareInfo.get_l2_cache_size_in_bytes "Link to this definition")

    get\_device\_multiprocessor\_count() → int[#](#cutlass.utils.HardwareInfo.get_device_multiprocessor_count "Link to this definition")

    \_checkCudaErrors(*result*) → None[#](#cutlass.utils.HardwareInfo._checkCudaErrors "Link to this definition")

    \_cudaGetErrorEnum(*error*) → str[#](#cutlass.utils.HardwareInfo._cudaGetErrorEnum "Link to this definition")

    \_cuda\_driver\_version\_ge(*major: int*, *minor: int*) → bool[#](#cutlass.utils.HardwareInfo._cuda_driver_version_ge "Link to this definition")

    \_cuda\_driver\_version\_lt(*major: int*, *minor: int*) → bool[#](#cutlass.utils.HardwareInfo._cuda_driver_version_lt "Link to this definition")

    \_empty\_kernel()[#](#cutlass.utils.HardwareInfo._empty_kernel "Link to this definition")

    \_host\_function()[#](#cutlass.utils.HardwareInfo._host_function "Link to this definition")

    \_get\_device\_function() → cuda.bindings.driver.CUfunction[#](#cutlass.utils.HardwareInfo._get_device_function "Link to this definition")
    :   Get a device function by compiling a dummy kernel using cuteDSL pipeline.

*class* cutlass.utils.TransformMode(*value*)[#](#cutlass.utils.TransformMode "Link to this definition")
:   Bases: `Enum`

    An enumeration for the possible transform modes of a mixed-input GEMM.

    ConvertOnly *= 1*[#](#cutlass.utils.TransformMode.ConvertOnly "Link to this definition")

    ConvertScale *= 2*[#](#cutlass.utils.TransformMode.ConvertScale "Link to this definition")

cutlass.utils.scale\_tma\_partition( : *tCsS: cutlass.cute.typing.Tensor*, : *tCgS: cutlass.cute.typing.Tensor*, : *tma\_atom\_s: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *block\_in\_cluster\_coord\_vmnk: cutlass.cute.typing.Coord*, : *scale\_cta\_layout: cutlass.cute.typing.Layout*, ) → tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor][#](#cutlass.utils.scale_tma_partition "Link to this definition")
:   Perform TMA partition for scale tensor.
    This method partitions the global memory and shared memory buffer for the scale tensor for TMA load.
    :param tCsS: Input scale shared memory tensor
    :type tCsS: cute.Tensor
    :param tCgS: Input scale global memory tensor
    :type tCgS: cute.Tensor
    :param tma\_atom\_s: TMA copy atom for scale tensor
    :type tma\_atom\_s: cute.CopyAtom
    :param block\_in\_cluster\_coord\_vmnk: CTA coord in the cluster
    :type block\_in\_cluster\_coord\_vmnk: cute.Coord
    :param scale\_cta\_layout: Layout of CTA from the view of the scale tensor
    :type scale\_cta\_layout: cute.Layout
    :return: A tuple containing (tSsS, tSgS) where:

    > * tSsS: Partitioned scale tensor in shared memory
    > * tSgS: Partitioned scale tensor in global memory

    Return type:
    :   tuple[cute.Tensor, cute.Tensor]

cutlass.utils.transform\_partition( : *transform\_a\_source: [tcgen05.OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")*, : *scale\_mode: [TransformMode](#cutlass.utils.TransformMode "cutlass.utils.TransformMode")*, : *copy\_atom\_a\_input: [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")*, : *copy\_atom\_a\_transform: [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")*, : *sA\_input: cute.Tensor*, : *A\_transform: cute.Tensor*, : *transform\_local\_tidx: cutlass.Int32*, ) → tuple[[cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy") | None, [cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy") | None, cute.Tensor, cute.Tensor][#](#cutlass.utils.transform_partition "Link to this definition")
:   Partition tensors for transform input and output.
    This method sets up the copy atoms and partitions the shared/tensor memory
    for the transformation of tensor A.
    :param transform\_a\_source: Where the transformed tensor A is stored (TMEM or SMEM)
    :type transform\_a\_source: tcgen05.OperandSource
    :param scale\_mode: The transform mode (ConvertOnly or ConvertScale)
    :type scale\_mode: TransformMode
    :param copy\_atom\_a\_input: Copy atom for loading A from shared memory
    :type copy\_atom\_a\_input: cute.CopyAtom
    :param copy\_atom\_a\_transform: Copy atom for storing transformed A
    :type copy\_atom\_a\_transform: cute.CopyAtom
    :param sA\_input: Input tensor A in shared memory
    :type sA\_input: cute.Tensor
    :param A\_transform: Transformed tensor A in tensor or shared memory
    :type A\_transform: cute.Tensor
    :param transform\_local\_tidx: Local thread index for transformation warps
    :type transform\_local\_tidx: cutlass.Int32
    :return: A tuple containing (src\_copy\_a, dst\_copy\_a, tAsA\_input, tA\_transform) where:

    > * src\_copy\_a: Tiled copy for source tensor
    > * dst\_copy\_a: Tiled copy for destination tensor
    > * tAsA\_input: Partitioned input tensor A
    > * tA\_transform: Partitioned transformed tensor A

    Return type:
    :   tuple[Optional[[cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], Optional[[cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], cute.Tensor, cute.Tensor]

cutlass.utils.scale\_partition( : *src\_copy\_a: [cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")*, : *tCsS: cute.Tensor*, : *transform\_local\_tidx: cutlass.Int32*, : *mma\_dtype: type[cutlass.Numeric]*, ) → tuple[[cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy"), cute.Tensor, cute.Tensor, cute.Tensor][#](#cutlass.utils.scale_partition "Link to this definition")
:   Partition the scale tensor for transformation.
    This method prepares the copy atom and partitions the shared memory for the scale tensor.
    :param src\_copy\_a: Tiled copy for the source tensor
    :type src\_copy\_a: cute.TiledCopy
    :param tCsS: Scale tensor in shared memory
    :type tCsS: cute.Tensor
    :param transform\_local\_tidx: Local thread index for transformation warps
    :type transform\_local\_tidx: cutlass.Int32
    :param mma\_dtype: Data type for the MMA operation
    :type mma\_dtype: type[cutlass.Numeric]
    :return: A tuple containing (smem\_thr\_copy\_S, tSsS\_trans, tSrS\_copy, tSrS) where:

    > * smem\_thr\_copy\_S: Tiled copy for the scale tensor
    > * tSsS\_trans: Partitioned scale tensor for transformation
    > * tSrS\_copy: Register fragment for the scale tensor
    > * tSrS: View of scale tensor used for transformation computation

    Return type:
    :   tuple[[cute.TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy"), cute.Tensor, cute.Tensor, cute.Tensor]

cutlass.utils.get\_gmem\_layout\_scale( : *scale\_shape\_mkl: tuple[int, int, int]*, : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *scale\_major\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → cutlass.cute.typing.Layout[#](#cutlass.utils.get_gmem_layout_scale "Link to this definition")
:   Get the layout of the scale tensor in global memory.
    :param scale\_shape\_mkl: The shape of the scale tensor (M, K, L).
    :type scale\_shape\_mkl: tuple[int, int, int]
    :return: The layout of the scale tensor in global memory.
    :rtype: cute.Layout

cutlass.utils.get\_smem\_layout\_scale( : *mma\_tiler: tuple[int, int, int]*, : *use\_2cta\_instrs: bool*, : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *scale\_major\_mode: [tcgen05.OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")*, : *a\_scale\_dtype: type[cutlass.Numeric]*, : *num\_scale\_load2trans\_stage: int*, ) → tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout][#](#cutlass.utils.get_smem_layout_scale "Link to this definition")
:   Get the layout of the scale tensor in shared memory.
    :return: A tuple containing (scale\_tile\_shape, smem\_layout\_scale\_per\_stage, smem\_layout\_scale) where:

    > * scale\_tile\_shape: The tile shape
    > * smem\_layout\_scale\_per\_stage: Shared memory layout for scale tensor per stage
    > * smem\_layout\_scale: Shared memory layout for scale tensor

    Return type:
    :   tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout]

cutlass.utils.compute\_smem\_layout( : *tiled\_mma: [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")*, : *mma\_tiler\_mnk: tuple[int, int, int]*, : *a\_dtype: type[cutlass.Numeric]*, : *b\_dtype: type[cutlass.Numeric]*, : *load2trans\_stage\_count: int*, : *trans2mma\_stage\_count: int*, ) → tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout][#](#cutlass.utils.compute_smem_layout "Link to this definition")
:   Compute shared memory layouts for tensor A, transformed A and tensor B.
    :param tiled\_mma: The tiled MMA object defining the core computation.
    :type tiled\_mma: cute.TiledMma
    :param mma\_tiler\_mnk: The shape (M, N, K) of the MMA tiler.
    :type mma\_tiler\_mnk: tuple[int, int, int]
    :param a\_dtype: Data type of operand A.
    :type a\_dtype: type[cutlass.Numeric]
    :param b\_dtype: Data type of operand B.
    :type b\_dtype: type[cutlass.Numeric]
    :param load2trans\_stage\_count: Number of stages for load-to-transform pipeline.
    :type load2trans\_stage\_count: int
    :param trans2mma\_stage\_count: Number of stages for transform-to-MMA pipeline.
    :type trans2mma\_stage\_count: int
    :return: A tuple containing (smem\_layout\_a, smem\_layout\_a\_transform, smem\_layout\_b) where:

    > * smem\_layout\_a: Shared memory layout for tensor A
    > * smem\_layout\_a\_transform: Shared memory layout for transformed tensor A
    > * smem\_layout\_b: Shared memory layout for tensor B

    Return type:
    :   tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]

cutlass.utils.get\_transform\_a\_source( : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")[#](#cutlass.utils.get_transform_a_source "Link to this definition")
:   Determine the operand source for transformed A tensor based on the operand major mode.

cutlass.utils.get\_tma\_atom\_kind( : *mcast: cutlass.Boolean*, : *use\_2cta\_instrs: bool*, : *is\_b: bool*, ) → [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") | [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")[#](#cutlass.utils.get_tma_atom_kind "Link to this definition")
:   Get the TMA atom kind based on 1) whether itâs a multicast operation,
    2) whether 2CTA tcgen05.mma instruction is enabled, and
    3) whether itâs a B tensor

cutlass.utils.get\_copy\_atom\_a\_transform( : *mma\_dtype: type[cutlass.Numeric]*, : *use\_2cta\_instrs: bool*, : *transform\_a\_source: [tcgen05.OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")*, : *a\_smem\_shape: cute.Shape*, : *a\_dtype: type[cutlass.Numeric]*, ) → [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")[#](#cutlass.utils.get_copy_atom_a_transform "Link to this definition")
:   Determine the copy atom for transformed A tensor based on the operand source and tile size.

cutlass.utils.is\_valid\_scale\_granularity( : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *a\_dtype: type[cutlass.Numeric]*, : *k: int*, : *mma\_tiler\_k: int*, ) → bool[#](#cutlass.utils.is_valid_scale_granularity "Link to this definition")
:   Check if the scale granularity settings are valid for the given data type and problem size.

cutlass.utils.get\_divisibility( : *contiguous\_dim\_size: int*, : *upper\_bound: int = 128*, ) → int[#](#cutlass.utils.get_divisibility "Link to this definition")
:   Calculate the largest power of 2 divisibility factor for memory alignment.

cutlass.utils.compute\_epilogue\_tile\_shape( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *use\_2cta\_instrs: bool*, : *layout\_d: [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *\**, : *layout\_c: [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum") | None = None*, : *elem\_ty\_c: Type[cutlass.cutlass\_dsl.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tile[#](#cutlass.utils.compute_epilogue_tile_shape "Link to this definition")
:   Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) â A tuple or list representing the dimensions of the CTA tile, where
          cta\_tile\_shape[0] corresponds to the height (M) and cta\_tile\_shape[1]
          corresponds to the width (N) of the tile.
        * **use\_2cta\_instrs** (*bool*) â A flag indicating whether the configuration is for a 2SM setup.
        * **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type of output tensor D.
        * **layout\_c** ([*LayoutEnum*](#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")*,* *optional*) â The layout enum of the input tensor C. Defaults to None.
        * **elem\_ty\_c** (*Union**[**Type**[**Numeric**]**,* *None**]**,* *optional*) â The element type for input tensor C. Defaults to None.

    Returns:
    :   Returns epilog tiler, which is used in subsequent epilog partitions.

    Return type:
    :   cute.Tile

    Raises:
    :   **ValueError** â If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

cutlass.utils.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *tiled\_tmem\_load: [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.utils.get_smem_store_op "Link to this definition")
:   Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM\_LOADâs thread-value ownership.

    Parameters:
    :   * **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) â The element type for accumulator.
        * **tiled\_tmem\_load** ([*cute.TiledCopy*](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) â An instance of TiledCopy that represents the tmem load operation.

    Returns:
    :   Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

cutlass.utils.get\_tmem\_load\_op( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *layout\_d: [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_tile: cutlass.cute.typing.Tile*, : *use\_2cta\_instrs: bool*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")[#](#cutlass.utils.get_tmem_load_op "Link to this definition")
:   Finds a performant TMEM\_LOAD copy op for the selected epilogue
    tile (epi\_tile), element types, and tcgen05.mma instruction used.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) â A tuple or list representing the dimensions of the CTA tile.
        * **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) â The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) â The element type for accumulation.
        * **epi\_tile** (*cute.Tile*) â The epilogue tile configuration.
        * **use\_2cta\_instrs** (*bool*) â A flag indicating whether the configuration is for 2 SMs.

    Returns:
    :   An instance of Sm100TmemLoad with the computed configuration.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

    Raises:
    :   **ValueError** â If the function cannot handle the given combination of accumulation
        and dimension types, or if it cannot determine the appropriate configuration based on
        the input parameters.

cutlass.utils.make\_smem\_layout\_a( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.make_smem_layout_a "Link to this definition")
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â The tiled MMA used to partition tensor A
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape
        * **a\_dtype** (*Type**[**Numeric**]*) â The element type for tensor A
        * **num\_stages** (*int*) â The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_smem\_layout\_b( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.make_smem_layout_b "Link to this definition")
:   This function helps:

    1. Get the partitioned shape of the B tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensorâs majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) â The tiled MMA which is used to partition the B tensor.
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) â The MMA tile shape.
        * **b\_dtype** (*Type**[**Numeric**]*) â The element type for the B tensor.
        * **num\_stages** (*int*) â The stage of the B tensor.

    Returns:
    :   SMEM layout for the B tensor.

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_smem\_layout\_epi( : *epi\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_layout: [LayoutEnum](#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout[#](#cutlass.utils.make_smem_layout_epi "Link to this definition")
:   This function helps:

    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensorâs majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **epi\_dtype** (*Type**[**Numeric**]*) â The element type for the epilog tensor.
        * **epi\_layout** ([*LayoutEnum*](#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) â The layout enum for the epilog tensor.
        * **epi\_tile** (*cute.cute.Tile*) â The epilogue tile shape.
        * **epi\_stage** (*int*) â The stage of the epilog tensor.

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *acc\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")[#](#cutlass.utils.make_trivial_tiled_mma "Link to this definition")
:   Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) â Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand B (1 for K, 0 for M/N).
        * **acc\_dtype** (*type**[**Numeric**]*) â Data type of the accumulator.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) â The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) â The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) â The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** â If the data type is not supported.

cutlass.utils.make\_blockscaled\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *sf\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *sf\_vec\_size: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")[#](#cutlass.utils.make_blockscaled_trivial_tiled_mma "Link to this definition")
:   Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) â Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) â Leading dimension of operand B (1 for K, 0 for M/N).
        * **sf\_dtype** (*type**[**Numeric**]*) â Data type of the Scale Factor.
        * **sf\_vec\_size** (*int*) â The vector size of the Scale Factor.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) â The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) â The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) â The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** â If the data type is not supported.

*class* cutlass.utils.ClcDynamicPersistentTileSchedulerParams( : *problem\_shape\_ntile\_mnl: cutlass.cute.typing.Shape*, : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "Link to this definition")
:   Bases: `object`

    A class to represent parameters for a dynamic persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    Variables:
    :   **cluster\_shape\_mn** â Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).

    \_\_init\_\_( : *problem\_shape\_ntile\_mnl: cutlass.cute.typing.Shape*, : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams.__init__ "Link to this definition")
    :   Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

        Parameters:
        :   * **problem\_shape\_ntile\_mnl** (*cute.Shape*) â The shape of the problem in terms of
              number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
            * **cluster\_shape\_mnk** (*cute.Shape*) â The shape of the cluster in (m, n) dimensions.

        Raises:
        :   **ValueError** â If cluster\_shape\_k is not 1.

    get\_grid\_shape( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer][#](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams.get_grid_shape "Link to this definition")
    :   Computes the grid shape based on the problem shape and cluster shape.

        Returns:
        :   the grid is the CTA numbers that has aligned with cluster shape.

*class* cutlass.utils.ClcDynamicPersistentTileScheduler( : *params: [ClcDynamicPersistentTileSchedulerParams](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *block\_idx: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, )[#](#cutlass.utils.ClcDynamicPersistentTileScheduler "Link to this definition")
:   Bases: `object`

    A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

    Variables:
    :   * **params** â Tile schedule related params, including cluster shape.
        * **cta\_id\_in\_cluster** â ID of the CTA within its cluster
        * **\_num\_tiles\_executed** â Counter for executed tiles

    \_\_init\_\_( : *params: [ClcDynamicPersistentTileSchedulerParams](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: cutlass.cutlass\_dsl.Int32*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *block\_idx: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, )[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.__init__ "Link to this definition")
    :   Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

        Parameters:
        :   * **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) â Tile schedule related params, including cluster shape.
            * **cta\_id\_in\_cluster** (*cute.Coord*) â ID of the CTA within its cluster.
            * **num\_tiles\_executed** (*Int32*) â Counter for executed tiles.
            * **clc\_response\_ptr** (*cute.Pointer*) â Pointer of the clc rsponse.
            * **block\_idx** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The block index.

    create( : *params: [ClcDynamicPersistentTileSchedulerParams](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *block\_idx: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *grid\_dim: Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer]*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.create "Link to this definition")
    :   Initialize the dynamic persistent tile scheduler.

        Parameters:
        :   * **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) â Parameters for the persistent
              tile scheduler.
            * **block\_idx** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d block index in the format (bidx, bidy, bidz).
            * **grid\_dim** (*Tuple**[**Integer**,* *Integer**,* *Integer**]*) â The 3d grid dimensions for kernel launch.

        Returns:
        :   A ClcDynamicPersistentTileScheduler object.

        Return type:
        :   [ClcDynamicPersistentTileScheduler](#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.ClcDynamicPersistentTileScheduler")

    get\_grid\_shape( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer, cutlass.cutlass\_dsl.Integer][#](#cutlass.utils.ClcDynamicPersistentTileScheduler.get_grid_shape "Link to this definition")
    :   Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        Parameters:
        :   **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) â Parameters for grid shape calculation.

        Returns:
        :   The calculated 3d grid shape.

        Return type:
        :   Tuple[Integer, Integer, Integer]

    work\_tile\_info\_from\_clc\_response( : *result\_addr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response "Link to this definition")
    :   Simulates parsing CLC response data in Python.
        result\_addr: 16-byte response data (simulating shared memory access)

    get\_current\_work( : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.get_current_work "Link to this definition")

    initial\_work\_tile\_info( : *\**, : *loc=None*, : *ip=None*, ) → [WorkTileInfo](#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.initial_work_tile_info "Link to this definition")

    advance\_to\_next\_work( : *mbarrier\_addr*, : *loc=None*, : *ip=None*, )[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.advance_to_next_work "Link to this definition")

    *property* num\_tiles\_executed*: cutlass.cutlass\_dsl.Int32*[#](#cutlass.utils.ClcDynamicPersistentTileScheduler.num_tiles_executed "Link to this definition")

cutlass.utils.print\_latex( : *x: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *\**, : *color: ~typing.Callable = <function tikz\_color\_bwx8>*, )[#](#cutlass.utils.print_latex "Link to this definition")
:   Prints a layout.
    :param x: A layout
    :type x: Union[Layout, ComposedLayout]
    :param color: A function that returns TiKZ colors
    :type color: Callable

cutlass.utils.print\_latex\_tv( : *layout\_tv: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *tile\_mn: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Layout*, : *\**, : *color: ~typing.Callable = <function tikz\_color\_tv>*, )[#](#cutlass.utils.print_latex_tv "Link to this definition")
:   Prints a tv layout for a tile M N. Everything must be static.
    :param layout\_tv: A static thread value layout
    :type layout\_tv: Union[Layout, ComposedLayout]
    :param tile\_mn: A static M N tile
    :type tile\_mn: Union[IntTuple, Layout]
    :param color: A function that returns TiKZ colors
    :type color: Callable

cutlass.utils.is\_fp8\_dtype(*dtype: Type[cutlass.cute.typing.Numeric]*) → bool[#](#cutlass.utils.is_fp8_dtype "Link to this definition")
:   Check if dtype is a float8 type that doesnât support dlpack.
    params dtype: The cutlass numeric type to check
    type dtype: Type[cutlass.Numeric]
    return: True if the dtype is Float8E5M2 or Float8E4M3FN, False otherwise

cutlass.utils.create\_cute\_tensor\_for\_fp8( : *storage\_tensor*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *leading\_dim: int*, : *source\_f32\_tensor=None*, )[#](#cutlass.utils.create_cute_tensor_for_fp8 "Link to this definition")
:   Create cute tensor, handling float8 types that donât support dlpack.

    For float8 types, the storage\_tensor should be uint8 (for DLPack compatibility).
    The source\_f32\_tensor provides the actual float32 values to convert to fp8.

    params storage\_tensor: Tensor for DLPack (uint8 for fp8, otherwise the actual dtype)
    params dtype: Target cutlass dtype
    params leading\_dim: Leading dimension for dynamic layout
    paramas source\_f32\_tensor: Float32 source data for fp8 conversion (required for fp8)
    return: A cute tensor with the appropriate dtype and layout

[previous

cutlass.pipeline](pipeline.html "previous page")
[next

Utilities for SM90](utils_sm90.html "next page")

On this page

* [`get_smem_capacity_in_bytes()`](#cutlass.utils.get_smem_capacity_in_bytes)
* [`SmemAllocator`](#cutlass.utils.SmemAllocator)
  + [`capacity_in_bytes()`](#cutlass.utils.SmemAllocator.capacity_in_bytes)
  + [`__init__()`](#cutlass.utils.SmemAllocator.__init__)
  + [`allocate()`](#cutlass.utils.SmemAllocator.allocate)
  + [`allocate_array()`](#cutlass.utils.SmemAllocator.allocate_array)
  + [`allocate_tensor()`](#cutlass.utils.SmemAllocator.allocate_tensor)
* [`TmemAllocator`](#cutlass.utils.TmemAllocator)
  + [`__init__()`](#cutlass.utils.TmemAllocator.__init__)
  + [`check_valid_num_columns()`](#cutlass.utils.TmemAllocator.check_valid_num_columns)
  + [`wait_for_alloc()`](#cutlass.utils.TmemAllocator.wait_for_alloc)
  + [`retrieve_ptr()`](#cutlass.utils.TmemAllocator.retrieve_ptr)
* [`get_num_tmem_alloc_cols()`](#cutlass.utils.get_num_tmem_alloc_cols)
* [`LayoutEnum`](#cutlass.utils.LayoutEnum)
  + [`ROW_MAJOR`](#cutlass.utils.LayoutEnum.ROW_MAJOR)
  + [`COL_MAJOR`](#cutlass.utils.LayoutEnum.COL_MAJOR)
  + [`mma_major_mode()`](#cutlass.utils.LayoutEnum.mma_major_mode)
  + [`sm90_mma_major_mode()`](#cutlass.utils.LayoutEnum.sm90_mma_major_mode)
  + [`is_k_major_a()`](#cutlass.utils.LayoutEnum.is_k_major_a)
  + [`is_m_major_a()`](#cutlass.utils.LayoutEnum.is_m_major_a)
  + [`is_n_major_b()`](#cutlass.utils.LayoutEnum.is_n_major_b)
  + [`is_k_major_b()`](#cutlass.utils.LayoutEnum.is_k_major_b)
  + [`is_n_major_c()`](#cutlass.utils.LayoutEnum.is_n_major_c)
  + [`is_m_major_c()`](#cutlass.utils.LayoutEnum.is_m_major_c)
  + [`from_tensor()`](#cutlass.utils.LayoutEnum.from_tensor)
* [`WorkTileInfo`](#cutlass.utils.WorkTileInfo)
  + [`__init__()`](#cutlass.utils.WorkTileInfo.__init__)
  + [`is_valid_tile`](#cutlass.utils.WorkTileInfo.is_valid_tile)
  + [`tile_idx`](#cutlass.utils.WorkTileInfo.tile_idx)
* [`PersistentTileSchedulerParams`](#cutlass.utils.PersistentTileSchedulerParams)
  + [`__init__()`](#cutlass.utils.PersistentTileSchedulerParams.__init__)
  + [`get_grid_shape()`](#cutlass.utils.PersistentTileSchedulerParams.get_grid_shape)
* [`StaticPersistentTileScheduler`](#cutlass.utils.StaticPersistentTileScheduler)
  + [`__init__()`](#cutlass.utils.StaticPersistentTileScheduler.__init__)
  + [`create()`](#cutlass.utils.StaticPersistentTileScheduler.create)
  + [`get_grid_shape()`](#cutlass.utils.StaticPersistentTileScheduler.get_grid_shape)
  + [`_get_current_work_for_linear_idx()`](#cutlass.utils.StaticPersistentTileScheduler._get_current_work_for_linear_idx)
  + [`_get_cluster_work_idx_with_fastdivmod()`](#cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod)
  + [`get_current_work()`](#cutlass.utils.StaticPersistentTileScheduler.get_current_work)
  + [`initial_work_tile_info()`](#cutlass.utils.StaticPersistentTileScheduler.initial_work_tile_info)
  + [`advance_to_next_work()`](#cutlass.utils.StaticPersistentTileScheduler.advance_to_next_work)
  + [`num_tiles_executed`](#cutlass.utils.StaticPersistentTileScheduler.num_tiles_executed)
* [`StaticPersistentRuntimeTileScheduler`](#cutlass.utils.StaticPersistentRuntimeTileScheduler)
  + [`__init__()`](#cutlass.utils.StaticPersistentRuntimeTileScheduler.__init__)
  + [`create()`](#cutlass.utils.StaticPersistentRuntimeTileScheduler.create)
  + [`_get_current_work_for_linear_idx()`](#cutlass.utils.StaticPersistentRuntimeTileScheduler._get_current_work_for_linear_idx)
* [`TensorMapUpdateMode`](#cutlass.utils.TensorMapUpdateMode)
  + [`GMEM`](#cutlass.utils.TensorMapUpdateMode.GMEM)
  + [`SMEM`](#cutlass.utils.TensorMapUpdateMode.SMEM)
* [`TensorMapManager`](#cutlass.utils.TensorMapManager)
  + [`tensormap_update_mode`](#cutlass.utils.TensorMapManager.tensormap_update_mode)
  + [`bytes_per_tensormap`](#cutlass.utils.TensorMapManager.bytes_per_tensormap)
  + [`get_tensormap_ptr()`](#cutlass.utils.TensorMapManager.get_tensormap_ptr)
  + [`fence_tensormap_initialization()`](#cutlass.utils.TensorMapManager.fence_tensormap_initialization)
  + [`fence_tensormap_update()`](#cutlass.utils.TensorMapManager.fence_tensormap_update)
  + [`__init__()`](#cutlass.utils.TensorMapManager.__init__)
* [`GroupSearchResult`](#cutlass.utils.GroupSearchResult)
  + [`__init__()`](#cutlass.utils.GroupSearchResult.__init__)
* [`GroupedGemmGroupSearchState`](#cutlass.utils.GroupedGemmGroupSearchState)
  + [`__init__()`](#cutlass.utils.GroupedGemmGroupSearchState.__init__)
* [`create_initial_search_state()`](#cutlass.utils.create_initial_search_state)
* [`GroupedGemmTileSchedulerHelper`](#cutlass.utils.GroupedGemmTileSchedulerHelper)
  + [`__init__()`](#cutlass.utils.GroupedGemmTileSchedulerHelper.__init__)
  + [`delinearize_z()`](#cutlass.utils.GroupedGemmTileSchedulerHelper.delinearize_z)
  + [`search_cluster_tile_count_k()`](#cutlass.utils.GroupedGemmTileSchedulerHelper.search_cluster_tile_count_k)
  + [`_prefix_sum()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._prefix_sum)
  + [`_get_problem_for_group()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._get_problem_for_group)
  + [`_get_cluster_tile_count_mn()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._get_cluster_tile_count_mn)
  + [`_compute_cta_tile_coord()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._compute_cta_tile_coord)
  + [`_group_search()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._group_search)
  + [`_group_search_and_load_problem_shape()`](#cutlass.utils.GroupedGemmTileSchedulerHelper._group_search_and_load_problem_shape)
* [`HardwareInfo`](#cutlass.utils.HardwareInfo)
  + [`__init__()`](#cutlass.utils.HardwareInfo.__init__)
  + [`get_max_active_clusters()`](#cutlass.utils.HardwareInfo.get_max_active_clusters)
  + [`get_l2_cache_size_in_bytes()`](#cutlass.utils.HardwareInfo.get_l2_cache_size_in_bytes)
  + [`get_device_multiprocessor_count()`](#cutlass.utils.HardwareInfo.get_device_multiprocessor_count)
  + [`_checkCudaErrors()`](#cutlass.utils.HardwareInfo._checkCudaErrors)
  + [`_cudaGetErrorEnum()`](#cutlass.utils.HardwareInfo._cudaGetErrorEnum)
  + [`_cuda_driver_version_ge()`](#cutlass.utils.HardwareInfo._cuda_driver_version_ge)
  + [`_cuda_driver_version_lt()`](#cutlass.utils.HardwareInfo._cuda_driver_version_lt)
  + [`_empty_kernel()`](#cutlass.utils.HardwareInfo._empty_kernel)
  + [`_host_function()`](#cutlass.utils.HardwareInfo._host_function)
  + [`_get_device_function()`](#cutlass.utils.HardwareInfo._get_device_function)
* [`TransformMode`](#cutlass.utils.TransformMode)
  + [`ConvertOnly`](#cutlass.utils.TransformMode.ConvertOnly)
  + [`ConvertScale`](#cutlass.utils.TransformMode.ConvertScale)
* [`scale_tma_partition()`](#cutlass.utils.scale_tma_partition)
* [`transform_partition()`](#cutlass.utils.transform_partition)
* [`scale_partition()`](#cutlass.utils.scale_partition)
* [`get_gmem_layout_scale()`](#cutlass.utils.get_gmem_layout_scale)
* [`get_smem_layout_scale()`](#cutlass.utils.get_smem_layout_scale)
* [`compute_smem_layout()`](#cutlass.utils.compute_smem_layout)
* [`get_transform_a_source()`](#cutlass.utils.get_transform_a_source)
* [`get_tma_atom_kind()`](#cutlass.utils.get_tma_atom_kind)
* [`get_copy_atom_a_transform()`](#cutlass.utils.get_copy_atom_a_transform)
* [`is_valid_scale_granularity()`](#cutlass.utils.is_valid_scale_granularity)
* [`get_divisibility()`](#cutlass.utils.get_divisibility)
* [`compute_epilogue_tile_shape()`](#cutlass.utils.compute_epilogue_tile_shape)
* [`get_smem_store_op()`](#cutlass.utils.get_smem_store_op)
* [`get_tmem_load_op()`](#cutlass.utils.get_tmem_load_op)
* [`make_smem_layout_a()`](#cutlass.utils.make_smem_layout_a)
* [`make_smem_layout_b()`](#cutlass.utils.make_smem_layout_b)
* [`make_smem_layout_epi()`](#cutlass.utils.make_smem_layout_epi)
* [`make_trivial_tiled_mma()`](#cutlass.utils.make_trivial_tiled_mma)
* [`make_blockscaled_trivial_tiled_mma()`](#cutlass.utils.make_blockscaled_trivial_tiled_mma)
* [`ClcDynamicPersistentTileSchedulerParams`](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)
  + [`__init__()`](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams.__init__)
  + [`get_grid_shape()`](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams.get_grid_shape)
* [`ClcDynamicPersistentTileScheduler`](#cutlass.utils.ClcDynamicPersistentTileScheduler)
  + [`__init__()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.__init__)
  + [`create()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.create)
  + [`get_grid_shape()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.get_grid_shape)
  + [`work_tile_info_from_clc_response()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response)
  + [`get_current_work()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.get_current_work)
  + [`initial_work_tile_info()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.initial_work_tile_info)
  + [`advance_to_next_work()`](#cutlass.utils.ClcDynamicPersistentTileScheduler.advance_to_next_work)
  + [`num_tiles_executed`](#cutlass.utils.ClcDynamicPersistentTileScheduler.num_tiles_executed)
* [`print_latex()`](#cutlass.utils.print_latex)
* [`print_latex_tv()`](#cutlass.utils.print_latex_tv)
* [`is_fp8_dtype()`](#cutlass.utils.is_fp8_dtype)
* [`create_cute_tensor_for_fp8()`](#cutlass.utils.create_cute_tensor_for_fp8)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.