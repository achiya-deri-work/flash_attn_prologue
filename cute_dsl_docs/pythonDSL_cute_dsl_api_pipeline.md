

cutlass.pipeline — NVIDIA CUTLASS Documentation



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
  + [pipeline](#)
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
* cutlass.pipeline

# cutlass.pipeline[#](#module-cutlass.pipeline "Link to this heading")

*class* cutlass.pipeline.Agent(*value*)[#](#cutlass.pipeline.Agent "Link to this definition")
:   Bases: `Enum`

    Agent indicates what is participating in the pipeline synchronization.

    Thread *= 1*[#](#cutlass.pipeline.Agent.Thread "Link to this definition")

    ThreadBlock *= 2*[#](#cutlass.pipeline.Agent.ThreadBlock "Link to this definition")

    ThreadBlockCluster *= 3*[#](#cutlass.pipeline.Agent.ThreadBlockCluster "Link to this definition")

*class* cutlass.pipeline.CooperativeGroup( : *agent: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int = 1*, : *alignment=None*, )[#](#cutlass.pipeline.CooperativeGroup "Link to this definition")
:   Bases: `object`

    CooperativeGroup contains size and alignment restrictions for an Agent.

    \_\_init\_\_( : *agent: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int = 1*, : *alignment=None*, )[#](#cutlass.pipeline.CooperativeGroup.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineOp(*value*)[#](#cutlass.pipeline.PipelineOp "Link to this definition")
:   Bases: `Enum`

    PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

    AsyncThread *= 1*[#](#cutlass.pipeline.PipelineOp.AsyncThread "Link to this definition")

    TCGen05Mma *= 2*[#](#cutlass.pipeline.PipelineOp.TCGen05Mma "Link to this definition")

    TmaLoad *= 3*[#](#cutlass.pipeline.PipelineOp.TmaLoad "Link to this definition")

    ClcLoad *= 4*[#](#cutlass.pipeline.PipelineOp.ClcLoad "Link to this definition")

    TmaStore *= 5*[#](#cutlass.pipeline.PipelineOp.TmaStore "Link to this definition")

    Composite *= 6*[#](#cutlass.pipeline.PipelineOp.Composite "Link to this definition")

    AsyncLoad *= 7*[#](#cutlass.pipeline.PipelineOp.AsyncLoad "Link to this definition")

*class* cutlass.pipeline.SyncObject[#](#cutlass.pipeline.SyncObject "Link to this definition")
:   Bases: `ABC`

    Abstract base class for hardware synchronization primitives.

    This class defines the interface for different types of hardware synchronization
    mechanisms including shared memory barriers, named barriers, and fences.

    *abstract* arrive() → None[#](#cutlass.pipeline.SyncObject.arrive "Link to this definition")

    *abstract* wait() → None[#](#cutlass.pipeline.SyncObject.wait "Link to this definition")

    *abstract* arrive\_and\_wait() → None[#](#cutlass.pipeline.SyncObject.arrive_and_wait "Link to this definition")

    *abstract* arrive\_and\_drop() → None[#](#cutlass.pipeline.SyncObject.arrive_and_drop "Link to this definition")

    *abstract* get\_barrier() → cutlass.cute.typing.Pointer | int | None[#](#cutlass.pipeline.SyncObject.get_barrier "Link to this definition")

    *abstract* max() → int | None[#](#cutlass.pipeline.SyncObject.max "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.pipeline.SyncObject._abc_impl "Link to this definition")

*class* cutlass.pipeline.MbarrierArray[#](#cutlass.pipeline.MbarrierArray "Link to this definition")
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    MbarrierArray implements an abstraction for an array of smem barriers.

    \_\_init\_\_( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.__init__ "Link to this definition")

    recast\_to\_new\_op\_type( : *new\_op\_type: [PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")*, ) → [MbarrierArray](#cutlass.pipeline.MbarrierArray "cutlass.pipeline.helpers.MbarrierArray")[#](#cutlass.pipeline.MbarrierArray.recast_to_new_op_type "Link to this definition")
    :   Creates a copy of MbarrierArray with a different op\_type without re-initializing barriers

    mbarrier\_init(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.MbarrierArray.mbarrier_init "Link to this definition")
    :   Initializes an array of mbarriers using warp 0.

    arrive( : *index: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive "Link to this definition")
    :   Select the arrive corresponding to this MbarrierArrayâs PipelineOp.

        Parameters:
        :   * **index** (*int*) â Index of the mbarrier in the array to arrive on
            * **dst** (*int* *|* *None*) â Destination parameter for selective arrival, which can be either a mask or destination cta rank.
              When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
              - For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
              in the cluster with rank = 0, 1, and 3).
              - For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
              the mbarrier with rank = 3 in the cluster).
            * **cta\_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) â CTA group for `TCGen05Mma`, defaults to None for other op types

    arrive\_mbarrier( : *index: int*, : *dst\_rank: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive_mbarrier "Link to this definition")

    arrive\_cp\_async\_mbarrier( : *index: int*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.MbarrierArray.arrive_cp_async_mbarrier "Link to this definition")

    arrive\_tcgen05mma( : *index: int*, : *mask: int | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive_tcgen05mma "Link to this definition")

    arrive\_and\_expect\_tx( : *index: int*, : *tx\_count: int*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive_and_expect_tx "Link to this definition")

    arrive\_and\_expect\_tx\_with\_dst( : *index: int*, : *tx\_count: int*, : *dst: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive_and_expect_tx_with_dst "Link to this definition")

    try\_wait( : *index: int*, : *phase: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean[#](#cutlass.pipeline.MbarrierArray.try_wait "Link to this definition")

    wait( : *index: int*, : *phase: int*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.wait "Link to this definition")

    arrive\_and\_wait( : *index: int*, : *phase: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None[#](#cutlass.pipeline.MbarrierArray.arrive_and_wait "Link to this definition")

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.MbarrierArray.arrive_and_drop "Link to this definition")

    get\_barrier( : *index: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.pipeline.MbarrierArray.get_barrier "Link to this definition")

    max() → int[#](#cutlass.pipeline.MbarrierArray.max "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.pipeline.MbarrierArray._abc_impl "Link to this definition")

*class* cutlass.pipeline.NamedBarrier(*barrier\_id: int*, *num\_threads: int*)[#](#cutlass.pipeline.NamedBarrier "Link to this definition")
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    NamedBarrier is an abstraction for named barriers managed by hardware.
    There are 16 named barriers available, with barrier\_ids 0-15.

    See the [PTX documentation](https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

    barrier\_id*: int*[#](#cutlass.pipeline.NamedBarrier.barrier_id "Link to this definition")

    num\_threads*: int*[#](#cutlass.pipeline.NamedBarrier.num_threads "Link to this definition")

    arrive(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.arrive "Link to this definition")
    :   The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.

    arrive\_unaligned(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.arrive_unaligned "Link to this definition")
    :   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

    wait(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.wait "Link to this definition")
    :   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
        If synchronizing two warps in a producer/consumer pairing, the arrive count would be
        32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
        or consumer are counted for mbarriers, while all threads participating in the sync
        are counted for NamedBarriers.

    wait\_unaligned(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.wait_unaligned "Link to this definition")

    arrive\_and\_wait(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.arrive_and_wait "Link to this definition")

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.arrive_and_drop "Link to this definition")

    sync(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.NamedBarrier.sync "Link to this definition")

    get\_barrier(*\**, *loc=None*, *ip=None*) → int[#](#cutlass.pipeline.NamedBarrier.get_barrier "Link to this definition")

    max() → int[#](#cutlass.pipeline.NamedBarrier.max "Link to this definition")

    \_\_init\_\_(*barrier\_id: int*, *num\_threads: int*) → None[#](#cutlass.pipeline.NamedBarrier.__init__ "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.pipeline.NamedBarrier._abc_impl "Link to this definition")

*class* cutlass.pipeline.PipelineOrder( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )[#](#cutlass.pipeline.PipelineOrder "Link to this definition")
:   Bases: `object`

    PipelineOrder is used for managing ordered pipeline execution with multiple groups.

    This class implements a pipeline ordering mechanism where work is divided into groups
    and stages, allowing for controlled progression through pipeline stages with proper
    synchronization between different groups.

    The pipeline ordering works as follows:
    - The pipeline is divided into âlengthâ number of groups
    - Each group has âdepthâ number of stages
    - Groups execute in a specific order with synchronization barriers
    - Each group waits for the previous group to complete before proceeding

    **Example:**

    ```python
    # Create pipeline order with 3 groups, each with 2 stages
    pipeline_order = PipelineOrder.create(
        barrier_storage=smem_ptr,      # shared memory pointer for barriers
        depth=2,                       # 2 stages per group
        length=3,                      # 3 groups total
        group_id=0,                    # current group ID (0, 1, or 2)
        producer_group=producer_warp   # cooperative group for producers
    )

    # In the pipeline loop
    for stage in range(num_stages):
        pipeline_order.wait()          # Wait for previous group to complete
        # Process current stage
        pipeline_order.arrive()        # Signal completion to next group
    ```

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineOrder.sync_object_full "Link to this definition")

    depth*: int*[#](#cutlass.pipeline.PipelineOrder.depth "Link to this definition")

    length*: int*[#](#cutlass.pipeline.PipelineOrder.length "Link to this definition")

    group\_id*: int*[#](#cutlass.pipeline.PipelineOrder.group_id "Link to this definition")

    state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*[#](#cutlass.pipeline.PipelineOrder.state "Link to this definition")

    *static* create( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *depth: int*, : *length: int*, : *group\_id: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineOrder.create "Link to this definition")

    get\_barrier\_for\_current\_stage\_idx(*group\_id*)[#](#cutlass.pipeline.PipelineOrder.get_barrier_for_current_stage_idx "Link to this definition")

    arrive(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineOrder.arrive "Link to this definition")

    wait(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineOrder.wait "Link to this definition")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None[#](#cutlass.pipeline.PipelineOrder.__init__ "Link to this definition")

*class* cutlass.pipeline.TmaStoreFence(*num\_stages: int = 0*)[#](#cutlass.pipeline.TmaStoreFence "Link to this definition")
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    TmaStoreFence is used for a multi-stage epilogue buffer.

    \_\_init\_\_(*num\_stages: int = 0*) → None[#](#cutlass.pipeline.TmaStoreFence.__init__ "Link to this definition")

    arrive(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.arrive "Link to this definition")

    wait(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.wait "Link to this definition")

    arrive\_and\_wait(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.arrive_and_wait "Link to this definition")

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.arrive_and_drop "Link to this definition")

    get\_barrier(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.get_barrier "Link to this definition")

    max() → None[#](#cutlass.pipeline.TmaStoreFence.max "Link to this definition")

    tail(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.TmaStoreFence.tail "Link to this definition")

    \_abc\_impl *= <\_abc.\_abc\_data object>*[#](#cutlass.pipeline.TmaStoreFence._abc_impl "Link to this definition")

*class* cutlass.pipeline.PipelineUserType(*value*)[#](#cutlass.pipeline.PipelineUserType "Link to this definition")
:   Bases: `Enum`

    An enumeration.

    Producer *= 1*[#](#cutlass.pipeline.PipelineUserType.Producer "Link to this definition")

    Consumer *= 2*[#](#cutlass.pipeline.PipelineUserType.Consumer "Link to this definition")

    ProducerConsumer *= 3*[#](#cutlass.pipeline.PipelineUserType.ProducerConsumer "Link to this definition")

*class* cutlass.pipeline.PipelineState(*stages: int*, *count*, *index*, *phase*)[#](#cutlass.pipeline.PipelineState "Link to this definition")
:   Bases: `object`

    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

    \_\_init\_\_(*stages: int*, *count*, *index*, *phase*)[#](#cutlass.pipeline.PipelineState.__init__ "Link to this definition")

    clone() → [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")[#](#cutlass.pipeline.PipelineState.clone "Link to this definition")

    *property* index*: cutlass.cutlass\_dsl.Int32*[#](#cutlass.pipeline.PipelineState.index "Link to this definition")

    *property* count*: cutlass.cutlass\_dsl.Int32*[#](#cutlass.pipeline.PipelineState.count "Link to this definition")

    *property* stages*: int*[#](#cutlass.pipeline.PipelineState.stages "Link to this definition")

    *property* phase*: cutlass.cutlass\_dsl.Int32*[#](#cutlass.pipeline.PipelineState.phase "Link to this definition")

    reset\_count(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineState.reset_count "Link to this definition")

    advance(*\**, *loc=None*, *ip=None*) → None[#](#cutlass.pipeline.PipelineState.advance "Link to this definition")

    reverse(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineState.reverse "Link to this definition")

*class* cutlass.pipeline.PipelineAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )[#](#cutlass.pipeline.PipelineAsync "Link to this definition")
:   Bases: `object`

    PipelineAsync is a generic pipeline class where both the producer and consumer are
    AsyncThreads. It also serves as a base class for specialized pipeline classes.

    This class implements a producer-consumer pipeline pattern where both sides operate
    asynchronously. The pipeline maintains synchronization state using barrier objects
    to coordinate between producer and consumer threads.

    The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

    Table 2 Pipeline State Transitions[#](#id1 "Link to this table")

    | Barrier | State | p.acquire | p.commit | c.wait | c.release |
    | --- | --- | --- | --- | --- | --- |
    | empty\_bar | empty | <Return> | n/a | n/a |  |
    | empty\_bar | wait | <Block> | n/a | n/a | -> empty |
    | full\_bar | wait | n/a | -> full | <Block > | n/a |
    | full\_bar | full | n/a |  | <Return> | n/a |

    Where:

    * p: producer
    * c: consumer
    * <Block>: This action is blocked until transition to a state allow it to proceed by other side
      - e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

    ```python
    Array of mbarriers as circular buffer:

         Advance Direction
       <-------------------

        Producer   Consumer
            |         ^
            V         |
       +-----------------+
     --|X|X|W|D|D|D|D|R|X|<-.
    /  +-----------------+   \
    |                        |
    `------------------------'
    ```

    Where:

    * X: Empty buffer (initial state)
    * W: Producer writing (producer is waiting for buffer to be empty)
    * D: Data ready (producer has written data to buffer)
    * R: Consumer reading (consumer is consuming data from buffer)

    **Example:**

    ```python
    # Create pipeline with 5 stages
    pipeline = PipelineAsync.create(
        num_stages=5,                   # number of pipeline stages
        producer_group=producer_warp,
        consumer_group=consumer_warp
        barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
    )

    producer, consumer = pipeline.make_participants()
    # Producer side
    for i in range(num_iterations):
        handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
        # Write data to pipeline buffer
        handle.commit()   # Signal buffer is full

    # Consumer side
    for i in range(num_iterations):
        handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
        # Read data from pipeline buffer
        handle.release()  # Signal buffer is empty
    ```

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineAsync.sync_object_full "Link to this definition")

    sync\_object\_empty*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineAsync.sync_object_empty "Link to this definition")

    num\_stages*: int*[#](#cutlass.pipeline.PipelineAsync.num_stages "Link to this definition")

    producer\_mask*: cutlass.cutlass\_dsl.Int32 | None*[#](#cutlass.pipeline.PipelineAsync.producer_mask "Link to this definition")

    consumer\_mask*: cutlass.cutlass\_dsl.Int32 | None*[#](#cutlass.pipeline.PipelineAsync.consumer_mask "Link to this definition")

    *static* \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, ) → [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")[#](#cutlass.pipeline.PipelineAsync._make_sync_object "Link to this definition")
    :   Returns a SyncObject corresponding to an agentâs PipelineOp.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineAsync.create "Link to this definition")
    :   Creates and initializes a new PipelineAsync instance.

        This helper function computes necessary attributes and returns an instance of PipelineAsync
        with the specified configuration for producer and consumer synchronization.

        Parameters:
        :   * **barrier\_storage** (*cute.Pointer*) â Pointer to the shared memory address for this pipelineâs mbarriers
            * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the consumer agent
            * **producer\_mask** (*Int32**,* *optional*) â Mask for signaling arrives for the producer agent
            * **consumer\_mask** (*Int32**,* *optional*) â Mask for signaling arrives for the consumer agent

        Raises:
        :   **ValueError** â If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new `PipelineAsync` instance

        Return type:
        :   [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.producer_acquire "Link to this definition")

    producer\_try\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.producer_try_acquire "Link to this definition")

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.producer_commit "Link to this definition")

    consumer\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.consumer_wait "Link to this definition")

    consumer\_try\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.consumer_try_wait "Link to this definition")

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.consumer_release "Link to this definition")

    producer\_get\_barrier( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.pipeline.PipelineAsync.producer_get_barrier "Link to this definition")

    producer\_tail( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsync.producer_tail "Link to this definition")
    :   Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        Parameters:
        :   **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) â The pipeline state that points to next useful buffer

    make\_producer(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineAsync.make_producer "Link to this definition")

    make\_consumer(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineAsync.make_consumer "Link to this definition")

    make\_participants(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineAsync.make_participants "Link to this definition")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None[#](#cutlass.pipeline.PipelineAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineCpAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )[#](#cutlass.pipeline.PipelineCpAsync "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineCpAsync is used for CpAsync producers and AsyncThread consumers

    *static* create( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: cutlass.cutlass\_dsl.Int32*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineCpAsync.create "Link to this definition")
    :   Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

        Parameters:
        :   * **barrier\_storage** (*cute.Pointer*) â Pointer to the shared memory address for this pipelineâs mbarriers
            * **num\_stages** (*Int32*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the consumer agent
            * **producer\_mask** (*Int32**,* *optional*) â Mask for signaling arrives for the producer agent, defaults to None
            * **consumer\_mask** (*Int32**,* *optional*) â Mask for signaling arrives for the consumer agent, defaults to None

        Returns:
        :   A new `PipelineCpAsync` instance configured with the provided parameters

        Return type:
        :   [PipelineCpAsync](#cutlass.pipeline.PipelineCpAsync "cutlass.pipeline.PipelineCpAsync")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None[#](#cutlass.pipeline.PipelineCpAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineTmaAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, )[#](#cutlass.pipeline.PipelineTmaAsync "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

    is\_signalling\_thread*: cutlass.cutlass\_dsl.Boolean*[#](#cutlass.pipeline.PipelineTmaAsync.is_signalling_thread "Link to this definition")

    *static* init\_empty\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: cutlass.cutlass\_dsl.Int32*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, )[#](#cutlass.pipeline.PipelineTmaAsync.init_empty_barrier_arrive_signal "Link to this definition")
    :   Initialize the empty barrier arrive signal.

        This function determines which threads should signal empty barrier arrives based on the cluster layout
        and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

        Parameters:
        :   * **cta\_layout\_vmnk** (*cute.Layout*) â Layout describing the cluster shape and CTA arrangement
            * **tidx** (*Int32*) â Thread index within the warp
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]*) â Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

        Raises:
        :   **AssertionError** â If both multicast modes are disabled (0,0)

        Returns:
        :   Tuple containing destination CTA rank and boolean indicating if current thread signals

        Return type:
        :   tuple[Int32, Boolean]

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *tidx: cutlass.cutlass\_dsl.Int32 | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineTmaAsync.create "Link to this definition")
    :   Create a new `PipelineTmaAsync` instance.

        Parameters:
        :   * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the consumer agent
            * **tx\_count** (*int*) â Number of bytes expected to be written to the transaction barrier for one stage
            * **barrier\_storage** (*cute.Pointer**,* *optional*) â Pointer to the shared memory address for this pipelineâs mbarriers, defaults to None
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) â Layout of the cluster shape, defaults to None
            * **tidx** (*Int32**,* *optional*) â Thread index to consumer async threads, defaults to None
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) â Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

        Raises:
        :   **ValueError** â If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New `PipelineTmaAsync` instance

        Return type:
        :   [PipelineTmaAsync](#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.PipelineTmaAsync")

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaAsync.producer_acquire "Link to this definition")
    :   TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaAsync.producer_commit "Link to this definition")
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaAsync.consumer_release "Link to this definition")
    :   TMA consumer release conditionally signals the empty buffer to the producer.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, ) → None[#](#cutlass.pipeline.PipelineTmaAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineTmaUmma( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )[#](#cutlass.pipeline.PipelineTmaUmma "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

    is\_leader\_cta*: bool*[#](#cutlass.pipeline.PipelineTmaUmma.is_leader_cta "Link to this definition")

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*[#](#cutlass.pipeline.PipelineTmaUmma.cta_group "Link to this definition")

    \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *\**, : *loc=None*, : *ip=None*, ) → [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")[#](#cutlass.pipeline.PipelineTmaUmma._make_sync_object "Link to this definition")
    :   Returns a SyncObject corresponding to an agentâs PipelineOp.

    \_compute\_mcast\_arrival\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *mcast\_mode\_mn: tuple[int, int]*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaUmma._compute_mcast_arrival_mask "Link to this definition")
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaUmma._compute_is_leader_cta "Link to this definition")
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaUmma.create "Link to this definition")
    :   Creates and initializes a new PipelineTmaUmma instance.

        Parameters:
        :   * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â CooperativeGroup for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â CooperativeGroup for the consumer agent
            * **tx\_count** (*int*) â Number of bytes expected to be written to the transaction barrier for one stage
            * **barrier\_storage** (*cute.Pointer**,* *optional*) â Pointer to the shared memory address for this pipelineâs mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) â Layout of the cluster shape
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) â Tuple specifying multicast modes for m and n dimensions (each 0 or 1)

        Raises:
        :   **ValueError** â If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineTmaUmma instance configured with the provided parameters

        Return type:
        :   [PipelineTmaUmma](#cutlass.pipeline.PipelineTmaUmma "cutlass.pipeline.PipelineTmaUmma")

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaUmma.consumer_release "Link to this definition")
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaUmma.producer_acquire "Link to this definition")
    :   TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )[#](#cutlass.pipeline.PipelineTmaUmma.producer_commit "Link to this definition")
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None[#](#cutlass.pipeline.PipelineTmaUmma.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineAsyncUmma( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )[#](#cutlass.pipeline.PipelineAsyncUmma "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*[#](#cutlass.pipeline.PipelineAsyncUmma.cta_group "Link to this definition")

    \_compute\_leading\_cta\_rank( : *cta\_v\_size*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsyncUmma._compute_leading_cta_rank "Link to this definition")
    :   Computes the leading CTA rank.

    \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsyncUmma._compute_is_leader_cta "Link to this definition")
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    \_compute\_peer\_cta\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsyncUmma._compute_peer_cta_mask "Link to this definition")
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsyncUmma.create "Link to this definition")
    :   Creates and initializes a new PipelineAsyncUmma instance.

        Parameters:
        :   * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â CooperativeGroup for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â CooperativeGroup for the consumer agent
            * **barrier\_storage** (*cute.Pointer**,* *optional*) â Pointer to the shared memory address for this pipelineâs mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) â Layout of the cluster shape

        Raises:
        :   **ValueError** â If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineAsyncUmma instance configured with the provided parameters

        Return type:
        :   [PipelineAsyncUmma](#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.PipelineAsyncUmma")

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineAsyncUmma.consumer_release "Link to this definition")
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None[#](#cutlass.pipeline.PipelineAsyncUmma.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineUmmaAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )[#](#cutlass.pipeline.PipelineUmmaAsync "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*[#](#cutlass.pipeline.PipelineUmmaAsync.cta_group "Link to this definition")

    \_compute\_tmem\_sync\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineUmmaAsync._compute_tmem_sync_mask "Link to this definition")
    :   Computes a mask to signal completion of tmem buffers for 2CTA kernels.

    \_compute\_peer\_cta\_rank(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineUmmaAsync._compute_peer_cta_rank "Link to this definition")
    :   Computes a mask to signal release of tmem buffers for 2CTA kernels.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineUmmaAsync.create "Link to this definition")
    :   Creates an instance of PipelineUmmaAsync with computed attributes.

        Parameters:
        :   * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the consumer agent
            * **barrier\_storage** (*cute.Pointer**,* *optional*) â Pointer to the shared memory address for this pipelineâs mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) â Layout of the cluster shape

        Raises:
        :   **ValueError** â If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New instance of `PipelineUmmaAsync`

        Return type:
        :   [PipelineUmmaAsync](#cutlass.pipeline.PipelineUmmaAsync "cutlass.pipeline.PipelineUmmaAsync")

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineUmmaAsync.producer_commit "Link to this definition")
    :   UMMA producer commit buffer full, cta\_group needs to be provided.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None[#](#cutlass.pipeline.PipelineUmmaAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineClcFetchAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, )[#](#cutlass.pipeline.PipelineClcFetchAsync "Link to this definition")
:   Bases: `object`

    PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
    Control based dynamic scheduling. Both producer and consumer operate asynchronously
    using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

    * Producer: waits for empty buffer, signals full barrier with transection bytes
      across all CTAs in cluster, hardware autosignals each CTAâs mbarrier when
      transaction bytes are written, then the satte advance to next buffer slot.
    * Consumer: waits for full barrier, then load respinse from local SMEM, then
      sigals CTA 0âs empty barrier to allow buffer reuse.

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineClcFetchAsync.sync_object_full "Link to this definition")

    sync\_object\_empty*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineClcFetchAsync.sync_object_empty "Link to this definition")

    num\_stages*: int*[#](#cutlass.pipeline.PipelineClcFetchAsync.num_stages "Link to this definition")

    producer\_mask*: cutlass.cutlass\_dsl.Int32 | None*[#](#cutlass.pipeline.PipelineClcFetchAsync.producer_mask "Link to this definition")

    consumer\_mask*: cutlass.cutlass\_dsl.Int32 | None*[#](#cutlass.pipeline.PipelineClcFetchAsync.consumer_mask "Link to this definition")

    is\_signalling\_thread*: cutlass.cutlass\_dsl.Boolean*[#](#cutlass.pipeline.PipelineClcFetchAsync.is_signalling_thread "Link to this definition")

    *static* \_init\_full\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: cutlass.cutlass\_dsl.Int32*, )[#](#cutlass.pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal "Link to this definition")
    :   Computes producer barrier signaling parameters, returns destination CTA rank
        (0 to cluster\_size-1) based on thread ID, and a boolean flag indicating if
        this thread participates in signaling.

        Parameters:
        :   * **cta\_layout\_vmnk** â Cluster layout defining CTA count
            * **tidx** â Thread ID within the CTA

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineClcFetchAsync.create "Link to this definition")
    :   This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
        :param barrier\_storage: Pointer to the shared memory address for this pipelineâs mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: int
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group: CooperativeGroup for the consumer agent
        :type consumer\_group: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param producer\_mask: Mask for signaling arrives for the producer agent, defaults to `None`
        :type producer\_mask: Int32, optional
        :param consumer\_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
        :type consumer\_mask: Int32, optional

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineClcFetchAsync.producer_acquire "Link to this definition")
    :   Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

        Parameters:
        :   * **state** â Pipeline state pointing to the current buffer stage
            * **try\_acquire\_token** â Optional token to skip the empty barrier wait

    consumer\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineClcFetchAsync.consumer_wait "Link to this definition")
    :   Consumer waits for full barrier to be signaled by hardware multicast.

        Parameters:
        :   * **state** â Pipeline state pointing to the current buffer stage
            * **try\_wait\_token** â Optional token to skip the full barrier wait

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineClcFetchAsync.consumer_release "Link to this definition")

    producer\_get\_barrier( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer[#](#cutlass.pipeline.PipelineClcFetchAsync.producer_get_barrier "Link to this definition")

    producer\_tail( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineClcFetchAsync.producer_tail "Link to this definition")
    :   Ensures all in-flight buffers are released before producer exits.

        Parameters:
        :   * **state** â Pipeline state with current position in the buffer
            * **try\_acquire\_token** â Optional token to skip the empty barrier waits

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, ) → None[#](#cutlass.pipeline.PipelineClcFetchAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineTmaMultiConsumersAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

    is\_leader\_cta*: bool*[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_leader_cta "Link to this definition")

    sync\_object\_empty\_umma*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_umma "Link to this definition")

    sync\_object\_empty\_async*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_async "Link to this definition")

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.cta_group "Link to this definition")

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_umma: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_async: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, )[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.create "Link to this definition")
    :   This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
        :param barrier\_storage: Pointer to the smem address for this pipelineâs mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: Int32
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group\_umma: CooperativeGroup for the UMMA consumer agent
        :type consumer\_group\_umma: CooperativeGroup
        :param consumer\_group\_async: CooperativeGroup for the AsyncThread consumer agent
        :type consumer\_group\_async: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param cta\_layout\_vmnk: Layout of the cluster shape
        :type cta\_layout\_vmnk: cute.Layout | None

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_acquire "Link to this definition")
    :   TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_commit "Link to this definition")
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *op\_type: [PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_release "Link to this definition")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None[#](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineTmaStore( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )[#](#cutlass.pipeline.PipelineTmaStore "Link to this definition")
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )[#](#cutlass.pipeline.PipelineTmaStore.create "Link to this definition")
    :   This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

        Parameters:
        :   * **num\_stages** (*int*) â Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â `CooperativeGroup` for the producer agent

        Returns:
        :   A new `PipelineTmaStore` instance

        Return type:
        :   [PipelineTmaStore](#cutlass.pipeline.PipelineTmaStore "cutlass.pipeline.PipelineTmaStore")

    producer\_acquire(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineTmaStore.producer_acquire "Link to this definition")

    producer\_commit(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineTmaStore.producer_commit "Link to this definition")

    consumer\_wait(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineTmaStore.consumer_wait "Link to this definition")

    consumer\_release(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineTmaStore.consumer_release "Link to this definition")

    producer\_tail(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineTmaStore.producer_tail "Link to this definition")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None[#](#cutlass.pipeline.PipelineTmaStore.__init__ "Link to this definition")

*class* cutlass.pipeline.PipelineProducer( : *pipeline*, : *state*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )[#](#cutlass.pipeline.PipelineProducer "Link to this definition")
:   Bases: `object`

    A class representing a producer in an asynchronous pipeline.

    This class manages the producer side of an asynchronous pipeline, handling
    synchronization and state management for producing data. It provides methods for
    acquiring, committing, and advancing through pipeline stages.

    Variables:
    :   * **\_\_pipeline** â The asynchronous pipeline this producer belongs to
        * **\_\_state** â The current state of the producer in the pipeline
        * **\_\_group** â The cooperative group this producer operates in

    **Examples:**

    ```python
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to acquire the current buffer without blocking
        try_acquire_token = producer.try_acquire()

        # Do something else independently
        ...

        # Wait for current buffer to be empty & Move index to next stage
        # If try_acquire_token is True, return immediately
        # If try_acquire_token is False, block until buffer is empty
        handle = producer.acquire_and_advance(try_acquire_token)

        # Produce data
        handle.commit()
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )[#](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "Link to this definition")
    :   Bases: `ImmutableResourceHandle`

        *property* barrier[#](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.barrier "Link to this definition")
        :   Get the barrier pointer for the current pipeline stage.

            Returns:
            :   Pointer to the barrier for the current stage

            Return type:
            :   cute.Pointer

        commit(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.commit "Link to this definition")
        :   Signal that data production is complete for the current stage.

            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None[#](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.__init__ "Link to this definition")

    \_\_init\_\_( : *pipeline*, : *state*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )[#](#cutlass.pipeline.PipelineProducer.__init__ "Link to this definition")
    :   Initialize a new Producer instance.

        Parameters:
        :   * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) â The pipeline this producer belongs to
            * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) â Initial pipeline state
            * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*[#](#cutlass.pipeline.PipelineProducer.__pipeline "Link to this definition")

    \_\_state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*[#](#cutlass.pipeline.PipelineProducer.__state "Link to this definition")

    \_\_group*: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*[#](#cutlass.pipeline.PipelineProducer.__group "Link to this definition")

    clone()[#](#cutlass.pipeline.PipelineProducer.clone "Link to this definition")
    :   Create a new Producer instance with the same state.

    reset(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineProducer.reset "Link to this definition")
    :   Reset the count of how many handles this producer has committed.

    acquire( : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")[#](#cutlass.pipeline.PipelineProducer.acquire "Link to this definition")
    :   Wait for the current buffer to be empty before producing data.
        This is a blocking operation.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[**Boolean**]*) â Optional token to try to acquire the buffer

        Returns:
        :   A handle to the producer for committing the data

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    advance(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineProducer.advance "Link to this definition")
    :   Move to the next pipeline stage.

    acquire\_and\_advance( : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")[#](#cutlass.pipeline.PipelineProducer.acquire_and_advance "Link to this definition")
    :   Acquire the current buffer and advance to the next pipeline stage.

        This method combines the acquire() and advance() operations into a single call.
        It first waits for the current buffer to be empty before producing data,
        then advances the pipeline to the next stage.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[**Boolean**]*) â Token indicating whether to try non-blocking acquire.
            If True, returns immediately without waiting. If False or None, blocks
            until buffer is empty.

        Returns:
        :   A handle to the producer that can be used to commit data to the
            acquired buffer stage

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    try\_acquire( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean[#](#cutlass.pipeline.PipelineProducer.try_acquire "Link to this definition")
    :   Attempt to acquire the current buffer without blocking.

        This method tries to acquire the current buffer stage for producing data
        without waiting. It can be used to check buffer availability before
        committing to a blocking acquire operation.

        Returns:
        :   A boolean token indicating whether the buffer was successfully acquired

        Return type:
        :   Boolean

    commit( : *handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle") | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineProducer.commit "Link to this definition")
    :   Signal that data production is complete for the current stage.

        This allows consumers to start processing the data.

        Parameters:
        :   **handle** (*Optional**[*[*ImmutableResourceHandle*](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")*]*) â Optional handle to commit, defaults to None

        Raises:
        :   **AssertionError** â If provided handle does not belong to this producer

    tail(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineProducer.tail "Link to this definition")
    :   Ensure all used buffers are properly synchronized before producer exit.

        This should be called before the producer finishes to avoid dangling signals.

*class* cutlass.pipeline.PipelineConsumer( : *pipeline*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )[#](#cutlass.pipeline.PipelineConsumer "Link to this definition")
:   Bases: `object`

    A class representing a consumer in an asynchronous pipeline.

    The Consumer class manages the consumer side of an asynchronous pipeline, handling
    synchronization and state management for consuming data. It provides methods for
    waiting, releasing, and advancing through pipeline stages.

    Variables:
    :   * **\_\_pipeline** â The asynchronous pipeline this consumer belongs to
        * **\_\_state** â The current state of the consumer in the pipeline
        * **\_\_group** â The cooperative group this consumer operates in

    **Examples:**

    ```python
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to wait for buffer to be full
        try_wait_token = consumer.try_wait()

        # Do something else independently
        ...

        # Wait for buffer to be full & Move index to next stage
        # If try_wait_token is True, return immediately
        # If try_wait_token is False, block until buffer is full
        handle = consumer.wait_and_advance(try_wait_token)

        # Consume data
        handle.release(  )  # Signal buffer is empty

        # Alternative way to do this is:
        # handle.release()  # Signal buffer is empty
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )[#](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "Link to this definition")
    :   Bases: `ImmutableResourceHandle`

        release(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.release "Link to this definition")
        :   Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None[#](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.__init__ "Link to this definition")

    \_\_init\_\_( : *pipeline*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )[#](#cutlass.pipeline.PipelineConsumer.__init__ "Link to this definition")
    :   Initialize a new Consumer instance.

        Parameters:
        :   * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) â The pipeline this consumer belongs to
            * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) â Initial pipeline state
            * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) â The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*[#](#cutlass.pipeline.PipelineConsumer.__pipeline "Link to this definition")

    \_\_group*: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*[#](#cutlass.pipeline.PipelineConsumer.__group "Link to this definition")

    \_\_state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*[#](#cutlass.pipeline.PipelineConsumer.__state "Link to this definition")

    clone()[#](#cutlass.pipeline.PipelineConsumer.clone "Link to this definition")
    :   Create a new Consumer instance with the same state.

    reset(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineConsumer.reset "Link to this definition")
    :   Reset the count of how many handles this consumer has consumed.

    wait( : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")[#](#cutlass.pipeline.PipelineConsumer.wait "Link to this definition")
    :   Wait for data to be ready in the current buffer. This is a blocking operation
        that will not return until data is available.

        Parameters:
        :   **try\_wait\_token** (*Optional**[**Boolean**]*) â Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    advance(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.PipelineConsumer.advance "Link to this definition")
    :   Advance the consumer to the next pipeline stage.

        This updates the internal state to point to the next buffer in the pipeline.
        Should be called after consuming data from the current buffer.

    wait\_and\_advance( : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")[#](#cutlass.pipeline.PipelineConsumer.wait_and_advance "Link to this definition")
    :   Atomically wait for data and advance to next pipeline stage.

        This is a convenience method that combines wait() and advance() into a single
        atomic operation. It will block until data is available in the current buffer,
        then automatically advance to the next stage.

        Parameters:
        :   **try\_wait\_token** (*Optional**[**Boolean**]*) â Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    try\_wait( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean[#](#cutlass.pipeline.PipelineConsumer.try_wait "Link to this definition")
    :   Non-blocking check if data is ready in the current buffer.

        This method provides a way to test if data is available without blocking.
        Unlike wait(), this will return immediately regardless of buffer state.

        Returns:
        :   True if data is ready to be consumed, False if the buffer is not yet ready

        Return type:
        :   Boolean

    release( : *handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle") | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.PipelineConsumer.release "Link to this definition")
    :   Signal that data consumption is complete for the current stage.
        This allows producers to start producing new data.

cutlass.pipeline.make\_pipeline\_state( : *type: [PipelineUserType](#cutlass.pipeline.PipelineUserType "cutlass.pipeline.helpers.PipelineUserType")*, : *stages: int*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.make_pipeline_state "Link to this definition")
:   Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

cutlass.pipeline.pipeline\_init\_arrive( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, : *is\_relaxed: bool = False*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.pipeline_init_arrive "Link to this definition")
:   Fences the mbarrier\_init and sends an arrive if using clusters.

cutlass.pipeline.pipeline\_init\_wait( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.pipeline_init_wait "Link to this definition")
:   Syncs the threadblock or cluster

cutlass.pipeline.agent\_sync( : *group: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *is\_relaxed: bool = False*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.agent_sync "Link to this definition")
:   Syncs all threads within an agent.

cutlass.pipeline.arrive(*barrier\_id: int*, *num\_threads: int*, *\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.arrive "Link to this definition")
:   The aligned flavor of arrive is used when all threads in the CTA will execute the
    same instruction. See PTX documentation.

cutlass.pipeline.arrive\_unaligned( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.arrive_unaligned "Link to this definition")
:   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

cutlass.pipeline.wait(*\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.wait "Link to this definition")
:   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
    If synchronizing two warps in a producer/consumer pairing, the arrive count would be
    32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
    or consumer are counted for mbarriers, while all threads participating in the sync
    are counted for NamedBarriers.

cutlass.pipeline.wait\_unaligned( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.wait_unaligned "Link to this definition")

cutlass.pipeline.arrive\_and\_wait( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )[#](#cutlass.pipeline.arrive_and_wait "Link to this definition")

cutlass.pipeline.sync(*barrier\_id: int = 0*, *\**, *loc=None*, *ip=None*)[#](#cutlass.pipeline.sync "Link to this definition")

[previous

tcgen05 submodule](cute_nvgpu_tcgen05.html "previous page")
[next

cutlass.utils](utils.html "next page")

On this page

* [`Agent`](#cutlass.pipeline.Agent)
  + [`Thread`](#cutlass.pipeline.Agent.Thread)
  + [`ThreadBlock`](#cutlass.pipeline.Agent.ThreadBlock)
  + [`ThreadBlockCluster`](#cutlass.pipeline.Agent.ThreadBlockCluster)
* [`CooperativeGroup`](#cutlass.pipeline.CooperativeGroup)
  + [`__init__()`](#cutlass.pipeline.CooperativeGroup.__init__)
* [`PipelineOp`](#cutlass.pipeline.PipelineOp)
  + [`AsyncThread`](#cutlass.pipeline.PipelineOp.AsyncThread)
  + [`TCGen05Mma`](#cutlass.pipeline.PipelineOp.TCGen05Mma)
  + [`TmaLoad`](#cutlass.pipeline.PipelineOp.TmaLoad)
  + [`ClcLoad`](#cutlass.pipeline.PipelineOp.ClcLoad)
  + [`TmaStore`](#cutlass.pipeline.PipelineOp.TmaStore)
  + [`Composite`](#cutlass.pipeline.PipelineOp.Composite)
  + [`AsyncLoad`](#cutlass.pipeline.PipelineOp.AsyncLoad)
* [`SyncObject`](#cutlass.pipeline.SyncObject)
  + [`arrive()`](#cutlass.pipeline.SyncObject.arrive)
  + [`wait()`](#cutlass.pipeline.SyncObject.wait)
  + [`arrive_and_wait()`](#cutlass.pipeline.SyncObject.arrive_and_wait)
  + [`arrive_and_drop()`](#cutlass.pipeline.SyncObject.arrive_and_drop)
  + [`get_barrier()`](#cutlass.pipeline.SyncObject.get_barrier)
  + [`max()`](#cutlass.pipeline.SyncObject.max)
  + [`_abc_impl`](#cutlass.pipeline.SyncObject._abc_impl)
* [`MbarrierArray`](#cutlass.pipeline.MbarrierArray)
  + [`__init__()`](#cutlass.pipeline.MbarrierArray.__init__)
  + [`recast_to_new_op_type()`](#cutlass.pipeline.MbarrierArray.recast_to_new_op_type)
  + [`mbarrier_init()`](#cutlass.pipeline.MbarrierArray.mbarrier_init)
  + [`arrive()`](#cutlass.pipeline.MbarrierArray.arrive)
  + [`arrive_mbarrier()`](#cutlass.pipeline.MbarrierArray.arrive_mbarrier)
  + [`arrive_cp_async_mbarrier()`](#cutlass.pipeline.MbarrierArray.arrive_cp_async_mbarrier)
  + [`arrive_tcgen05mma()`](#cutlass.pipeline.MbarrierArray.arrive_tcgen05mma)
  + [`arrive_and_expect_tx()`](#cutlass.pipeline.MbarrierArray.arrive_and_expect_tx)
  + [`arrive_and_expect_tx_with_dst()`](#cutlass.pipeline.MbarrierArray.arrive_and_expect_tx_with_dst)
  + [`try_wait()`](#cutlass.pipeline.MbarrierArray.try_wait)
  + [`wait()`](#cutlass.pipeline.MbarrierArray.wait)
  + [`arrive_and_wait()`](#cutlass.pipeline.MbarrierArray.arrive_and_wait)
  + [`arrive_and_drop()`](#cutlass.pipeline.MbarrierArray.arrive_and_drop)
  + [`get_barrier()`](#cutlass.pipeline.MbarrierArray.get_barrier)
  + [`max()`](#cutlass.pipeline.MbarrierArray.max)
  + [`_abc_impl`](#cutlass.pipeline.MbarrierArray._abc_impl)
* [`NamedBarrier`](#cutlass.pipeline.NamedBarrier)
  + [`barrier_id`](#cutlass.pipeline.NamedBarrier.barrier_id)
  + [`num_threads`](#cutlass.pipeline.NamedBarrier.num_threads)
  + [`arrive()`](#cutlass.pipeline.NamedBarrier.arrive)
  + [`arrive_unaligned()`](#cutlass.pipeline.NamedBarrier.arrive_unaligned)
  + [`wait()`](#cutlass.pipeline.NamedBarrier.wait)
  + [`wait_unaligned()`](#cutlass.pipeline.NamedBarrier.wait_unaligned)
  + [`arrive_and_wait()`](#cutlass.pipeline.NamedBarrier.arrive_and_wait)
  + [`arrive_and_drop()`](#cutlass.pipeline.NamedBarrier.arrive_and_drop)
  + [`sync()`](#cutlass.pipeline.NamedBarrier.sync)
  + [`get_barrier()`](#cutlass.pipeline.NamedBarrier.get_barrier)
  + [`max()`](#cutlass.pipeline.NamedBarrier.max)
  + [`__init__()`](#cutlass.pipeline.NamedBarrier.__init__)
  + [`_abc_impl`](#cutlass.pipeline.NamedBarrier._abc_impl)
* [`PipelineOrder`](#cutlass.pipeline.PipelineOrder)
  + [`sync_object_full`](#cutlass.pipeline.PipelineOrder.sync_object_full)
  + [`depth`](#cutlass.pipeline.PipelineOrder.depth)
  + [`length`](#cutlass.pipeline.PipelineOrder.length)
  + [`group_id`](#cutlass.pipeline.PipelineOrder.group_id)
  + [`state`](#cutlass.pipeline.PipelineOrder.state)
  + [`create()`](#cutlass.pipeline.PipelineOrder.create)
  + [`get_barrier_for_current_stage_idx()`](#cutlass.pipeline.PipelineOrder.get_barrier_for_current_stage_idx)
  + [`arrive()`](#cutlass.pipeline.PipelineOrder.arrive)
  + [`wait()`](#cutlass.pipeline.PipelineOrder.wait)
  + [`__init__()`](#cutlass.pipeline.PipelineOrder.__init__)
* [`TmaStoreFence`](#cutlass.pipeline.TmaStoreFence)
  + [`__init__()`](#cutlass.pipeline.TmaStoreFence.__init__)
  + [`arrive()`](#cutlass.pipeline.TmaStoreFence.arrive)
  + [`wait()`](#cutlass.pipeline.TmaStoreFence.wait)
  + [`arrive_and_wait()`](#cutlass.pipeline.TmaStoreFence.arrive_and_wait)
  + [`arrive_and_drop()`](#cutlass.pipeline.TmaStoreFence.arrive_and_drop)
  + [`get_barrier()`](#cutlass.pipeline.TmaStoreFence.get_barrier)
  + [`max()`](#cutlass.pipeline.TmaStoreFence.max)
  + [`tail()`](#cutlass.pipeline.TmaStoreFence.tail)
  + [`_abc_impl`](#cutlass.pipeline.TmaStoreFence._abc_impl)
* [`PipelineUserType`](#cutlass.pipeline.PipelineUserType)
  + [`Producer`](#cutlass.pipeline.PipelineUserType.Producer)
  + [`Consumer`](#cutlass.pipeline.PipelineUserType.Consumer)
  + [`ProducerConsumer`](#cutlass.pipeline.PipelineUserType.ProducerConsumer)
* [`PipelineState`](#cutlass.pipeline.PipelineState)
  + [`__init__()`](#cutlass.pipeline.PipelineState.__init__)
  + [`clone()`](#cutlass.pipeline.PipelineState.clone)
  + [`index`](#cutlass.pipeline.PipelineState.index)
  + [`count`](#cutlass.pipeline.PipelineState.count)
  + [`stages`](#cutlass.pipeline.PipelineState.stages)
  + [`phase`](#cutlass.pipeline.PipelineState.phase)
  + [`reset_count()`](#cutlass.pipeline.PipelineState.reset_count)
  + [`advance()`](#cutlass.pipeline.PipelineState.advance)
  + [`reverse()`](#cutlass.pipeline.PipelineState.reverse)
* [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)
  + [`sync_object_full`](#cutlass.pipeline.PipelineAsync.sync_object_full)
  + [`sync_object_empty`](#cutlass.pipeline.PipelineAsync.sync_object_empty)
  + [`num_stages`](#cutlass.pipeline.PipelineAsync.num_stages)
  + [`producer_mask`](#cutlass.pipeline.PipelineAsync.producer_mask)
  + [`consumer_mask`](#cutlass.pipeline.PipelineAsync.consumer_mask)
  + [`_make_sync_object()`](#cutlass.pipeline.PipelineAsync._make_sync_object)
  + [`create()`](#cutlass.pipeline.PipelineAsync.create)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineAsync.producer_acquire)
  + [`producer_try_acquire()`](#cutlass.pipeline.PipelineAsync.producer_try_acquire)
  + [`producer_commit()`](#cutlass.pipeline.PipelineAsync.producer_commit)
  + [`consumer_wait()`](#cutlass.pipeline.PipelineAsync.consumer_wait)
  + [`consumer_try_wait()`](#cutlass.pipeline.PipelineAsync.consumer_try_wait)
  + [`consumer_release()`](#cutlass.pipeline.PipelineAsync.consumer_release)
  + [`producer_get_barrier()`](#cutlass.pipeline.PipelineAsync.producer_get_barrier)
  + [`producer_tail()`](#cutlass.pipeline.PipelineAsync.producer_tail)
  + [`make_producer()`](#cutlass.pipeline.PipelineAsync.make_producer)
  + [`make_consumer()`](#cutlass.pipeline.PipelineAsync.make_consumer)
  + [`make_participants()`](#cutlass.pipeline.PipelineAsync.make_participants)
  + [`__init__()`](#cutlass.pipeline.PipelineAsync.__init__)
* [`PipelineCpAsync`](#cutlass.pipeline.PipelineCpAsync)
  + [`create()`](#cutlass.pipeline.PipelineCpAsync.create)
  + [`__init__()`](#cutlass.pipeline.PipelineCpAsync.__init__)
* [`PipelineTmaAsync`](#cutlass.pipeline.PipelineTmaAsync)
  + [`is_signalling_thread`](#cutlass.pipeline.PipelineTmaAsync.is_signalling_thread)
  + [`init_empty_barrier_arrive_signal()`](#cutlass.pipeline.PipelineTmaAsync.init_empty_barrier_arrive_signal)
  + [`create()`](#cutlass.pipeline.PipelineTmaAsync.create)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineTmaAsync.producer_acquire)
  + [`producer_commit()`](#cutlass.pipeline.PipelineTmaAsync.producer_commit)
  + [`consumer_release()`](#cutlass.pipeline.PipelineTmaAsync.consumer_release)
  + [`__init__()`](#cutlass.pipeline.PipelineTmaAsync.__init__)
* [`PipelineTmaUmma`](#cutlass.pipeline.PipelineTmaUmma)
  + [`is_leader_cta`](#cutlass.pipeline.PipelineTmaUmma.is_leader_cta)
  + [`cta_group`](#cutlass.pipeline.PipelineTmaUmma.cta_group)
  + [`_make_sync_object()`](#cutlass.pipeline.PipelineTmaUmma._make_sync_object)
  + [`_compute_mcast_arrival_mask()`](#cutlass.pipeline.PipelineTmaUmma._compute_mcast_arrival_mask)
  + [`_compute_is_leader_cta()`](#cutlass.pipeline.PipelineTmaUmma._compute_is_leader_cta)
  + [`create()`](#cutlass.pipeline.PipelineTmaUmma.create)
  + [`consumer_release()`](#cutlass.pipeline.PipelineTmaUmma.consumer_release)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineTmaUmma.producer_acquire)
  + [`producer_commit()`](#cutlass.pipeline.PipelineTmaUmma.producer_commit)
  + [`__init__()`](#cutlass.pipeline.PipelineTmaUmma.__init__)
* [`PipelineAsyncUmma`](#cutlass.pipeline.PipelineAsyncUmma)
  + [`cta_group`](#cutlass.pipeline.PipelineAsyncUmma.cta_group)
  + [`_compute_leading_cta_rank()`](#cutlass.pipeline.PipelineAsyncUmma._compute_leading_cta_rank)
  + [`_compute_is_leader_cta()`](#cutlass.pipeline.PipelineAsyncUmma._compute_is_leader_cta)
  + [`_compute_peer_cta_mask()`](#cutlass.pipeline.PipelineAsyncUmma._compute_peer_cta_mask)
  + [`create()`](#cutlass.pipeline.PipelineAsyncUmma.create)
  + [`consumer_release()`](#cutlass.pipeline.PipelineAsyncUmma.consumer_release)
  + [`__init__()`](#cutlass.pipeline.PipelineAsyncUmma.__init__)
* [`PipelineUmmaAsync`](#cutlass.pipeline.PipelineUmmaAsync)
  + [`cta_group`](#cutlass.pipeline.PipelineUmmaAsync.cta_group)
  + [`_compute_tmem_sync_mask()`](#cutlass.pipeline.PipelineUmmaAsync._compute_tmem_sync_mask)
  + [`_compute_peer_cta_rank()`](#cutlass.pipeline.PipelineUmmaAsync._compute_peer_cta_rank)
  + [`create()`](#cutlass.pipeline.PipelineUmmaAsync.create)
  + [`producer_commit()`](#cutlass.pipeline.PipelineUmmaAsync.producer_commit)
  + [`__init__()`](#cutlass.pipeline.PipelineUmmaAsync.__init__)
* [`PipelineClcFetchAsync`](#cutlass.pipeline.PipelineClcFetchAsync)
  + [`sync_object_full`](#cutlass.pipeline.PipelineClcFetchAsync.sync_object_full)
  + [`sync_object_empty`](#cutlass.pipeline.PipelineClcFetchAsync.sync_object_empty)
  + [`num_stages`](#cutlass.pipeline.PipelineClcFetchAsync.num_stages)
  + [`producer_mask`](#cutlass.pipeline.PipelineClcFetchAsync.producer_mask)
  + [`consumer_mask`](#cutlass.pipeline.PipelineClcFetchAsync.consumer_mask)
  + [`is_signalling_thread`](#cutlass.pipeline.PipelineClcFetchAsync.is_signalling_thread)
  + [`_init_full_barrier_arrive_signal()`](#cutlass.pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal)
  + [`create()`](#cutlass.pipeline.PipelineClcFetchAsync.create)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineClcFetchAsync.producer_acquire)
  + [`consumer_wait()`](#cutlass.pipeline.PipelineClcFetchAsync.consumer_wait)
  + [`consumer_release()`](#cutlass.pipeline.PipelineClcFetchAsync.consumer_release)
  + [`producer_get_barrier()`](#cutlass.pipeline.PipelineClcFetchAsync.producer_get_barrier)
  + [`producer_tail()`](#cutlass.pipeline.PipelineClcFetchAsync.producer_tail)
  + [`__init__()`](#cutlass.pipeline.PipelineClcFetchAsync.__init__)
* [`PipelineTmaMultiConsumersAsync`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync)
  + [`is_leader_cta`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_leader_cta)
  + [`sync_object_empty_umma`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_umma)
  + [`sync_object_empty_async`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_async)
  + [`cta_group`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.cta_group)
  + [`create()`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.create)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_acquire)
  + [`producer_commit()`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_commit)
  + [`consumer_release()`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_release)
  + [`__init__()`](#cutlass.pipeline.PipelineTmaMultiConsumersAsync.__init__)
* [`PipelineTmaStore`](#cutlass.pipeline.PipelineTmaStore)
  + [`create()`](#cutlass.pipeline.PipelineTmaStore.create)
  + [`producer_acquire()`](#cutlass.pipeline.PipelineTmaStore.producer_acquire)
  + [`producer_commit()`](#cutlass.pipeline.PipelineTmaStore.producer_commit)
  + [`consumer_wait()`](#cutlass.pipeline.PipelineTmaStore.consumer_wait)
  + [`consumer_release()`](#cutlass.pipeline.PipelineTmaStore.consumer_release)
  + [`producer_tail()`](#cutlass.pipeline.PipelineTmaStore.producer_tail)
  + [`__init__()`](#cutlass.pipeline.PipelineTmaStore.__init__)
* [`PipelineProducer`](#cutlass.pipeline.PipelineProducer)
  + [`ImmutableResourceHandle`](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)
    - [`barrier`](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.barrier)
    - [`commit()`](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.commit)
    - [`__init__()`](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.__init__)
  + [`__init__()`](#cutlass.pipeline.PipelineProducer.__init__)
  + [`__pipeline`](#cutlass.pipeline.PipelineProducer.__pipeline)
  + [`__state`](#cutlass.pipeline.PipelineProducer.__state)
  + [`__group`](#cutlass.pipeline.PipelineProducer.__group)
  + [`clone()`](#cutlass.pipeline.PipelineProducer.clone)
  + [`reset()`](#cutlass.pipeline.PipelineProducer.reset)
  + [`acquire()`](#cutlass.pipeline.PipelineProducer.acquire)
  + [`advance()`](#cutlass.pipeline.PipelineProducer.advance)
  + [`acquire_and_advance()`](#cutlass.pipeline.PipelineProducer.acquire_and_advance)
  + [`try_acquire()`](#cutlass.pipeline.PipelineProducer.try_acquire)
  + [`commit()`](#cutlass.pipeline.PipelineProducer.commit)
  + [`tail()`](#cutlass.pipeline.PipelineProducer.tail)
* [`PipelineConsumer`](#cutlass.pipeline.PipelineConsumer)
  + [`ImmutableResourceHandle`](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)
    - [`release()`](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.release)
    - [`__init__()`](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.__init__)
  + [`__init__()`](#cutlass.pipeline.PipelineConsumer.__init__)
  + [`__pipeline`](#cutlass.pipeline.PipelineConsumer.__pipeline)
  + [`__group`](#cutlass.pipeline.PipelineConsumer.__group)
  + [`__state`](#cutlass.pipeline.PipelineConsumer.__state)
  + [`clone()`](#cutlass.pipeline.PipelineConsumer.clone)
  + [`reset()`](#cutlass.pipeline.PipelineConsumer.reset)
  + [`wait()`](#cutlass.pipeline.PipelineConsumer.wait)
  + [`advance()`](#cutlass.pipeline.PipelineConsumer.advance)
  + [`wait_and_advance()`](#cutlass.pipeline.PipelineConsumer.wait_and_advance)
  + [`try_wait()`](#cutlass.pipeline.PipelineConsumer.try_wait)
  + [`release()`](#cutlass.pipeline.PipelineConsumer.release)
* [`make_pipeline_state()`](#cutlass.pipeline.make_pipeline_state)
* [`pipeline_init_arrive()`](#cutlass.pipeline.pipeline_init_arrive)
* [`pipeline_init_wait()`](#cutlass.pipeline.pipeline_init_wait)
* [`agent_sync()`](#cutlass.pipeline.agent_sync)
* [`arrive()`](#cutlass.pipeline.arrive)
* [`arrive_unaligned()`](#cutlass.pipeline.arrive_unaligned)
* [`wait()`](#cutlass.pipeline.wait)
* [`wait_unaligned()`](#cutlass.pipeline.wait_unaligned)
* [`arrive_and_wait()`](#cutlass.pipeline.arrive_and_wait)
* [`sync()`](#cutlass.pipeline.sync)





[![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.