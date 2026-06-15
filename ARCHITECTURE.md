# Project Technical Architecture

## Module Boundaries

The project is structured as a modular Python library with clear separation between the custom CuTe-based fused projection kernels, the Triton epilogue, and the pipelined cp.async GEMM:

1. **CuTe DSL Fused Milestone 1 Kernel (`flash_attn_prologue/fused_pipelined_kernel.py`):**
   - Implements a fused QKV projection kernel doing double-buffered cp.async GEMM, sequence-level RMSNorm reduction, Rotary Position Embedding (RoPE) rotations, and query scaling entirely in registers.
   - Leverages a flat aligned Float32 shared memory layout with manual offset calculation and pointer casting (`recast_ptr`) to optimize occupancy and prevent layout JIT compile bugs.
   - Registers under `flash_prologue::fused_qkv_pipelined_cute` with PyTorch.
2. **CuTe DSL Fused Kernel (`flash_attn_prologue/kernel.py`):**
   - Entry point for the JIT-compiled custom CUDA kernel using CuTe Python DSL (`cutlass.cute`).
   - Fuses linear projection, Q/K RMSNorm, RoPE, and scaling in a single unified thread-block GEMM loop.
   - Registers under `flash_prologue::fused_qkv` with PyTorch.
3. **CuTe DSL Split-N Fused Kernel (`flash_attn_prologue/splitn_kernel.py`):**
   - Specialized fused projection kernel using a 13-warp Cooperative Thread Array (CTA) spatial channel partition design.
   - Designed to avoid TMEM hardware constraints on consumer Blackwell platforms (`sm_120a`) by splitting Q, K, and V computations across 3 parallel compute warpgroups.
   - Registers under `flash_prologue::fused_qkv_cute_splitn` with PyTorch.
4. **Triton Fused Epilogue (`flash_attn_prologue/epilogue_kernel.py`):**
   - Entry point for the standalone Triton fused epilogue kernel `qknorm_rope_scale`.
   - Designed to run immediately after a standard `F.linear` projection. Fuses Q/K RMSNorm, RoPE, and scaling, while the V projection is sliced and permuted directly in PyTorch to minimize memory bandwidth overhead.
   - Registers under `flash_prologue::qknorm_rope_scale` with custom autograd hooks.
5. **Pure CuTe GEMM (`flash_attn_prologue/pure_gemm.py`):**
   - Entry point for the pure CuTe GEMM projection kernel `pure_gemm_cute_op` projecting inputs X to QKV outputs without epilogue fusion.
   - Registers under `flash_prologue::pure_gemm_cute` with PyTorch.
6. **Pipelined cp.async GEMM (`flash_attn_prologue/pipelined_gemm.py`):**
   - Entry point for the optimized 2-stage double-buffered pipelined GEMM kernel `pipelined_gemm_cute_op` using `cp.async` cooperative loading and warp-level MMA.
   - Registers under `flash_prologue::pipelined_gemm_cute` with PyTorch.
7. **Quack TMA GEMM Module (`quack/quack/gemm_sm120.py`, `quack/quack/gemm.py`):**
   - Implements a TMA-based persistent GEMM on sm_120a using Blackwell's asynchronous Pipeline and TMA descriptor operations.
   - Parameterized to support overrides for pipeline stages (`ab_stage`), MMA warp layout (`atom_layout_mnk`), and register allocation (`num_regs_mma`, `num_regs_load`).
8. **Eager Baselines (`flash_attn_prologue/layers/FusedQKV.py`):**
   - Pure PyTorch eager equivalents of the operations used for parity checking.
9. **Fused QKV Kernel (`flash_attn_prologue/fused_qkv.py`):**
   - Implements a resource-optimized fused QKV GEMM projection kernel with parameterized pipeline stages (`ab_stage = 2` or `3`).
   - Grid searched to find the optimal memory and register configuration.
   - Registers under `flash_prologue::fused_qkv` with PyTorch.
10. **In-Place Split-QK Standalone Fused Epilogue (`flash_attn_prologue/fused_epilogue.py`):**
    - Implements a layout-agnostic, standalone high-performance fused epilogue kernel that processes separate input tensors Q and K of shape `[B, H, N, D]`.
    - **Separate Input/Output Pointers & Zero-Copy**: Accepts separate input (`q_in`, `k_in`) and output (`q_out_gmem`, `k_out_gmem`) pointers to remain optimal for PyTorch Inductor layout optimizations, and allocates output buffers directly as uninitialized memory using `torch.empty_strided(...)` to avoid eager CPU/GPU memory copies.
    - **On-Chip Register Upcasting**: Pass parameters (weights, scales, sin/cos tables) directly in their native dtypes (e.g. `BFloat16`), avoiding host-side `.to(torch.float32)` casting overhead. Loads parameters into registers in their native dtypes to match memory widths for `autovec_copy` and performs dynamic register-level upcasting to `Float32` in compute loops.
    - Specialized on input strides at JIT compile time, allowing it to execute directly on permuted non-contiguous view slices at zero runtime cost without PyTorch contiguous conversions.
    - Applies Float32 Q/K RMSNorm, on-the-fly RoPE rotations using half-dimension precomputed `sin`/`cos` buffers of shape `[..., dhead // 2]` loaded directly once to registers and reused twice on-chip, and query scaling.
    - **Stride-Based Batch Broadcasting of RoPE Buffers**: Supports batch dimension broadcasting for RoPE buffers (`sin` and `cos`). When the batch dimension of positional embeddings is 1 (e.g. shape `[1, 1, seq, dhead/2]`), their batch strides are forced to `0` in the JIT compiler to prevent out-of-bounds memory accesses for `cta_b >= 1` and ensure correct broadcasting across all batches.
    - **No Sequence Padding Requirement**: Since the sequences in the data pipeline do not have padding, the redundant `active_mask` zero-masking logic from the backward operators has been completely removed to avoid extra overhead.
    - Padded shared memory strides and block-level validation filters to eliminate LDS bank conflicts and block-level loops.
    - Maps the sequence dimension directly to the grid blocks to process exactly one tile of size 64 per CTA (loopless design), achieving a 2.33x speedup over Triton on non-contiguous slices.
    - Exposes `qknorm_rope_scale_cute_op` and `qknorm_rope_scale_cute_backward_op` via `flash_attn_prologue/__init__.py`, registering them as custom PyTorch operators under `flash_prologue::qknorm_rope_scale_cute` and `flash_prologue::qknorm_rope_scale_cute_backward` with independent autotuning.
 11. **Cloned Backward GEMM Module (`flash_attn_prologue/gemm_bwd.py`):**
    - Implements decoupled `matmul_bwd_dx` and `matmul_bwd_dw` JIT compilation and runner wrappers using Blackwell (SM120) hardware TMA (Tensor Memory Accelerator) loads (`CopyBulkTensorTileG2SOp`) and asynchronous `mbarrier` transaction barriers.
    - Provides two isolated compilation cache routines (`get_matmul_bwd_dx_kernel` and `get_matmul_bwd_dw_kernel`) using QuACK FFI binders.
    - Specializes the JIT compilation cache with static dimensions `B = 32` and `N = 512` to enable compile-time optimization of sequence index division/modulo operations on the GPU (converting costly dynamic division instructions into 1-cycle bit-shifts and mask operations).
    - **Concurrent Component Layouts & Parallel TMA Loads**: Shared memory allocations represent the three components (`Q`, `K`, `V`) contiguously using nested layouts, enabling concurrent loading from split global memory tensors directly into the contiguous SMEM tiles via parallel TMA copies.
    - **Block-Contiguous Layout Alignment**: Aligned `a_smem_layout_staged` and `a_smem_layout_staged_row` in `GemmBwdDwCute` to use block-contiguous layout strides `stride=((1, BLOCK_M * BLOCK_K_STAGE), BLOCK_M, 3 * BLOCK_M * BLOCK_K_STAGE)` matching the load path exactly, resolving a critical layout mismatch that read incorrect shared memory offsets.
    - **TMA Coordinate Head-Index Modulo Map**: Corrected batch-head coordinate calculation in both dx and dw kernels to use `b_h = b * self.NUM_HEADS + (h % self.NUM_HEADS)`, preventing out-of-bounds index wrapping.
    - **Static Contiguous Strides compilation**: Hardcoded static contiguous integer stride tuples `(None, N * HEAD_DIM, HEAD_DIM, 1)` for JIT compilation, ensuring NVCC generates specialized code for contiguous input views and avoiding cache pollution with dynamic strides.
    - **Concrete Integer Shape Casting (`SymInt` Shield)**: Cast shape inputs (`B`, `N`, `NUM_HEADS`, `HEAD_DIM`, `hidden`, `N_dim`, etc.) to Python concrete `int` in the custom op wrappers, ensuring PyTorch trace-time `SymInt`s do not leak into autotuner configurations or JIT compilation keys.
    - **Empty Input Guards**: Bypasses JIT kernel execution and returns zero-initialized outputs immediately if `grad_q.numel() == 0` or `x.numel() == 0`, supporting empty sequence boundaries without compiling/launching threads.
    - **In-place dw Accumulation Custom Op Registration**: To avoid PyTorch autograd output-aliasing errors, registered two distinct ops: `matmul_bwd_dw_cute` (non-mutating) and `matmul_bwd_dw_accumulate_cute` (mutating, `mutates_args={"out"}`). If an output `out` tensor is provided (such as `weight.grad`), the epilogue accumulates in-place directly into it.
    - **3x Activation Bandwidth Savings**: For `bwd_dw`, activations `x` are loaded exactly once into `sW` while split gradients `grad_q`, `grad_k`, and `grad_v` are loaded in parallel into `sX`, eliminating two out of three global activation loads and saving 3x memory bandwidth.
    - **Vectorized BFloat16 Writeback**: Replaced manual writeback loops with a 128-bit `BFloat16` `TiledCopy` vectorized writeback to global memory, completely avoiding coordinate division/modulo overhead in threads. For `SPLIT_K = 1` (or the workspace slices when `SPLIT_K > 1`), Float32 accumulators are cast directly in registers and written as BFloat16.
    - **BFloat16 Workspace & FP32 Register Reduction**: When `SPLIT_K > 1`, blocks write their partial accumulations into unique slices of a BFloat16 global memory workspace (halving bandwidth and VRAM). A custom JIT reduction kernel `SplitKReductionCute` loads these BFloat16 elements and sums them in Float32 register accumulators before writing out the final BFloat16 result, maintaining full numerical precision.
    - **Zero-overhead Persistent Workspace Caching**: Persistent workspaces are allocated using `torch.empty` instead of `torch.zeros` and retrieved without the hot-path `.zero_()` call, eliminating unnecessary GMEM zeroing overhead.
    - **Autotuner Integration**: Integrates the autotuner with a pruned, highly efficient candidate space to independently profile `bwd_dx` and `bwd_dw`. Added `store_thr_n * 8 <= bn` constraint to prevent out-of-bounds JIT compilation. The autotuner dynamically selects and caches the optimal configurations in `autotune_cache.json`:
      - `bwd_dx` (shape 16384_512_3072): `BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 16, ab_stage = 3, num_warps = 4, maxrregcount = 80, swizzle_x = 16, atom_layout_mnk = [1, 4, 1]`.
      - `bwd_dw` (shape 3072_512_16384): `BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 16, ab_stage = 3, num_warps = 4, maxrregcount = 255, SPLIT_K = 12, swizzle_x = 8, atom_layout_mnk = (4, 1, 1)`.
    - **Nested Incremental Counter Loop Optimization**: Eliminated dynamic division and modulo instructions in `GemmBwdDwCute.kernel`'s main loop by refactoring coordinate tracking (`b` and `n_tile`) into registers updated via simple addition and conditional reset steps. Reverted `GemmBwdDxCute.kernel` to `constexpr` unrolling to preserve constant-folding benefits, achieving a 4.0% standalone execution speedup on `dw_kernel`.
12. **Coordinate Descent Autotuner (`advantage_first/layers/cute/coordinate_descent.py`):**
    - Implements the generic, dynamic `CoordinateDescentAutotuner` class used to tune hyperparameters such as block sizes, stages, warp layouts, and registers.
    - Supports dynamic parameter candidate generation of different step types (`power_of_2`, `step_1`, `both`, `maxrregcount`) as well as static list specifications.
    - Implements a greedy local coordinate walk to rapidly optimize parameters individually while maintaining backward compatibility with static grids.
13. **Asynchronous Multi-process Logging Pipeline (`advantage_first/logging/workers.py`):**
    - Decouples the training loops from logging and heavy computational profiling via multi-process isolation:
      - `ScalarLogWorker` (Wandb Interface): Manages active connection handles to Weights & Biases on a background process, ensuring FFI network and HTTP request latencies do not block local training execution.
      - `HeavyMetricsWorker` (CPU Math Profiling): Executes CPU-bound matrix mathematics, including Norm statistics, Cosine collapse ratios, and SVD Entropy-based Effective Rank calculations for all active model state layers.


## Pipeline & Data Flow

### 1. Milestone 1 Fused QKV Pipeline (Dynamic 2 or 3-Stage Cooperative cp.async GEMM + RMSNorm)
```mermaid
graph TD
    X[Input Tensor X: B, N, hidden] --> Prefetch["cp.async Prefetch (ab_stage - 1) Stages: 128 threads"]
    W[Weights W: Q, K, V] --> Prefetch
    Prefetch --> Commit1["cute.arch.cp_async_commit_group()"]
    Commit1 --> Loop["Mainloop: k_tile = 0 to k_tile_cnt (cutlass.range_constexpr)"]
    Loop --> cp_async["cp.async Prefetch Next Stage (k_tile + ab_stage - 1)"]
    cp_async --> Commit2["cute.arch.cp_async_commit_group()"]
    Commit2 --> Wait["cute.arch.cp_async_wait_group(ab_stage - 2 or 0) & sync_threads()"]
    Wait --> LdSharedNext["Prefetch next stage SMEM to alternate registers: cute.autovec_copy (vectorized)"]
    LdSharedNext --> MMA["Tiled MMA on current stage registers: cute.gemm (128 threads)"]
    MMA --> RegAccum["Private Registers (accum)"]
    RegAccum --> Loop
    Loop --> FinalSync["sync_threads() after mainloop"]
    FinalSync --> WriteSmem["copy_atom_r2s: Write accum (F32) to smem_accum"]
    WriteSmem --> EpilogueSync1["sync_threads()"]
    EpilogueSync1 --> EpilogueLoad["Load Float32 inputs from smem_accum into registers"]
    EpilogueLoad --> EpilogueSync2["sync_threads() to prevent RAW hazard on shared memory"]
    EpilogueSync2 --> EpiloguePrefetch["Load cos and sin from global memory inside the epilogue block (weights and scales are preloaded AOT)"]
    EpiloguePrefetch --> EpilogueMath["Apply RMSNorm, RoPE, and scaling in registers"]
    EpilogueMath --> EpilogueWrite["Write BFloat16 registers back to padded sOut_bf16 (stride=392)"]
    EpilogueWrite --> EpilogueSync3["sync_threads()"]
    EpilogueSync3 --> Store["Cooperatively Store (BF16) to Global Memory: Vectorized 128-bit Stores"]
```

#### Key Optimization Components
1. **Dynamic Pipelining (`ab_stage = 2 or 3`):** Dynamically allocates 3-stage triple-buffering memory if it stays below Blackwell's 99 KB SMEM limit, falling back to 2-stage double-buffering otherwise. Prefetch loop and mainloop adapt via modular arithmetic.
2. **Register-Level Double-Buffering & Compile-Time Range_Constexpr Unrolling:** Prefetches the next stage's shared memory data into an alternate set of register fragments (`rA_0`/`rB_0` or `rA_1`/`rB_1`) while performing MMA computations on the current stage's registers. To prevent GPU register spills to local memory due to dynamic loop indexing, the loop uses `cutlass.range_constexpr` and static index expressions, fully unrolling the loop and resolving register destinations at compile time.
3. **Explicit 128-bit Store Vectorization:** Configures epilogue global write copy atom with `num_bits_per_copy=128` and aligns tensor iterators to 16 bytes via `.align(16)`.
4. **2D Grid Swizzling:** Decodes block coordinates using an 8-block swizzling factor (`SWIZZLE_X = 8`) to run head groups and sequence tiles concurrently, maximizing L2 cache hits. Sequence dimensions are padded at launch and validated inside the kernel to prevent out-of-bounds access.


### 2. Unified CuTe DSL Fused Pipeline (Base)
```mermaid
graph TD
    X[Input Tensor X: B, N, H*D] --> Load["cp.async Load: Block_K_Tile=32, Block_M=64"]
    W[Weights W: Q, K, V] --> Load
    Load --> SharedMem["sX & sW (Single-buffered SMEM: 64 KB)"]
    SharedMem --> MMA["Tiled MMA (4 math warps: 128 threads, Grid Z = NUM_HEADS)"]
    MMA --> RegAccum["Private Registers (accum)"]
    RegAccum --> Epilogue["Vectorized RMSNorm -> RoPE -> Scale"]
    Epilogue --> Store["Global Memory Store: Q_proj, K_proj, V_proj"]
```

### 3. CuTe DSL Split-N Fused Pipeline (Cooperative Load)
```mermaid
graph TD
    X[Input Tensor X: B, N, H*D] --> Load["Threads 0-127: cp.async Load (128 threads)"]
    W[Weights W: Q, K, V] --> Load
    Load --> SharedMem["sX (4 KB) & sW (12 KB)"]
    SharedMem --> MMA_Q["WG 0 (128 threads): Tiled MMA for Q"]
    SharedMem --> MMA_K["WG 1 (128 threads): Tiled MMA for K"]
    SharedMem --> MMA_V["WG 2 (128 threads): Tiled MMA for V"]
    MMA_Q --> RegAccum_Q["Registers (accum_q)"]
    MMA_K --> RegAccum_K["Registers (accum_k)"]
    MMA_V --> RegAccum_V["Registers (accum_v)"]
    RegAccum_Q --> sBuf["sBuf (48 KB)"]
    RegAccum_K --> sBuf
    RegAccum_V --> sBuf
    sBuf --> Epilogue["Vectorized RMSNorm -> RoPE -> Scale (Threads 0-63)"]
    Epilogue --> Store["Global Memory Store: Q_proj, K_proj, V_proj"]
```

### 4. Standalone Triton Fused Epilogue Pipeline
```mermaid
graph TD
    X[Input Tensor X: B, N, hidden] --> Linear["F.linear (Tensor Cores)"]
    Linear --> QKV["QKV Projection Tensor (shape: B, N, 3*H*D)"]
    QKV --> TritonEpilogue["Triton Kernel: qknorm_rope_scale (Grid: B, NUM_HEADS, N)"]
    TritonEpilogue --> LoadTriton["Coalesced Loads of Q and K chunks"]
    LoadTriton --> NormTriton["Triton RMSNorm on Q and K"]
    NormTriton --> RoPETriton["Triton RoPE Rotation"]
    RoPETriton --> ScaleTriton["Scale Q by exp(scales)"]
    ScaleTriton --> StoreTriton["Store to output tensors Q_proj, K_proj (V_proj is sliced/permuted directly in PyTorch)"]
```

### 5. Pipelined cp.async GEMM Pipeline (2-Stage Double-Buffered)
```mermaid
graph TD
    X[Input Tensor X: B*N, hidden] --> Prefetch["cp.async Prefetch Stage 0: 128 threads (BLOCK_K=64, BLOCK_M=128, BLOCK_N=128)"]
    W[Weight: out_features, hidden] --> Prefetch
    Prefetch --> Commit1["cute.arch.cp_async_commit_group()"]
    Commit1 --> Loop["Mainloop: k_tile = 0 to k_tile_cnt"]
    Loop --> cp_async["cp.async Prefetch Next Stage (k_tile + 1)"]
    cp_async --> Commit2["cute.arch.cp_async_commit_group()"]
    Commit2 --> Wait["cute.arch.cp_async_wait_group(1 or 0) & sync_threads()"]
    Wait --> LdShared["Direct Shared-to-Register Loads: rA[i] = tCsA[i] (128 threads)"]
    LdShared --> MMA["Tiled MMA (4 warps: 128 threads)"]
    MMA --> Accum["Float32 Accumulators (accum: 64 regs/thread)"]
    Accum --> SyncAdvance["sync_threads() & Advance read/write stages"]
    SyncAdvance --> Loop
    Loop --> FinalSync["sync_threads() after mainloop"]
    FinalSync --> StoreSmem["Downcast & Element-wise Store: accum to sC_epi (aliased to sX)"]
    StoreSmem --> EpilogueSync["sync_threads()"]
    EpilogueSync --> Store["Direct Coalesced Vector Writes to GMEM"]
### 6. Fused QKV Epilogue Pipeline (Pure GEMM Writeback)
```mermaid
graph TD
    Accum["GEMM Accumulators (q_reg, k_reg, v_reg in RMEM)"] --> Sync1["sync_threads() after accum write to smem_accum"]
    Sync1 --> LoadRegs["Load Q, K, V into registers, cast to BFloat16"]
    LoadRegs --> WriteSmem["Write Q, K, V to sOut_bf16 (SMEM) in coalesced format"]
    WriteSmem --> Sync2["sync_threads() after writeback completion"]
    Sync2 --> Store["gmem_tiled_copy_store Q, K, V to Global Memory in parallel"]
```

#### Key Design Choices
1. **Thread Realignment via SMEM Round-Trip:** The Tensor Core MMA layout (`tCsC`) splits head dimensions across threads. Performing a brief shared memory round-trip aligns all elements of `HEAD_DIM = 64` to the same thread, ensuring coalesced vector writeback.
2. **Synchronized Execution Flow:** The epilogue uses exactly two synchronization barriers (`sync_threads()`) to safely stage the Float32-to-BFloat16 recast and organize the data layout before writing back to global memory.
3. **Zero-Cost Weight Permutation:** Fuses the Q, K, V head-group weight permutation directly into the global memory copy layout (`gW_layout`) in CuTe. It reads the unpermuted weight matrix of shape `[3 * NUM_HEADS * HEAD_DIM, hidden]` by defining a hierarchical shape `((HEAD_DIM, 3, HEAD_GROUP), BLOCK_K_STAGE)` with strides `((hidden, NUM_HEADS * HEAD_DIM * hidden, HEAD_DIM * hidden), 1)`, loading the weights directly into the thread-block's shared memory layout at zero runtime cost.

### 7. Standalone In-Place Split-QK Fused Epilogue Pipeline
```mermaid
graph TD
    Q[Input Tensor Q from GMEM: B, H, N, D] --> Load["cp.async Load Stage 0: Block_M=64, Block_N=HEAD_GROUP*HEAD_DIM (128 threads)"]
    K[Input Tensor K from GMEM: B, H, N, D] --> Load
    Load --> SharedMem["sQ, sK (Double-buffered stages: sQ_size + sK_size)"]
    SharedMem --> LoadStage1["cp.async Load Stage 1 (if ab_stage == 2) with sequence clamped offsets"]
    LoadStage1 --> Wait0["sync_threads() after cp.async wait stage 0"]
    Wait0 --> LdShared0["autovec_copy stage 0 SMEM to BFloat16 registers: q_reg, k_reg"]
    SinCos[Precomputed Sin/Cos from GMEM: B, 1, N, D] --> LdSinCos["Direct GMEM load to registers: sin_reg, cos_reg"]
    LdShared0 --> RMSNorm0["Compute Float32 RMSNorm on stage 0 q_reg and k_reg"]
    RMSNorm0 --> RoPE0["Apply precomputed sin_reg and cos_reg, rotate q_reg & k_reg"]
    RoPE0 --> Scale0["Apply Q scale-factor in Float32 to q_reg"]
    Scale0 --> Store0["Vectorized 128-bit stores (STG.128) back to GMEM Q and K in-place"]
    Store0 --> Wait1["sync_threads() after cp.async wait stage 1"]
    Wait1 --> Compute1["Compute and writeback stage 1 elements (if ab_stage == 2)"]
```

#### Key Design Choices
1. **Double-Buffered Cooperative Pipelining (`ab_stage = 2`):** Supports stage-level double-buffering to overlap global memory load latency of inputs with epilogue math and register execution. Thread block tiles of size `BLOCK_M` are partitioned into sub-tiles of size `SUB_M = BLOCK_M // ab_stage`.
2. **Out-of-Bounds Sequence Clamping:** During double-buffered preloading of stage 1 (`stage_idx = 1`), sequence blocks near the end of the sequence length `N` (or invalid blocks that are swizzle-padded) clamp their global memory loading offsets to `N - 1` using `cutlass.min(stage_seq_start, N - 1)`. This ensures that global memory loads via `cp.async` read from valid memory addresses and completely avoids illegal memory accesses.
3. **Single-Stage Loopless CTA Mapping:** Bypasses sequence-dimension loops inside the block. Grid swizzling maps the sequence tile directly to block coordinates, so each block runs a single/double stage of asynchronous loading and processing.
4. **AOT Precomputed Register-Direct Loads:** `sin` and `cos` tensors are precomputed in PyTorch outside the kernel with shape `[..., dhead // 2]` and loaded directly to registers via vectorized `cute.autovec_copy`, bypassing shared memory entirely. This completely eliminates dynamic trigonometric math from the GPU execution pipeline and releases the 8 KB freqs shared memory buffer.
5. **Dynamic Autotuned Execution Parameters:** Rather than using hardcoded threadblock tile dimensions, the kernel shape (`BLOCK_M`, `HEAD_GROUP`), block size (`num_warps`), and compiler register bounds (`maxrregcount`) are dynamically selected by a multi-dimensional grid-search autotuner, compiling custom specialized JIT kernels for the specific hardware/problem sizes.
6. **Direct-Store In-Place Writeback:** Bypasses intermediate shared memory writeback buffers and block barriers, writing outputs directly from BFloat16 registers back to the original `q` and `k` GMEM addresses using coalesced vectorized 128-bit store instructions (`STG.128`).
7. **Loop-Unrolling Alignment & Warning Resolution:** Uses hardcoded branches for `HEAD_GROUP` (e.g. `HEAD_GROUP == 1` vs `HEAD_GROUP == 2`) to let the compiler statically track and propagate 128-bit memory alignment. Uses `cutlass.range(..., unroll_full=True)` for non-register loops to silence slow trace-time compilation warnings.
8. **LDS Bank Conflict Padding:** Padded layout strides from `BLOCK_N = 128` to `BLOCK_N_padded = 136` (shifting memory columns) to ensure conflict-free LDS read/write accesses.
9. **Padding Block Execution Guard:** Guarded asynchronous loads inside the kernel using `if is_valid` block-level branches to completely skip global memory transactions and redundant math for grid padding blocks.
10. **Thread Occupancy Constraints:** Restricts the candidate configurations in `autotune.py` to configurations where `num_warps * 32 >= (BLOCK_M // ab_stage) * HEAD_GROUP` (for forward pass) and `num_warps * 32 >= BLOCK_M * HEAD_GROUP` (for backward pass). This ensures there is a 1-to-1 mapping of threads to sequence/head group elements so that no elements remain uncomputed in the block.

### 8. Standalone In-Place Split-QK Fused Epilogue Backward Pipeline
```mermaid
graph TD
    gQ[Original Q: B, H, N, D] --> Load["Vectorized GMEM loads directly to Registers (no SMEM)"]
    gK[Original K: B, H, N, D] --> Load
    gdQ_out[Incoming grad_q_out: B, H, N, D] --> Load
    gdK_out[Incoming grad_k_out: B, H, N, D] --> Load
    sin[GMEM Sin: B, 1, N, D // 2] --> Load
    cos[GMEM Cos: B, 1, N, D // 2] --> Load
    Load --> RMSNorm["Recompute Q/K RMSNorm stats on-the-fly in registers"]
    RMSNorm --> InvRoPE["Apply inverse RoPE rotation to incoming grad_q_out/grad_k_out in registers"]
    InvRoPE --> ParamGrads["Compute local gradients for weights (dw_rmsnorm) and scales (ds_scale) in registers"]
    ParamGrads --> SMEMReduce["Accumulate conflict-free in SMEM & write back cooperatively via atomic_add to GMEM"]
    InvRoPE --> InputGrads["Compute input gradients grad_q/grad_k using RMSNorm backward projection"]
    InputGrads --> Store["Vectorized 128-bit store of grad_q/grad_k back to GMEM"]
```

#### Key Design Choices
1. **Conflict-Free Shared Memory Reduction:** Allocates a tiny shared memory footprint for local block reductions. Threads write gradients to unique block slots, avoiding shared memory bank conflicts and serialization. The block then cooperatively sums the gradients along the sequence dimension, reducing global atomic additions by 64x.
2. **On-the-fly Recomputation:** Recomputes RMSNorm statistics ($rsqrt_q$, $rsqrt_k$) and inverse RoPE rotations locally in registers from the original $q$ and $k$ inputs, avoiding the need to store intermediate forward values in VRAM.
3. **Vectorized 128-bit LDS Shared Memory Reductions:** Contiguous groups of 4 Float32 gradient elements are loaded from shared memory reduction buffers in parallel using 128-bit vectorized Load Shared (`LDS.128`) instructions. Loop structures are unrolled at compile time using `cutlass.range_constexpr(...)` combined with manual 4-element unrolling to prevent JIT compiler type tracking errors.
4. **Cooperative Parallel Tree Reduction for Scales Gradient:** Replaced sequential sequence-level loops (which ran on only a single thread per head group) with a parallel shared memory tree reduction unrolled at compile time. Threads cooperatively sum scales gradient values in $\log_2(\text{BLOCK\_M})$ steps, distributing the compute/memory load across all threads in the CTA to maximize warp utilization.

---


## Shared Memory Map

### 1. Milestone 1 Fused QKV Kernel Map
- **Total Active Shared Memory:** **Up to 96 KB** (reused/aliased during execution and dynamically sized depending on autotuned configurations)
- **Allocation Details (for optimal config HEAD_GROUP = 2, BLOCK_M = 64, BLOCK_N = 384, BLOCK_K = 32):**
  - `sX` (BFloat16, 2-stage staged layout): `BLOCK_M * BLOCK_K_STAGE * 2` = `64 * 32 * 2 = 4,096` elements (**8 KB**).
  - `sW` (BFloat16, 2-stage staged layout): `BLOCK_N * BLOCK_K_STAGE * 2` = `384 * 32 * 2 = 24,576` elements (**48 KB**).
  - `sBuf` (Float32, epilogue writeback buffer): `BLOCK_M * self.BLOCK_N_padded` = `64 * 392 = 25,088` elements (**98 KB**).
  - Peak SMEM utilization is exactly **98 KB** (reusing a single flat Float32 storage block of 25,088 elements via manual offset mapping, fitting under Blackwell's 99 KB SMEM limit and allowing maximum occupancy).

### 2. Base CuTe DSL Kernel Map
- **Total Active Shared Memory:** **64 KB**
- **Allocation Details:**
  - `sX` (BFloat16, single-buffered): `BLOCK_M * BLOCK_K_TILE` = `64 * 32 = 2048` elements (**4 KB**).
  - `sW` (BFloat16, single-buffered): `(3 * HEAD_DIM) * BLOCK_K_TILE` = `192 * 32 = 6100` elements (**12 KB**).
  - `sBuf` (Float32, single-buffered): `BLOCK_M * (3 * HEAD_DIM)` = `64 * 192 = 12288` elements (**48 KB**).

### 3. Split-N CuTe DSL Kernel Map
- **Total Active Shared Memory:** **80 KB** (within 99 KB SMEM limit)
- **Allocation Details:**
  - `sX` (BFloat16, 2-stage staged layout): `BLOCK_M * BLOCK_K_TILE * 2` = `64 * 32 * 2 = 4,096` elements (**8 KB**).
  - `sW_q`, `sW_k`, `sW_v` (BFloat16, 2-stage staged layout): `3 * HEAD_DIM * BLOCK_K_TILE * 2` = `3 * 64 * 32 * 2 = 12,288` elements (**24 KB**).
  - `sBuf_q`, `sBuf_k`, `sBuf_v` (Float32, epilogue accumulation buffer): `3 * BLOCK_M * HEAD_DIM` = `3 * 64 * 64 = 12,288` elements (**48 KB**).

### 4. Pipelined cp.async GEMM Kernel Map (Blackwell Optimized)
- **Total Active Shared Memory:** **64 KB** (within 99 KB SMEM limit)
- **Allocation Details:**
  - `sX` (BFloat16, 2-stage staged layout): `BLOCK_M * BLOCK_K * 2` = `128 * 64 * 2 = 16,384` elements (**32 KB**).
  - `sW` (BFloat16, 2-stage staged layout): `BLOCK_N * BLOCK_K * 2` = `128 * 64 * 2 = 16,384` elements (**32 KB**).
  - `sC_epi` (BFloat16, epilogue buffer): Aliased directly to `sX` memory location using pointer casting, reusing the first 32 KB of the memory chunk. (**0 KB** extra overhead).

### 5. Fused QKV Kernel Map
- **Total Active Shared Memory:** **98 KB** (fits under the 99 KB limit) for the optimal config `HEAD_GROUP = 2, BLOCK_M = 64, BLOCK_K = 16, ab_stage = 3`.
- **Allocation Details:**
  - `sX` (BFloat16, 3 stages): `BLOCK_M * BLOCK_K * 3` = `64 * 16 * 3 = 3,072` elements (**6 KB**).
  - `sW` (BFloat16, 3 stages): `BLOCK_N * BLOCK_K * 3` = `384 * 16 * 3 = 18,432` elements (**36.864 KB**).
  - `sBuf` (Float32, epilogue writeback buffer): `BLOCK_M * BLOCK_N_padded` = `64 * 392 = 25,088` elements (**98 KB**).
  - Peak SMEM utilization is exactly **98 KB** (by reusing the flat Float32 storage block of 25,088 elements for GEMM + epilogue writeback, with no frequencies buffer).

### 6. Standalone In-Place Fused Epilogue Kernel Map
- **Total Active Shared Memory:** **34 KB** (single-stage flat allocation for `HEAD_GROUP = 2`, with padding)
- **Allocation Details:**
  - `sQ` (BFloat16): `BLOCK_M * BLOCK_N_padded` = `64 * 136 = 8,704` elements (**17 KB**).
  - `sK` (BFloat16): `BLOCK_M * BLOCK_N_padded` = `64 * 136 = 8,704` elements (**17 KB**).
  - Peak SMEM utilization is exactly **34 KB** (by sharing a single flat Float32 block of 8,704 elements via recast pointer offsets for staging, completely avoiding epilogue writeback buffer overhead, LDS bank conflicts, and frequencies shared buffer footprint).
  - Note: The shared memory footprint scales dynamically with the autotuned configuration. For the autotuned optimal shape (`HEAD_GROUP = 1, BLOCK_M = 32`), the footprint is reduced to only **9.2 KB** (sQ + sK = `2 * 32 * 72 * 2` bytes), saving 73% of dynamic shared memory compared to the baseline.

### 7. Standalone Fused Epilogue Backward Kernel Map
- **Total Active Shared Memory:** **~16.1 KB** (for autotuned shape `BLOCK_M = 32, HEAD_GROUP = 1`) or **~32.2 KB** (for `BLOCK_M = 64, HEAD_GROUP = 1`)
- **Allocation Details:**
  - `s_reduction` (Float32): `BLOCK_M * 2 * HEAD_GROUP * HEAD_DIM` elements (**16 KB** for `BLOCK_M = 32, HEAD_GROUP = 1`).
  - `s_scales_reduction` (Float32): `BLOCK_M * HEAD_GROUP` elements (**128 bytes**).
  - Peak SMEM utilization is exactly **16.1 KB** or **32.2 KB** (depending on autotuned sequence tile size). Threads write to disjoint elements of shared memory to avoid bank conflicts and serialization. A cooperative block reduction sums the gradients along the sequence dimension `BLOCK_M` before writing back to global memory, reducing global atomic writes by 64x.

### 8. Blackwell TMA Pipelined Backward GEMM dx Kernel Map (`bwd_dx`)
- **Total Active Shared Memory:** **~68 KB** (well within 99 KB Blackwell limit)
- **Allocation Details (for optimal config BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32, ab_stage = 3):**
  - `sX` (BFloat16, 3 stages): `BLOCK_M * BLOCK_K_STAGE * 3` = `128 * 32 * 3 = 12,288` elements (**24.576 KB**).
  - `sW` (BFloat16, 3 stages): `BLOCK_N * BLOCK_K_STAGE * 3` = `128 * 32 * 3 = 12,288` elements (**24.576 KB**).
  - `smem_accum` (Float32, epilogue buffer): `BLOCK_M * BLOCK_N_padded` = `128 * 136 = 17,408` elements (**69.632 KB**).
  - Peak SMEM utilization is exactly **~68 KB** by aliasing the BFloat16 output buffer `sOut_bf16` with the `smem_accum` buffer, reducing epilogue SMEM requirements from 102 KB (violating the 99 KB limit) to 69.6 KB.

### 9. Blackwell TMA Pipelined Backward GEMM dw Kernel Map (`bwd_dw`)
- **Total Active Shared Memory:** **~42.0 KB** (within 99 KB Blackwell limit)
- **Allocation Details (for optimal config BLOCK_M = 32, BLOCK_N = 128, BLOCK_K = 32, ab_stage = 3, SPLIT_K = 2):**
  - `sX` (BFloat16, 3 stages, 3 components): `BLOCK_M * BLOCK_K_STAGE * 3 * 3` = `32 * 32 * 3 * 3 = 9,216` elements (**18.432 KB**).
  - `sW` (BFloat16, 3 stages): `BLOCK_N * BLOCK_K_STAGE * 3` = `128 * 32 * 3 = 12,288` elements (**24.576 KB**).
  - `smem_accum` (BFloat16, epilogue buffer, 3 components): `3 * BLOCK_M * BLOCK_N_padded` = `3 * 32 * 136 = 13,056` elements (**26.112 KB**).
  - Peak SMEM utilization is exactly **43.008 KB** (since epilogue memory is aliased with mainloop staged buffers).

### 10. Thread Block Clusters for L2 Cache Saturation
- **Cluster Shape:** `cluster=[2, 1, 1]`
- **Blackwell Workaround:** On GeForce Blackwell (SM120), hardware TMA multicast is not supported. To work around this, we launch both [matmul_bwd_dx](file:///home/achiya/vs_code_main/flash_attn_prologue/flash_attn_prologue/gemm_bwd.py) and [matmul_bwd_dw](file:///home/achiya/vs_code_main/flash_attn_prologue/flash_attn_prologue/gemm_bwd.py) using a Thread Block Cluster of size 2 along the X grid dimension.
- **Scheduling and Memory Flow:** The cluster configuration instructs the GPU scheduler to map sibling CTAs onto physically adjacent SMs. When the first CTA issues a standard TMA load to bring a tile of `weight` (for `bwd_dx`) or `x_flat` (for `bwd_dw`) from global memory, the tile is cached in L2. Sibling CTAs requesting the same tile will hit the L2 cache instead of hitting global memory, effectively neutralizing the GMEM read bottleneck by up to a factor of 2x.

### 11. On-Chip JIT Split-K Reduction Kernel Map
- **Layout & Execution Flow:** For Split-K > 1, the `bwd_dw` JIT kernel writes intermediate Float32 sums to a global workspace tensor of shape `[SPLIT_K, M_dim, N_dim]`. Instead of running PyTorch eager `.sum(dim=0)`, we call a specialized, JIT-compiled CuTe reduction kernel `SplitKReductionCute` on the active stream.
- **Thread Block Tiling:** The reduction kernel tiles the workspace along M and N dimensions using blocks of `16x64`. With `num_threads = 256`, the loop unrolls 4 times per thread block to cover the 1024 elements in the tile. Threads load Float32 workspace slices directly from global memory using raw pointer subscripting, sum along the Z (split-K) dimension in registers, and store the reduced result directly to global memory as BFloat16, bypassing any PyTorch API or memory management overhead.

### 12. Fused Inverse Epilogue Operator Map
- **Total Active Shared Memory:**
  - **Forward Pass:** Matches the standalone forward Fused Epilogue kernel (`9.2 KB` to `34 KB` depending on the autotuned block size).
  - **Backward Pass:** Matches the standalone backward Fused Epilogue reduction buffers (`16 KB` to `32 KB` depending on the autotuned sequence tile size), used for conflict-free gradient tree accumulation.
- **Key Design Choices:**
  - Reconstructs intermediate activations (`q_normed`, `k_normed`) on-chip in registers during the backward pass to eliminate global memory storage of forward inputs.
  - Avoids saving full Q and K inputs, retaining only the low-overhead Float32 sequence-level RMS scale factors (`rms_q`, `rms_k`) of shape `(B, H, N, 1)`.

### 13. Package Editable Installation & FFI Auto-Discovery
- **Editable Registration**: The package is registered as an editable install inside the Python environment via [setup.py](file:///home/achiya/vs_code_main/flash_attn_prologue/setup.py). This inserts a path link inside `.venv`'s `site-packages`, resolving import paths from any directory without requiring `PYTHONPATH=.`.
- **Automatic FFI Discovery Hook**: The entrypoint [__init__.py](file:///home/achiya/vs_code_main/flash_attn_prologue/flash_attn_prologue/__init__.py) executes discovery routines to locate `libcute_dsl_runtime.so` within the active virtual environment structure (`sys.path` and `sys.prefix`), exporting it to `os.environ["CUTE_DSL_LIBS"]` automatically. This satisfies the JIT compiler's environment manager requirements without requiring manual configuration.

### 14. Standalone RMSNormRoPEScale Module Layer
- **Interface & Execution Flow**: We implement [RMSNormRoPEScale](file:///home/achiya/vs_code_main/flash_attn_prologue/flash_attn_prologue/layers/RMSNorm.py#L136) as a standard PyTorch `nn.Module` layer inside the `flash_attn_prologue.layers` namespace. It accepts two separate (split) Query (`q`) and Key (`k`) tensors, applies on-the-fly RMSNorm, RoPE, and query scaling, and returns the modified `q` and `k` tensors.
- **Dtype Management**: Weight parameters `rmsnorm_weight` and `scales` are initialized and stored as `BFloat16` parameters, saving memory and matching standard bfloat16 models. The PyTorch custom operator wrapper handles dynamic upcasting to Float32 on compilation or FFI invocation boundaries automatically, ensuring full correctness and precision.
- **torch.compile Compatibility**: The module is fully traced and optimized by PyTorch Inductor. Its compatibility is validated with `fullgraph=True`, ensuring no Python-to-C++ or custom operator boundary graph breaks in both the forward execution and autograd backward paths.
- **Layout Shape-Stride Specialization Alignment**: To guarantee that layouts can be specialized for any custom permuted strides (e.g. for non-contiguous views from unbind/split slices) without layout structure mismatches, both shapes and strides are dynamically cast to concrete Python integers in the operator wrappers. This allows the TVM/CuTe JIT compilation pipeline to construct layout types with consistent compile-time concrete shape dimensions and stride dimensions.
- **Inductor Compact Stride Policy**: To prevent stride mismatch assertions during layout propagation under PyTorch Inductor, the custom operator outputs are allocated with compact strides matching the contiguous-by-permutation order of the inputs. Any inputs with non-contiguous strides are copied to compact tensors on FFI boundaries (specifically in the backward pass) so that the JIT compiled layouts and actual runtime layouts are perfectly matched.
- **Stride-Based Batch Broadcasting of RoPE Buffers**: When `sin`/`cos` buffers are shared across the batch (batch dimension = 1), their batch strides are forced to `0` in the FFI kernel builders. This prevents pointer indexing out of bounds for batch indices `cta_b >= 1` (where the original batch stride would point to uninitialized memory) and correctly broadcasts the positional embeddings.
- **No Sequence Padding Requirement**: The sequences in the data pipeline do not have padding, so the redundant sequence padding active-mask logic (`active_mask`) has been removed from the backward kernels, eliminating extra zeroing overhead.

### 15. Temporal Mamba 3 FWD/BWD CuTe Kernels
- **Grid Mapping & Thread Block Tiling**:
  - Grid: `[shape_S, shape_H, shape_B]`. Each thread block maps to a single sequence token index $s$, attention head $h$, and batch element $b$.
  - Tiling dimensions: State Dimension ($N$) is $64$, Head Dimension ($P$) is $64$, MIMO Rank ($R$) is $4$.
- **Shared Memory Allocations**:
  - `sState` (Float32, double-buffered): `2 * num_warps * N * 16` elements (**32 KB** for 4 warps, $N=64$).
  - `sTrap` (Float32, double-buffered): `2 * num_warps * N * 16` elements (**32 KB**).
  - `sC` (Float32): `num_warps * 8 * N` elements (**8 KB**). Exposes coprime strides `stride=(10 * N, 1, 10)` to eliminate bank conflicts.
  - `sC_out` (Float32): `num_warps * 16 * 8` elements (**2 KB**). Exposes coprime strides `stride=(160, 1, 20)`.
  - In `CuteMamba3Bwd`, additional gradient buffers `sGradState` and `sGradV` are allocated to track recurrent scan backpropagation with similar double-buffered SMEM footprints.
- **On-Chip Execution Flow**:
  - **Asynchronous Double-Buffering**: Uses `cp.async` to prefetch subsequent sequence states from global memory (`ssm_state`, `trap_mem`) into shared memory, overlapping read latency with Tensor Core matrix math.
  - **Register-Level Fusion**: The thread block loads weights (`W_X`, `W_Z`, `W_O`) and skip scale parameters into register fragments during initialization. Skip-connections, SiLU gating, and MIMO rank-4 sums are computed entirely on-chip within register files, bypassing shared memory writebacks.
  - **Native BFloat16 Output & Eager Cast Elimination**: The output tensor `out` is allocated and written directly as `BFloat16` inside the JIT kernel using `.align(8)` for 128-bit stores (since 4 elements of `BFloat16` occupy 8 bytes), while the states remain strictly `Float32`. Eager-level `.to(...)` castings are eliminated from the custom operator and wrapper returns, avoiding autograd node tracking and copy overhead.
  - **BFloat16 Alignment Constraint**: Employs `.align(8)` alignment for BFloat16 output pointer castings (`out_flat`), avoiding address rounding/collision bugs during vectorized 128-bit global writes.
  - **Gradient Coalescing**: Parametric gradients (`dW_X`, `dW_Z`, `dW_O`, `dD_skip`) are accumulated locally in registers and written back to global memory via atomic additions exactly once per thread block at execution termination. Sequence-dependent gradients are stored directly without serialization.
  - **Decoupled Workload-Aware Autotuning**: Uses a hybrid search strategy to tune hyper-parameters (`num_warps`, `TILE_S`, `buf_size`, `layout_option`, `maxrregcount`, `copy_layout_shape`, `mma_shape`) dynamically based on shape and workload scale. To account for asymmetric register pressure and memory access patterns, FWD and BWD configuration paths are profiled and cached independently under suffix-split keys (e.g. `64_64_4_large_fwd` and `64_64_4_large_bwd`) in `autotune_cache.json`.
  - **Hybrid Search Strategy**:
    1. **Wide Grid Search**: Sweeps over 16-20 coarse candidate configurations varying structural parameters (`num_warps`, `TILE_S`, `buf_size`, `layout_option`) to find the best macro architecture.
    2. **Deep Coordinate Descent**: Refines the top-performing grid configuration using coordinate descent coordinate sweeps over fine-grained parameters, including an expanded register budget candidate set (`maxrregcount` in `[80, 128, 168, 192, 255]`). Prepopulates the evaluation cache from the grid search results to eliminate compile/profile overhead.
  - **Performance Parity & Benchmarks**: On the GeForce RTX 5070 Ti, the specialized CuTe kernels achieve a **2.08x speedup** on $N=16, R=2$ and a **1.70x speedup** on $N=32, R=4$ at X-Large scale ($B=64, S=128$) compared to PyTorch Inductor compiled execution, while saving **27% to 40% peak net activation VRAM** consistently across shapes.
  - **Compilation-Friendly Dispatch Architecture**: To support compilation under `torch.compile(fullgraph=True)` without graph breaks, all dynamic autotuning queries (which perform file IO to read `autotune_cache.json`) and shape checks are relocated from the plain Python wrappers into the registered PyTorch custom operators (`_cute_mamba3_custom_op` and `_cute_mamba3_custom_op_backward`). During tracing, Dynamo only traces the custom operators' fake/meta tensor implementations, bypassing the autotuner completely. At runtime, the custom operators perform the dispatch dynamically, executing either the optimized CuTe JIT kernel or falling back to the eager PyTorch implementation (`cute_mamba3_fallback` / `cute_mamba3_backward_fallback`).
  - **Parameter Data-Type Architecture**: Inner weight parameters (`W_X`, `W_Z`, `W_O`, `D_skip`) are maintained in FP32 for recurrent state space numerical stability. Aligning with the standard calling convention (running `.bfloat16()` followed by `.reset_parameters()`), all inner parameters are reset back to Float32 dynamically, allowing the native eager branch in `mamba3.py` to reference them directly without redundant casting overhead.
  - **Two-Pass Parameter Gradient Workspace Reduction**: Replaced direct global atomic additions (`cute.arch.atomic_add`) for parameter gradients `dW_X`, `dW_Z`, `dW_O`, and `dD_skip` in `CuteMamba3Bwd.kernel`. Instead of competing for global write locks, active threadblocks write their locally accumulated gradients directly to disjoint slices of a pre-allocated Float32 global workspace tensor of shape `(num_warps_total, H, P, R)` for weights, and `(num_warps_total, H, P)` for `D_skip` parameter. A secondary eager-level reduction pass (`.sum(dim=0)`) sums the slices to write back the final gradients, completely eliminating DRAM/L2 cache controller serialization.
  - **Compile-Time AST Loop Unrolling over Rank ($R$)**: Rank loops iterating over $R=4$ are unrolled at compile time using `cutlass.range(self.R, unroll_full=True)` in both forward and backward scan kernels. This ensures that the NVCC JIT compiler compiles these loops into fully unrolled PTX instructions, enabling constant folding and register-renaming.
  - **Direct-to-Global Epilogue Store for States**: Staged buffers in shared memory (`smem_tile_state` and `smem_tile_trap` of size 64 KB) are completely bypassed in the backward pass. The active registers `dssm_state_local` and `dtrap_mem_local` are written directly to global memory. This reduces the block SMEM footprint to under 8 KB, boosting SM block occupancy on Blackwell SM120.
  - **On-Chip 3D-to-4D Logical Layout Mapping**: To avoid wrapper-level PyTorch `.view()` or `.reshape()` operations that incur compilation tracing overhead, the forward and backward kernels accept 3D tensors of shape `[B, S, H * P]` and logically map them to 4D views of shape `[B, S, H, P]` using CuTe's native layout mapping stride arithmetic. The custom operator interfaces receive and return 3D tensors directly, aligning the calling boundary with the physical memory layout of the model.

### 16. Dynamic Evaluation RoPE Theta Alignment
- **Dynamic Batch Sizing**: During the evaluation phase (`eval_one_judge_epoch`), validation batch sizes can vary (e.g., the last batch of the epoch). Generating standard RoPE phase arrays using a static `loader.batch_size` outside the loop caused mismatched dimensions inside the attention layers.
- **Dynamic Resolution**: Moved the call to `generate_all_thetas` inside the validation loop, querying the batch size dynamically from `batch_data['images'].shape[1]` at each step. This enforces strict dimension matching across all evaluation batch boundaries and eliminates out-of-bounds page faults or shape crashes, maintaining identical logic to the training pipeline.



