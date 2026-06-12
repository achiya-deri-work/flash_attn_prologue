# Temporal Mamba 3 CuTe/Triton Kernel Specification

This document details the hardware-specialized design, precision guidelines, execution pipelines, and autotuning parameters for the optimized Mamba 3 recurrent update kernel on SM120 (Blackwell) and SM80 (Ampere) architectures.

---

## 1. Kernel Scope & Thread Block Mapping

### Scope
The kernel is isolated strictly to the following block:
* **Inputs:** `x_raw` (FP32), `z_raw` (FP32), `B_rot` (FP32, pre-rotated input state projection), `C_rot` (FP32, pre-rotated output state projection), `dt` (FP32), `A` (FP32), `lam` (FP32), `ssm_state` (FP32), `trap_mem` (FP32), and weight parameters ($W_X$, $W_Z$, $W_O$, $D\_skip$ in FP32).
* **Outputs:** `out_flat` (BFloat16), `curr_state` (FP32), `curr_V` (FP32).

### Grid/Block Tiling
* **Parallel Grid:** $B \times S \times NH = 32 \times 64 \times 4 = 8,192$ independent problems.
* **Block Mapping:** Exactly **one thread block** (or warp group) is assigned to a single $(b, s, h)$ task.
* **Internal Tiling Dimensions:**
  * State Dimension ($N$): $64$
  * Head Dimension ($P$): $64$
  * MIMO Rank ($R$): $4$

---

## 2. Precision and Datatype Flow

Maintaining the state space in FP32 is critical for long-horizon stability. To balance accuracy and execution throughput, the data types flow as follows:

| Stage / Operation | Inputs | Accumulator / Math | Outputs | Hardware Unit |
| :--- | :--- | :--- | :--- | :--- |
| **GEMM 1** ($B_{rot} \cdot W_X^T$) | FP32 | FP32 | FP32 (`curr_V_temp`) | Tensor Cores (TF32) |
| **State Update** ($curr\_state$) | FP32 | FP32 | FP32 (`curr_state`) | Vector ALUs (FP32) |
| **GEMM 2** ($curr\_state^T \cdot C_{rot}$) | BF16 *(Cast)* | FP32 | BF16 (`Y_t`) | Tensor Cores (BF16) |
| **Skip & Gate Path** | FP32 / BF16 | FP32 | FP32 (`out_gated_prime`) | Vector ALUs (FP32) |
| **Output Write** | FP32 | — | BF16 (`out_flat`) | TMA Store (BF16) |

> [!IMPORTANT]
> The gating math, SiLU activation, and rank-4 reduction are computed in **FP32** to prevent swamping errors and numerical instability. They are cast to BFloat16 only at the final global memory write boundary.

---

## 3. SM120 (GeForce Blackwell) Hardware Execution Constraints

Since this codebase targets the SM120 architecture (e.g. RTX 5090), the kernel must adhere strictly to its hardware characteristics:
* **No WGMMA (Warp Group MMA):** SM120 does not support reading operands directly from Shared Memory during matrix multiplications. We must explicitly copy inputs from Shared Memory (SMEM) to Registers (RMEM) using `ldmatrix` before each warp-level MMA instruction.
* **Warp-level MMA:** Matrix multiplications must utilize `MmaF16BF16Op` (16x8x16, warp-level) for BFloat16 math, and the corresponding warp-level operation for TF32.
* **No TMA Multicast:** SM120 lacks the multicast hardware feature present in SM90. GEMMs must configure `cluster_shape = (1,1,1)`.
* **SMEM Budget:** Shared memory is strictly limited to **99 KB** per block.

---

## 4. Execution Overlapping Pipelines

### A. Instruction-Level Overlapping (ILP)
While the Tensor Cores are crunching `GEMM 1` (which has a latency of multiple clock cycles), the warp scheduler issues independent scalar calculations to the standard ALUs and SFUs:

$$\text{GEMM 1: } B_{rot} \cdot W_X^T \quad \text{[Tensor Cores (TF32)]}$$
$$\text{Overlapped with: } \begin{cases} dt \cdot A & \text{[ALU]} \\ \alpha = \exp(dt \cdot A) & \text{[SFU]} \\ \beta = (1 - \lambda) \cdot dt \cdot \alpha & \text{[ALU]} \\ \gamma = \lambda \cdot dt & \text{[ALU]} \end{cases}$$

This hides the entire latency of the decay parameter calculation, making it runtime-latency free.

### B. Warp Specialization
We divide the thread block's warps into two groups:
* **Producer Warps (127 registers):** Allocate via `setmaxnreg.dec.sync`. They manage TMA loads, TMA stores, and mbarrier signaling.
* **Consumer Warps (255 registers):** Allocate via `setmaxnreg.inc.sync`. They execute Tensor Core GEMMs and elementwise/reduction math in registers.

```
Timeline ────────────────────────────────────────────────────────────────────────►

[Producer Warps]  │ Load inputs (x, z, B_rot)  │ Prefetch ssm_state &      │ Store curr_state &        │ Store final output
                  │ & weights via TMA         │ trap_mem via TMA          │ curr_V via TMA            │ out_flat via TMA
                  └───────────────────────────┴───────────────────────────┴───────────────────────────┴───────────────────────────
                              │                             │                           ▲
                    (mbarrier signal)             (mbarrier signal)             (TMA write-back)
                              ▼                             ▼                           │
[Consumer Warps]  │ Wait for inputs           │ GEMM 1: B_rot * W_X_T (TF32)│ Update curr_state (FP32) │ GEMM 2: curr_state.T * C_rot (BF16)
                  │                           │ & Compute curr_V (FP32)   │                          │ & Compute Gating/Skip (FP32)
```

---

## 5. Layout & Register-Level Optimizations

### Register-Only Fusions (No SMEM for $64 \times 4$ Intermediates)
Because the rank $R = 4$ is extremely narrow, the thread block does not write any 2D intermediate matrices ($DX\_W$, $Z_t$, $Y_{skip}$, $out\_gated$) to shared memory.
For each head dimension $p \in [0, 63]$ assigned to a thread, the thread computes the skip scaling, SiLU activation, and rank-4 sum reduction on the fly inside registers:
$$out\_gated\_prime[p] = \sum_{r=0}^3 \left( Y_t[p, r] + x[p] \cdot D\_skip[p] \cdot W_X[p, r] \right) \cdot \text{silu}\left( z[p] \cdot W_Z[p, r] \right) \cdot W_O[p, r]$$

### Compile-Time Layout Algebra
All transpositions, unsqueezes, and reshapes compile to zero-cost address arithmetic in CuTe:
* **Transpositions:** Swap layout dimensions and strides at compile time (e.g., swapping strides of `W_X` to construct `W_X_T`, and swapping strides of `curr_state` to perform matrix multiplication).
* **Unsqueezes / Broadcasting:** Map the unsqueezed dimensions (such as batch/seq broadcast for weights, and singleton dimensions for vector multiplication) to stride `_0` (static zero).
* **Reshapes:** Flattening dimensions maps directly to multi-dimensional layouts mapped to physical memory strides.

---

## 6. Backward Pass Design Decisions (B, A, A Configuration)

* **Activation Recomputation (Option B):**
  We will recompute $Y_t$, $Z_t$, and $Y_{skip}$ on the fly in the backward pass to save global memory read/write bandwidth.
* **Weight Storage (Option A):**
  Store weights ($W_X, W_Z, W_O, D\_skip$) in Constant Memory (`__constant__`) to save Shared Memory and Register space, with a compile-time fallback to SMEM if head configurations scale.
* **Direct Global `atomicAdd` (Option A):**
  Local parameter gradients are accumulated using `atomicAdd` on global memory. To minimize collisions, we perform warp-level register coalescing first, so only a single thread per warp performs the global atomic instruction.

### Gradient Coalescing and Accumulation
* In the backward pass, the parameter gradients $d\_W_X$ and $d\_x$ receive multiple paths of contributions (skip path and recurrent path).
* We accumulate these contributions directly in registers:
  $$d\_W_X = d\_W_X^{skip} + d\_W_X^{rec}$$
  $$d\_x = d\_x^{skip} + d\_x^{rec}$$
  This avoids storing intermediate gradient matrices and executing redundant write/read instructions.

### Mathematical Reordering of Gradients
To reduce compute overhead, we factor out row-wise scaling before executing GEMMs:
* **Gradient $d\_W_X^{rec}$:**
  $$d\_W_X^{rec} = \gamma \cdot x \odot \left( d\_S_{curr}^T \cdot B_{rot} \right) \quad \text{(Saves ~19% FLOPs)}$$
* **Gradient $d\_B_{rot}$:**
  Let $W_{X, scaled} = x \odot W_X$.
  $$d\_B_{rot} = \gamma \cdot \left( d\_S_{curr} \cdot W_{X, scaled} \right) \quad \text{(Saves ~17.5% FLOPs)}$$

---

## 7. Autotuner Search Parameters

The coordinate-descent autotuner will optimize over the following search space:
1. **Block Tile Size:** `(BLOCK_M, BLOCK_N, BLOCK_K)` for GEMM 1 and GEMM 2.
2. **Instruction Arrangement:** `atom_layout_mnk` layout mapping for warp execution.
3. **Swizzle Patterns:** `cute::Swizzle<B, M, S>` configurations to eliminate shared memory bank conflicts.
4. **Pipeline Stages:** `ab_stage` (1, 2, or 3) for double-buffering.
5. **Register Budget:** `maxrregcount` (127 vs 255).
6. **Warp Splits:** Number of compute warps vs loading warps.
