# Copyright (c) 2026, Antigravity.
# Raw JIT-compiled backward GEMM kernels using CuTe Python DSL.

import os
import torch
import math

# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass import BFloat16, Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor
from quack import copy_utils

class GemmBwdDxCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K_STAGE: int = 32, ab_stage: int = 3, M_dim = None, K_dim: int = 3072, N_dim: int = 512):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K_STAGE = BLOCK_K_STAGE
        self.ab_stage = ab_stage
        self.M_dim = M_dim
        self.K_dim = K_dim
        self.N_dim = N_dim
        self.k_tile_cnt = K_dim // BLOCK_K_STAGE
        
    def _setup(self):
        # sX is row-major (since grad_output is row-major)
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(self.BLOCK_K_STAGE, 1, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        # sW is column-major (since weight is row-major, sliced along columns=BLOCK_N and rows=BLOCK_K_STAGE)
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_N, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        self.BLOCK_N_padded = self.BLOCK_N
        self.epi_smem_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N_padded, 1)
        )
        
        # Calculate shared memory footprint
        sX_bytes = self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 2
        sW_bytes = self.ab_stage * self.BLOCK_N * self.BLOCK_K_STAGE * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = self.BLOCK_M * self.BLOCK_N_padded * 4
        
        self.max_smem_bytes = max(mainloop_bytes, epilogue_bytes)
        self.flat_smem_elements = cute.ceil_div(self.max_smem_bytes, 4)
        self.sW_offset_elements = sX_bytes // 4
        
        @cute.struct
        class SharedStorageBwdDx:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageBwdDx
        
        # TiledMMA Configuration
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=(2, 2, 1))
        
        # Copy atom for Float32 accumulator writeback
        copy_op_r2s = cute.nvUniversalCopyOp() if hasattr(cute, "nvUniversalCopyOp") else cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, Float32)

    @cute.jit
    def __call__(
        self,
        grad_output: cute.Tensor,
        weight: cute.Tensor,
        grad_x: cute.Tensor,
        stream
    ):
        self._setup()
        
        M_dim = grad_output.shape[0]
        num_threads = self.num_warps * 32
        
        gA = cute.make_tensor(grad_output.iterator, cute.make_layout((M_dim, self.K_dim), stride=(self.K_dim, 1)))
        gB = cute.make_tensor(weight.iterator, cute.make_layout((self.K_dim, self.N_dim), stride=(self.N_dim, 1)))
        gO = cute.make_tensor(grad_x.iterator, cute.make_layout((M_dim, self.N_dim), stride=(self.N_dim, 1)))
        
        # Make TMA load/store descriptors
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA,
            cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)),
            (self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op,
            gO,
            self.epi_smem_layout,
            (self.BLOCK_M, self.BLOCK_N)
        )
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(M_dim, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            grad_output, weight, grad_x,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.b_smem_layout_staged_row,
            self.epi_smem_layout,
            self.copy_atom_r2s,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_o, tma_tensor_o,
            Boolean(True)
        ).launch(
            grid=[swizzled_grid_x, cute.ceil_div(self.N_dim, self.BLOCK_N), 1],
            block=[num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        grad_output: cute.Tensor,
        weight: cute.Tensor,
        grad_x: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        b_smem_layout_row: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s: cute.CopyAtom,
        tma_atom_a: cute.CopyAtom,
        tma_tensor_a: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        tma_tensor_b: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        use_sync_in_loop: Boolean
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        tidx, _, _ = cute.arch.thread_idx()
        
        # 2D Grid Swizzling with compile-time constant group size
        swizzle_x = 8
        grid_dim_y_const = self.N_dim // self.BLOCK_N
        group_size = swizzle_x * grid_dim_y_const
        
        pid = block_idx_y * grid_dim_x + block_idx_x
        swizzle_group = pid // group_size
        offset_in_group = pid % group_size
        
        cta_m_raw = swizzle_group * swizzle_x + (offset_in_group % swizzle_x)
        cta_n = offset_in_group // swizzle_x
        
        M_dim = grad_output.shape[0]
        K_dim = self.K_dim
        N_dim = self.N_dim
        
        actual_grid_dim_x = cute.ceil_div(M_dim, self.BLOCK_M)
        is_valid = cta_m_raw < actual_grid_dim_x
        cta_m = cutlass.min(cta_m_raw, actual_grid_dim_x - 1)
        
        k_tile_cnt = self.k_tile_cnt
        
        # Prefetch TMA descriptors
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_o)
            
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        sW_aligned_row = cute.make_tensor(sW_aligned.iterator, b_smem_layout_row)
        
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Partition accumulator C (registers)
        smem_accum = cute.make_tensor(storage.sX_raw.data_ptr().align(16), epi_smem_layout)
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        # Partition SMEM stages
        tCsX = thr_mma.partition_A(sX_aligned)
        tCsW = thr_mma.partition_B(sW_aligned)
        
        # Pre-allocate double-buffered register fragments
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        # Partition global tensors inside kernel
        gA_tile = cute.local_tile(tma_tensor_a, (self.BLOCK_M, self.BLOCK_K_STAGE), (cta_m, None))
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_tile,
            dst_tensor=sX_aligned,
            single_stage=False
        )
        
        gB_tile = cute.local_tile(tma_tensor_b, (self.BLOCK_K_STAGE, self.BLOCK_N), (None, cta_n))
        copy_B, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_b,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gB_tile,
            dst_tensor=sW_aligned_row,
            single_stage=False
        )
        
        mbar_ptr = cute.arch.alloc_smem(Int64, self.ab_stage)
        if tidx == 0:
            for s in cutlass.range_constexpr(self.ab_stage):
                cute.arch.mbarrier_init(mbar_ptr + s, 1)
            cute.arch.mbarrier_init_fence()
            
        cute.arch.sync_threads()
        
        # Prefetch initial stages
        for stage in cutlass.range_constexpr(self.ab_stage - 1):
            if stage < k_tile_cnt:
                if tidx < 32:
                    tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                    copy_A(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_B(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                    
        # Wait for stage 0 and prefetch to registers
        if k_tile_cnt > 0:
            cute.arch.mbarrier_wait(mbar_ptr + 0, 0)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
            
        # Main execution loop
        for k_tile in cutlass.range_constexpr(self.k_tile_cnt):
            pipe_read = k_tile % self.ab_stage
            pipe_write = (k_tile + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (k_tile + 1) % self.ab_stage
            
            next_gmem_read = k_tile + (self.ab_stage - 1)
            if next_gmem_read < k_tile_cnt:
                if tidx < 32:
                    tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                    copy_A(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    
            # Prefetch next stage's registers
            if k_tile + 1 < k_tile_cnt:
                next_phase = ((k_tile + 1) // self.ab_stage) % 2
                cute.arch.mbarrier_wait(mbar_ptr + next_pipe_read, next_phase)
                if use_sync_in_loop:
                    cute.arch.sync_threads()
                    
                tCsA_next = tCsX[None, None, None, next_pipe_read]
                tCsB_next = tCsW[None, None, None, next_pipe_read]
                
                if k_tile % 2 == 0:
                    cute.autovec_copy(tCsA_next, rA_1)
                    cute.autovec_copy(tCsB_next, rB_1)
                else:
                    cute.autovec_copy(tCsA_next, rA_0)
                    cute.autovec_copy(tCsB_next, rB_0)
                    
            # Perform MMA
            if k_tile % 2 == 0:
                cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
            else:
                cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                
        # Write Float32 accumulator to sBuf
        cute.copy(copy_atom_r2s, accum, tCsC)
        cute.arch.sync_threads()
        
        r = tidx % self.BLOCK_M
        c = tidx // self.BLOCK_M
        
        cols_per_thread = self.BLOCK_N // (self.num_warps * 32 // self.BLOCK_M)
        col_start = c * cols_per_thread
        row_offset = r * self.BLOCK_N_padded + col_start
        
        # 1. Read Float32 accumulators from smem_accum to local registers
        q_reg = cute.make_rmem_tensor((cols_per_thread,), Float32)
        q_reg.fill(0.0)
        
        if is_valid and cta_m * self.BLOCK_M + r < M_dim and col_start < self.BLOCK_N:
            q_ptr = smem_accum.iterator + row_offset
            q_smem = cute.make_tensor(q_ptr, cute.make_layout((cols_per_thread,)))
            cute.autovec_copy(q_smem, q_reg)
            
        # Synchronize to ensure all threads have read their Float32 accumulators from shared memory
        cute.arch.sync_threads()
        
        # 2. Write BFloat16 values back to the same shared memory buffer (sOut_bf16)
        sOut_bf16 = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.BLOCK_N), stride=(self.BLOCK_N_padded, 1)))
        
        if is_valid and cta_m * self.BLOCK_M + r < M_dim and col_start < self.BLOCK_N:
            out_q_smem = cute.make_tensor(sOut_bf16.iterator + row_offset, cute.make_layout((cols_per_thread,)))
            q_bf16 = cute.make_rmem_tensor((cols_per_thread,), BFloat16)
            for j in cutlass.range(cols_per_thread, unroll_full=True):
                q_bf16[j] = BFloat16(q_reg[j])
            cute.autovec_copy(q_bf16, out_q_smem)
            
        # Synchronize to ensure all threads have written their BFloat16 values to shared memory before global store
        cute.arch.sync_threads()
        
        # TMA Epilogue store
        gO_tma_tile = cute.local_tile(tma_tensor_o, (self.BLOCK_M, self.BLOCK_N), (cta_m, cta_n))
        copy_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_o,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=sOut_bf16,
            dst_tensor=gO_tma_tile,
            single_stage=True
        )
        
        if is_valid and cta_m * self.BLOCK_M < M_dim and cta_n * self.BLOCK_N < N_dim:
            if tidx < 32:
                copy_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)


class GemmBwdDwCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K_STAGE: int = 32, ab_stage: int = 3, M_dim: int = 3072, K_dim: int = 512, N_dim: int = 512, SPLIT_K: int = 16):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K_STAGE = BLOCK_K_STAGE
        self.ab_stage = ab_stage
        self.M_dim = M_dim
        self.K_dim = K_dim
        self.N_dim = N_dim
        self.SPLIT_K = SPLIT_K
        self.k_tile_cnt = K_dim // BLOCK_K_STAGE
        
    def _setup(self):
        # Staged shared memory layouts (Dw is column-major/strided for both X and W)
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_M, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_N, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        self.a_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_M, self.ab_stage),
            stride=(self.BLOCK_M, 1, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        self.BLOCK_N_padded = self.BLOCK_N
        self.epi_smem_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N_padded, 1)
        )
        
        sX_bytes = self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 2
        sW_bytes = self.ab_stage * self.BLOCK_N * self.BLOCK_K_STAGE * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = self.BLOCK_M * self.BLOCK_N_padded * 4
        
        self.max_smem_bytes = max(mainloop_bytes, epilogue_bytes)
        self.flat_smem_elements = cute.ceil_div(self.max_smem_bytes, 4)
        self.sW_offset_elements = sX_bytes // 4
        
        @cute.struct
        class SharedStorageBwdDw:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageBwdDw
        
        # TiledMMA Configuration
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=(2, 2, 1))
        
        # Copy atom for Float32 accumulator writeback
        copy_op_r2s = cute.nvUniversalCopyOp() if hasattr(cute, "nvUniversalCopyOp") else cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, Float32)

    @cute.jit
    def __call__(
        self,
        grad_output_t: cute.Tensor,
        x: cute.Tensor,
        grad_weight: cute.Tensor,
        stream
    ):
        self._setup()
        
        num_threads = self.num_warps * 32
        
        gA = cute.make_tensor(grad_output_t.iterator, cute.make_layout((self.K_dim, self.M_dim), stride=(self.M_dim, 1)))
        gB = cute.make_tensor(x.iterator, cute.make_layout((self.K_dim, self.N_dim), stride=(self.N_dim, 1)))
        gO = cute.make_tensor(grad_weight.iterator, cute.make_layout((self.M_dim, self.N_dim), stride=(self.N_dim, 1)))
        
        # Make TMA load/store descriptors
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_M), stride=(self.BLOCK_M, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_M)
        )
        
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op,
            gO,
            self.epi_smem_layout,
            (self.BLOCK_M, self.BLOCK_N)
        )
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(self.M_dim, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            grad_output_t, x, grad_weight,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.a_smem_layout_staged_row,
            self.b_smem_layout_staged_row,
            self.epi_smem_layout,
            self.copy_atom_r2s,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_o, tma_tensor_o,
            Boolean(True)
        ).launch(
            grid=[swizzled_grid_x, cute.ceil_div(self.N_dim, self.BLOCK_N), self.SPLIT_K],
            block=[num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        grad_output_t: cute.Tensor,
        x: cute.Tensor,
        grad_weight: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        a_smem_layout_row: cute.Layout,
        b_smem_layout_row: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s: cute.CopyAtom,
        tma_atom_a: cute.CopyAtom,
        tma_tensor_a: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        tma_tensor_b: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        use_sync_in_loop: Boolean
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        tidx, _, _ = cute.arch.thread_idx()
        
        # 2D Grid Swizzling with compile-time constant group size
        swizzle_x = 8
        grid_dim_y_const = self.N_dim // self.BLOCK_N
        group_size = swizzle_x * grid_dim_y_const
        
        pid = block_idx_y * grid_dim_x + block_idx_x
        swizzle_group = pid // group_size
        offset_in_group = pid % group_size
        
        cta_m_raw = swizzle_group * swizzle_x + (offset_in_group % swizzle_x)
        cta_n = offset_in_group // swizzle_x
        
        M_dim = self.M_dim
        actual_grid_dim_x = cute.ceil_div(M_dim, self.BLOCK_M)
        is_valid = cta_m_raw < actual_grid_dim_x
        cta_m = cutlass.min(cta_m_raw, actual_grid_dim_x - 1)
        M_dim = self.M_dim
        K_dim = self.K_dim
        N_dim = self.N_dim
        
        k_tile_cnt = self.k_tile_cnt
        k_tiles_per_slice = (k_tile_cnt + self.SPLIT_K - 1) // self.SPLIT_K
        k_tile_start = block_idx_z * k_tiles_per_slice
        
        # Prefetch TMA descriptors
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_o)
            
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        sX_aligned_row = cute.make_tensor(sX_aligned.iterator, a_smem_layout_row)
        sW_aligned_row = cute.make_tensor(sW_aligned.iterator, b_smem_layout_row)
        
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Partition accumulator C (registers)
        smem_accum = cute.make_tensor(storage.sX_raw.data_ptr().align(16), epi_smem_layout)
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        # Partition SMEM stages
        tCsX = thr_mma.partition_A(sX_aligned)
        tCsW = thr_mma.partition_B(sW_aligned)
        
        # Pre-allocate double-buffered register fragments
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        # Partition global tensors inside kernel
        gA_tile = cute.local_tile(tma_tensor_a, (self.BLOCK_K_STAGE, self.BLOCK_M), (None, cta_m))
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_tile,
            dst_tensor=sX_aligned_row,
            single_stage=False
        )
        
        gB_tile = cute.local_tile(tma_tensor_b, (self.BLOCK_K_STAGE, self.BLOCK_N), (None, cta_n))
        copy_B, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_b,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gB_tile,
            dst_tensor=sW_aligned_row,
            single_stage=False
        )
        
        mbar_ptr = cute.arch.alloc_smem(Int64, self.ab_stage)
        if tidx == 0:
            for s in cutlass.range_constexpr(self.ab_stage):
                cute.arch.mbarrier_init(mbar_ptr + s, 1)
            cute.arch.mbarrier_init_fence()
            
        cute.arch.sync_threads()
        
        # Prefetch initial stages
        for stage in cutlass.range_constexpr(self.ab_stage - 1):
            if stage < k_tiles_per_slice:
                k_curr = cutlass.min(k_tile_start + stage, k_tile_cnt - 1)
                if tidx < 32:
                    tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                    copy_A(k_curr, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_B(k_curr, stage, tma_bar_ptr=mbar_ptr + stage)
                    
        # Wait for stage 0 and prefetch to registers
        if k_tiles_per_slice > 0:
            cute.arch.mbarrier_wait(mbar_ptr + 0, 0)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
            
        # Main execution loop
        for step in cutlass.range_constexpr(k_tiles_per_slice):
            pipe_read = step % self.ab_stage
            pipe_write = (step + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (step + 1) % self.ab_stage
            
            next_gmem_read = step + (self.ab_stage - 1)
            if next_gmem_read < k_tiles_per_slice:
                k_curr = cutlass.min(k_tile_start + next_gmem_read, k_tile_cnt - 1)
                if tidx < 32:
                    tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                    copy_A(k_curr, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B(k_curr, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    
            # Prefetch next stage's registers
            if step + 1 < k_tiles_per_slice:
                next_phase = ((step + 1) // self.ab_stage) % 2
                cute.arch.mbarrier_wait(mbar_ptr + next_pipe_read, next_phase)
                if use_sync_in_loop:
                    cute.arch.sync_threads()
                    
                tCsA_next = tCsX[None, None, None, next_pipe_read]
                tCsB_next = tCsW[None, None, None, next_pipe_read]
                
                if step % 2 == 0:
                    cute.autovec_copy(tCsA_next, rA_1)
                    cute.autovec_copy(tCsB_next, rB_1)
                else:
                    cute.autovec_copy(tCsA_next, rA_0)
                    cute.autovec_copy(tCsB_next, rB_0)
                    
            # Perform MMA
            if k_tile_start + step < k_tile_cnt:
                if step % 2 == 0:
                    cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
                else:
                    cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                    
        # Write Float32 accumulator to sBuf
        cute.copy(copy_atom_r2s, accum, tCsC)
        cute.arch.sync_threads()
        
        # TMA Epilogue atomic reduction store
        gO_tma_tile = cute.local_tile(tma_tensor_o, (self.BLOCK_M, self.BLOCK_N), (cta_m, cta_n))
        copy_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_o,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=smem_accum,
            dst_tensor=gO_tma_tile,
            single_stage=True
        )
        
        if is_valid and cta_m * self.BLOCK_M < M_dim and cta_n * self.BLOCK_N < N_dim:
            if tidx < 32:
                copy_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)


_dx_compile_cache = {}
_dw_compile_cache = {}

def get_matmul_bwd_dx_kernel(num_warps=4, BLOCK_M=64, BLOCK_N=64, BLOCK_K_STAGE=32, ab_stage=3, K_dim=3072, N_dim=512, maxrregcount=255):
    key = (num_warps, BLOCK_M, BLOCK_N, BLOCK_K_STAGE, ab_stage, K_dim, N_dim, maxrregcount)
    if key not in _dx_compile_cache:
        M_sym = cute.sym_int()
        
        op = GemmBwdDxCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K_STAGE=BLOCK_K_STAGE,
            ab_stage=ab_stage,
            M_dim=M_sym,
            K_dim=K_dim,
            N_dim=N_dim
        )
        
        grad_output_fake = make_fake_tensor(BFloat16, (M_sym, K_dim), 16)
        weight_fake = make_fake_tensor(BFloat16, (K_dim, N_dim), 16)
        grad_x_fake = make_fake_tensor(BFloat16, (M_sym, N_dim), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fn = cute.compile(
            op,
            grad_output_fake, weight_fake, grad_x_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _dx_compile_cache[key] = compiled_fn
    return _dx_compile_cache[key]

def get_matmul_bwd_dw_kernel(num_warps=4, BLOCK_M=64, BLOCK_N=64, BLOCK_K_STAGE=32, ab_stage=3, M_dim=3072, K_dim=512, N_dim=512, maxrregcount=255, SPLIT_K=16):
    key = (num_warps, BLOCK_M, BLOCK_N, BLOCK_K_STAGE, ab_stage, M_dim, K_dim, N_dim, maxrregcount, SPLIT_K)
    if key not in _dw_compile_cache:
        op = GemmBwdDwCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K_STAGE=BLOCK_K_STAGE,
            ab_stage=ab_stage,
            M_dim=M_dim,
            K_dim=K_dim,
            N_dim=N_dim,
            SPLIT_K=SPLIT_K
        )
        
        grad_output_t_fake = make_fake_tensor(BFloat16, (M_dim, K_dim), 16, leading_dim=0)
        x_fake = make_fake_tensor(BFloat16, (K_dim, N_dim), 16)
        grad_weight_fake = make_fake_tensor(Float32, (M_dim, N_dim), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fn = cute.compile(
            op,
            grad_output_t_fake, x_fake, grad_weight_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _dw_compile_cache[key] = compiled_fn
    return _dw_compile_cache[key]

@torch.library.custom_op("flash_prologue::matmul_bwd_dx_cute", mutates_args=())
def matmul_bwd_dx(
    grad_output: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    M_dim, K_dim = grad_output.shape
    K_dim2, N_dim = weight.shape
    assert K_dim == K_dim2, "Dimension mismatch for dx"
    
    grad_output_bf16 = grad_output.to(torch.bfloat16).contiguous()
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    
    grad_x = torch.empty((M_dim, N_dim), dtype=torch.bfloat16, device=grad_output.device)
    
    from flash_attn_prologue.autotune import get_autotuned_config
    config = get_autotuned_config(M_dim, N_dim, K_dim, "bwd_dx", device=str(grad_output.device))
    
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    ab_stage = config.get("ab_stage", 3)
    num_warps = config.get("num_warps", 4)
    maxrregcount = config.get("maxrregcount", 255)
    
    kernel = get_matmul_bwd_dx_kernel(
        num_warps=num_warps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K_STAGE=block_k,
        ab_stage=ab_stage,
        K_dim=K_dim,
        N_dim=N_dim,
        maxrregcount=maxrregcount
    )
    
    kernel(grad_output_bf16, weight_bf16, grad_x)
    return grad_x

@matmul_bwd_dx.register_fake
def _matmul_bwd_dx_fake(grad_output, weight):
    M_dim, _ = grad_output.shape
    _, N_dim = weight.shape
    return torch.empty((M_dim, N_dim), dtype=grad_output.dtype, device=grad_output.device)

@torch.library.custom_op("flash_prologue::matmul_bwd_dw_cute", mutates_args=())
def matmul_bwd_dw(
    grad_output_t: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    M_dim, K_dim = grad_output_t.shape
    K_dim2, N_dim = x.shape
    assert K_dim == K_dim2, "Dimension mismatch for dw"
    
    grad_output_t_bf16 = grad_output_t.to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16).contiguous()
    
    # Initialize output tensor as Float32 zeros for Split-K atomic accumulations
    grad_weight = torch.zeros((M_dim, N_dim), dtype=torch.float32, device=grad_output_t.device)
    
    from flash_attn_prologue.autotune import get_autotuned_config
    config = get_autotuned_config(M_dim, N_dim, K_dim, "bwd_dw", device=str(grad_output_t.device))
    
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_k = config["BLOCK_K"]
    ab_stage = config.get("ab_stage", 2)
    num_warps = config.get("num_warps", 4)
    maxrregcount = config.get("maxrregcount", 255)
    
    kernel = get_matmul_bwd_dw_kernel(
        num_warps=num_warps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K_STAGE=block_k,
        ab_stage=ab_stage,
        M_dim=M_dim,
        K_dim=K_dim,
        N_dim=N_dim,
        maxrregcount=maxrregcount,
        SPLIT_K=16
    )
    
    kernel(grad_output_t_bf16, x_bf16, grad_weight)
    return grad_weight.to(torch.bfloat16)

@matmul_bwd_dw.register_fake
def _matmul_bwd_dw_fake(grad_output_t, x):
    M_dim, _ = grad_output_t.shape
    _, N_dim = x.shape
    return torch.empty((M_dim, N_dim), dtype=grad_output_t.dtype, device=grad_output_t.device)
