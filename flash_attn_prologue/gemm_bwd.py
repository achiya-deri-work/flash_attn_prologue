# Copyright (c) 2026, Antigravity.
# Raw JIT-compiled backward GEMM kernels using CuTe Python DSL.

import os
import torch
import math
from typing import Optional


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
from flash_attn_prologue.autotune import get_autotuned_config

def _ensure_contiguous(t):
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()

class GemmBwdDxCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K_STAGE: int = 32, ab_stage: int = 3, B: int = 32, N: int = 512, M_dim = None, K_dim: int = 3072, N_dim: int = 512, NUM_HEADS: int = 16, HEAD_DIM: int = 64, q_strides=None, k_strides=None, v_strides=None, swizzle_x: int = 8, atom_layout_mnk = None):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K_STAGE = BLOCK_K_STAGE
        self.ab_stage = ab_stage
        self.B = B
        self.N = N
        self.M_dim = M_dim
        self.K_dim = K_dim
        self.N_dim = N_dim
        self.NUM_HEADS = NUM_HEADS
        self.HEAD_DIM = HEAD_DIM
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.swizzle_x = swizzle_x
        self.atom_layout_mnk = atom_layout_mnk
        
    def _setup(self):
        # sX is row-major (since grad_output is row-major)
        # Nested layouts represent the 3 components Q, K, V contiguously.
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, (self.BLOCK_K_STAGE, 3), self.ab_stage),
            stride=(self.BLOCK_K_STAGE, (1, self.BLOCK_M * self.BLOCK_K_STAGE), 3 * self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        # sW is column-major
        k_iter_val = self.BLOCK_K_STAGE // 16
        if k_iter_val > 1:
            k_shape = (16, (k_iter_val, 3))
            k_stride = (self.BLOCK_N, (16 * self.BLOCK_N, k_iter_val * 16 * self.BLOCK_N))
        else:
            k_shape = (16, 3)
            k_stride = (self.BLOCK_N, 16 * self.BLOCK_N)
            
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, k_shape, self.ab_stage),
            stride=(1, k_stride, 3 * self.BLOCK_K_STAGE * self.BLOCK_N)
        )
        self.b_smem_layout_staged_row = cute.make_layout(
            (k_shape, self.BLOCK_N, self.ab_stage),
            stride=(k_stride, 1, 3 * self.BLOCK_K_STAGE * self.BLOCK_N)
        )
        
        self.BLOCK_N_padded = self.BLOCK_N + 8  # Pad to resolve LDS bank conflicts
        self.epi_smem_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N_padded, 1)
        )
        
        # Calculate shared memory footprint
        sX_bytes = self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 3 * 2
        sW_bytes = self.ab_stage * self.BLOCK_N * self.BLOCK_K_STAGE * 3 * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = self.BLOCK_M * self.BLOCK_N_padded * 2  # Aliased, BFloat16 is the largest size
        
        self.max_smem_bytes = max(mainloop_bytes, epilogue_bytes)
        self.flat_smem_elements = cute.ceil_div(self.max_smem_bytes, 4)
        self.sW_offset_elements = sX_bytes // 4
        
        @cute.struct
        class SharedStorageBwdDx:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageBwdDx
        
        # TiledMMA Configuration
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        if self.atom_layout_mnk is not None:
            atom_layout_mnk = self.atom_layout_mnk
        else:
            if self.num_warps == 4:
                atom_layout_mnk = (2, 2, 1)
            elif self.num_warps == 2:
                atom_layout_mnk = (2, 1, 1)
            elif self.num_warps == 6:
                atom_layout_mnk = (3, 2, 1)
            elif self.num_warps == 8:
                atom_layout_mnk = (4, 2, 1)
            elif self.num_warps == 12:
                atom_layout_mnk = (4, 3, 1)
            elif self.num_warps == 16:
                atom_layout_mnk = (4, 4, 1)
            else:
                atom_layout_mnk = (self.num_warps, 1, 1)
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=atom_layout_mnk)
        
        # Copy atom for Float32 accumulator writeback
        copy_op_r2s = cute.nvUniversalCopyOp() if hasattr(cute, "nvUniversalCopyOp") else cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s_bf16 = cute.make_copy_atom(copy_op_r2s, BFloat16)

        # TiledCopy configuration for direct-store epilogue
        store_op = cute.nvgpu.CopyUniversalOp()
        store_atom = cute.make_copy_atom(store_op, BFloat16, num_bits_per_copy=128)
        store_thr_m = min(16, self.BLOCK_M)
        store_thr_n = (self.num_warps * 32) // store_thr_m
        self.gmem_tiled_copy_store = cute.make_tiled_copy_tv(
            store_atom,
            cute.make_ordered_layout((store_thr_m, store_thr_n), order=(1, 0)),
            cute.make_layout((1, 8))
        )

    @cute.jit
    def __call__(
        self,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_v: cute.Tensor,
        weight: cute.Tensor,
        grad_x: cute.Tensor,
        stream
    ):
        self._setup()
        
        M_dim = grad_q.shape[0] * grad_q.shape[2] # B * N
        num_threads = self.num_warps * 32
        
        K_dim_single = self.NUM_HEADS * self.HEAD_DIM
        offset_W = K_dim_single * self.N_dim
        
        gB_q = cute.make_tensor(weight.iterator, cute.make_layout((K_dim_single, self.N_dim), stride=(self.N_dim, 1)))
        gB_k = cute.make_tensor(weight.iterator + offset_W, cute.make_layout((K_dim_single, self.N_dim), stride=(self.N_dim, 1)))
        gB_v = cute.make_tensor(weight.iterator + 2 * offset_W, cute.make_layout((K_dim_single, self.N_dim), stride=(self.N_dim, 1)))
        
        # Create 3D global layouts for split gradients A
        gA_q = cute.make_tensor(
            grad_q.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.q_strides[1], self.q_strides[2], self.q_strides[3])
            )
        )
        gA_k = cute.make_tensor(
            grad_k.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.k_strides[1], self.k_strides[2], self.k_strides[3])
            )
        )
        gA_v = cute.make_tensor(
            grad_v.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.v_strides[1], self.v_strides[2], self.v_strides[3])
            )
        )
        
        # Make TMA load descriptors
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        tma_atom_bq, tma_tensor_bq = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB_q,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        tma_atom_bk, tma_tensor_bk = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB_k,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        tma_atom_bv, tma_tensor_bv = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB_v,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        
        smem_layout_a_comp = cute.make_layout(
            (1, self.BLOCK_M, self.BLOCK_K_STAGE),
            stride=(self.BLOCK_M * self.BLOCK_K_STAGE, self.BLOCK_K_STAGE, 1)
        )
        
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_q,
            smem_layout_a_comp,
            (1, self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_k,
            smem_layout_a_comp,
            (1, self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_v,
            smem_layout_a_comp,
            (1, self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        
        swizzle_x = self.swizzle_x
        grid_x_blocks = cute.ceil_div(M_dim, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            grad_q, grad_k, grad_v, weight, grad_x,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.b_smem_layout_staged_row,
            self.epi_smem_layout,
            self.copy_atom_r2s_bf16,
            self.gmem_tiled_copy_store,
            tma_atom_bq, tma_tensor_bq,
            tma_atom_bk, tma_tensor_bk,
            tma_atom_bv, tma_tensor_bv,
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v
        ).launch(
            grid=[swizzled_grid_x, cute.ceil_div(self.N_dim, self.BLOCK_N), 1],
            block=[num_threads, 1, 1],
            cluster=[2, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_v: cute.Tensor,
        weight: cute.Tensor,
        grad_x: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        b_smem_layout_row: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s_bf16: cute.CopyAtom,
        gmem_tiled_copy_store: cute.TiledCopy,
        tma_atom_bq: cute.CopyAtom,
        tma_tensor_bq: cute.Tensor,
        tma_atom_bk: cute.CopyAtom,
        tma_tensor_bk: cute.Tensor,
        tma_atom_bv: cute.CopyAtom,
        tma_tensor_bv: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = self.num_warps * 32
        
        # 2D Grid Swizzling with compile-time constant group size
        swizzle_x = self.swizzle_x
        grid_dim_y_const = self.N_dim // self.BLOCK_N
        group_size = swizzle_x * grid_dim_y_const
        
        pid = block_idx_y * grid_dim_x + block_idx_x
        swizzle_group = pid // group_size
        offset_in_group = pid % group_size
        
        cta_m_raw = swizzle_group * swizzle_x + (offset_in_group % swizzle_x)
        cta_n = offset_in_group // swizzle_x
        
        B = self.B
        N = self.N
        M_dim = B * N
        K_dim = self.K_dim
        N_dim = self.N_dim
        
        actual_grid_dim_x = cute.ceil_div(M_dim, self.BLOCK_M)
        is_valid = cta_m_raw < actual_grid_dim_x
        cta_m = cutlass.min(cta_m_raw, actual_grid_dim_x - 1)
        
        K_dim_single = K_dim // 3
        k_tile_cnt = K_dim_single // self.BLOCK_K_STAGE
        
        # Prefetch TMA descriptors
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_bq)
            cpasync.prefetch_descriptor(tma_atom_bk)
            cpasync.prefetch_descriptor(tma_atom_bv)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            
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
        
        mbar_ptr = cute.arch.alloc_smem(Int64, self.ab_stage)
        if tidx == 0:
            for s in cutlass.range_constexpr(self.ab_stage):
                cute.arch.mbarrier_init(mbar_ptr + s, 1)
            cute.arch.mbarrier_init_fence()
            
        cute.arch.sync_threads()
        
        b = (cta_m * self.BLOCK_M) // N
        n_start = (cta_m * self.BLOCK_M) % N
        n_tile = n_start // self.BLOCK_M
        
        comp_size_X = self.BLOCK_M * self.BLOCK_K_STAGE
        comp_size_W = self.BLOCK_K_STAGE * self.BLOCK_N
        
        # Prefetch initial stages
        for stage in cutlass.range_constexpr(self.ab_stage - 1):
            if stage < k_tile_cnt:
                k_curr = stage * self.BLOCK_K_STAGE
                h = k_curr // self.HEAD_DIM
                d_tile = (k_curr % self.HEAD_DIM) // self.BLOCK_K_STAGE
                b_h = b * self.NUM_HEADS + h
                
                # Gradients tiles
                gA_q_tile = cute.local_tile(tma_tensor_q, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                gA_k_tile = cute.local_tile(tma_tensor_k, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                gA_v_tile = cute.local_tile(tma_tensor_v, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                
                # Weights tiles
                gB_q_tile = cute.local_tile(tma_tensor_bq, (self.BLOCK_K_STAGE, self.BLOCK_N), (stage, cta_n))
                gB_k_tile = cute.local_tile(tma_tensor_bk, (self.BLOCK_K_STAGE, self.BLOCK_N), (stage, cta_n))
                gB_v_tile = cute.local_tile(tma_tensor_bv, (self.BLOCK_K_STAGE, self.BLOCK_N), (stage, cta_n))
                
                sX_stage_q = cute.make_tensor(sX_aligned.iterator + stage * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_q_3d = cute.make_tensor(sX_stage_q.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sX_stage_k = cute.make_tensor(sX_aligned.iterator + comp_size_X + stage * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_k_3d = cute.make_tensor(sX_stage_k.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sX_stage_v = cute.make_tensor(sX_aligned.iterator + 2 * comp_size_X + stage * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_v_3d = cute.make_tensor(sX_stage_v.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sW_stage_q_row = cute.make_tensor(sW_aligned_row.iterator + stage * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                sW_stage_k_row = cute.make_tensor(sW_aligned_row.iterator + comp_size_W + stage * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                sW_stage_v_row = cute.make_tensor(sW_aligned_row.iterator + 2 * comp_size_W + stage * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                
                copy_A_q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_q, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_q_tile, dst_tensor=sX_stage_q_3d, single_stage=True)
                copy_A_k, _, _ = copy_utils.tma_get_copy_fn(tma_atom_k, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_k_tile, dst_tensor=sX_stage_k_3d, single_stage=True)
                copy_A_v, _, _ = copy_utils.tma_get_copy_fn(tma_atom_v, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_v_tile, dst_tensor=sX_stage_v_3d, single_stage=True)
                
                copy_B_q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bq, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_q_tile, dst_tensor=sW_stage_q_row, single_stage=True)
                copy_B_k, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bk, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_k_tile, dst_tensor=sW_stage_k_row, single_stage=True)
                copy_B_v, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bv, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_v_tile, dst_tensor=sW_stage_v_row, single_stage=True)
                
                if tidx < 32:
                    if tidx == 0:
                        tx_bytes = 3 * (self.BLOCK_M * self.BLOCK_K_STAGE * 2 + self.BLOCK_K_STAGE * self.BLOCK_N * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                    cute.arch.sync_warp()
                    copy_A_q(tma_bar_ptr=mbar_ptr + stage)
                    copy_A_k(tma_bar_ptr=mbar_ptr + stage)
                    copy_A_v(tma_bar_ptr=mbar_ptr + stage)
                    copy_B_q(tma_bar_ptr=mbar_ptr + stage)
                    copy_B_k(tma_bar_ptr=mbar_ptr + stage)
                    copy_B_v(tma_bar_ptr=mbar_ptr + stage)
                    
        # Wait for stage 0 and prefetch to registers
        if k_tile_cnt > 0:
            cute.arch.mbarrier_wait(mbar_ptr + 0, 0)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
            
        cute.arch.setmaxregister_increase(248)
        
        # Main execution loop
        for k_tile in cutlass.range_constexpr(k_tile_cnt):
            pipe_read = k_tile % self.ab_stage
            pipe_write = (k_tile + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (k_tile + 1) % self.ab_stage
            
            next_gmem_read = k_tile + (self.ab_stage - 1)
            if next_gmem_read < k_tile_cnt:
                k_curr = next_gmem_read * self.BLOCK_K_STAGE
                h = k_curr // self.HEAD_DIM
                d_tile = (k_curr % self.HEAD_DIM) // self.BLOCK_K_STAGE
                b_h = b * self.NUM_HEADS + h
                
                # Gradients tiles
                gA_q_tile = cute.local_tile(tma_tensor_q, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                gA_k_tile = cute.local_tile(tma_tensor_k, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                gA_v_tile = cute.local_tile(tma_tensor_v, (1, self.BLOCK_M, self.BLOCK_K_STAGE), (b_h, n_tile, d_tile))
                
                # Weights tiles
                gB_q_tile = cute.local_tile(tma_tensor_bq, (self.BLOCK_K_STAGE, self.BLOCK_N), (next_gmem_read, cta_n))
                gB_k_tile = cute.local_tile(tma_tensor_bk, (self.BLOCK_K_STAGE, self.BLOCK_N), (next_gmem_read, cta_n))
                gB_v_tile = cute.local_tile(tma_tensor_bv, (self.BLOCK_K_STAGE, self.BLOCK_N), (next_gmem_read, cta_n))
                
                sX_stage_q = cute.make_tensor(sX_aligned.iterator + pipe_write * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_q_3d = cute.make_tensor(sX_stage_q.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sX_stage_k = cute.make_tensor(sX_aligned.iterator + comp_size_X + pipe_write * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_k_3d = cute.make_tensor(sX_stage_k.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sX_stage_v = cute.make_tensor(sX_aligned.iterator + 2 * comp_size_X + pipe_write * (3 * comp_size_X), cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)))
                sX_stage_v_3d = cute.make_tensor(sX_stage_v.iterator, cute.make_layout((1, self.BLOCK_M, self.BLOCK_K_STAGE), stride=(comp_size_X, self.BLOCK_K_STAGE, 1)))
                
                sW_stage_q_row = cute.make_tensor(sW_aligned_row.iterator + pipe_write * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                sW_stage_k_row = cute.make_tensor(sW_aligned_row.iterator + comp_size_W + pipe_write * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                sW_stage_v_row = cute.make_tensor(sW_aligned_row.iterator + 2 * comp_size_W + pipe_write * (3 * comp_size_W), cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)))
                
                copy_A_q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_q, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_q_tile, dst_tensor=sX_stage_q_3d, single_stage=True)
                copy_A_k, _, _ = copy_utils.tma_get_copy_fn(tma_atom_k, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_k_tile, dst_tensor=sX_stage_k_3d, single_stage=True)
                copy_A_v, _, _ = copy_utils.tma_get_copy_fn(tma_atom_v, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gA_v_tile, dst_tensor=sX_stage_v_3d, single_stage=True)
                
                copy_B_q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bq, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_q_tile, dst_tensor=sW_stage_q_row, single_stage=True)
                copy_B_k, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bk, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_k_tile, dst_tensor=sW_stage_k_row, single_stage=True)
                copy_B_v, _, _ = copy_utils.tma_get_copy_fn(tma_atom_bv, cta_coord=0, cta_layout=cute.make_layout(1), src_tensor=gB_v_tile, dst_tensor=sW_stage_v_row, single_stage=True)
                
                if tidx < 32:
                    if tidx == 0:
                        tx_bytes = 3 * (self.BLOCK_M * self.BLOCK_K_STAGE * 2 + self.BLOCK_K_STAGE * self.BLOCK_N * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                    cute.arch.sync_warp()
                    copy_A_q(tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_A_k(tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_A_v(tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B_q(tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B_k(tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B_v(tma_bar_ptr=mbar_ptr + pipe_write)
                    
            # Perform MMA using current stage registers immediately (overlaps with step 1 TMA load)
            if k_tile % 2 == 0:
                cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
            else:
                cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                
            # Wait for next stage's TMA load and copy it from SMEM to registers
            if k_tile + 1 < k_tile_cnt:
                next_phase = ((k_tile + 1) // self.ab_stage) % 2
                cute.arch.mbarrier_wait(mbar_ptr + next_pipe_read, next_phase)
                cute.arch.sync_threads()
                
                tCsA_next = tCsX[None, None, None, next_pipe_read]
                tCsB_next = tCsW[None, None, None, next_pipe_read]
                
                if k_tile % 2 == 0:
                    cute.autovec_copy(tCsA_next, rA_1)
                    cute.autovec_copy(tCsB_next, rB_1)
                else:
                    cute.autovec_copy(tCsA_next, rA_0)
                    cute.autovec_copy(tCsB_next, rB_0)

        cute.arch.sync_threads()
        cute.arch.setmaxregister_decrease(128)
        
        # Write Float32 accumulator directly to sOut_bf16 by casting to BFloat16 in registers first
        sOut_bf16 = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), epi_smem_layout)
        tCsOut = thr_mma.partition_C(sOut_bf16)
        accum_bf16 = cute.make_fragment_like(accum, BFloat16)
        
        for idx in cutlass.range(cute.size(accum), unroll_full=True):
            accum_bf16[idx] = BFloat16(accum[idx])
            
        cute.copy(copy_atom_r2s_bf16, accum_bf16, tCsOut)
        cute.arch.sync_threads()
        
        # 3. Vectorized global stores using gmem_tiled_copy_store
        if is_valid:
            g_offset = cta_m * self.BLOCK_M * N_dim + cta_n * self.BLOCK_N
            g_tile = cute.make_tensor((grad_x.iterator + g_offset).align(16), cute.make_layout((self.BLOCK_M, self.BLOCK_N), stride=(N_dim, 1)))
            
            thr_copy_store = gmem_tiled_copy_store.get_slice(tidx)
            tS = thr_copy_store.partition_S(sOut_bf16)
            tD = thr_copy_store.partition_D(g_tile)
            
            cute.copy(gmem_tiled_copy_store, tS, tD)


class GemmBwdDwCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K_STAGE: int = 32, ab_stage: int = 3, B: int = 32, N: int = 512, M_dim: int = 3072, K_dim = None, N_dim: int = 512, NUM_HEADS: int = 16, HEAD_DIM: int = 64, q_strides=None, k_strides=None, v_strides=None, SPLIT_K: int = 1, swizzle_x: int = 8, accumulate: bool = False, atom_layout_mnk = None):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K_STAGE = BLOCK_K_STAGE
        self.ab_stage = ab_stage
        self.B = B
        self.N = N
        self.M_dim = M_dim
        self.K_dim = K_dim
        self.N_dim = N_dim
        self.NUM_HEADS = NUM_HEADS
        self.HEAD_DIM = HEAD_DIM
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.SPLIT_K = SPLIT_K
        self.swizzle_x = swizzle_x
        self.accumulate = accumulate
        self.atom_layout_mnk = atom_layout_mnk
        
    def _setup(self):
        # Staged shared memory layouts (Dw is column-major/strided for both X and W)
        self.a_smem_layout_staged = cute.make_layout(
            ((self.BLOCK_M, 3), self.BLOCK_K_STAGE, self.ab_stage),
            stride=((1, self.BLOCK_M * self.BLOCK_K_STAGE), self.BLOCK_M, 3 * self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_N, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        self.a_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, (self.BLOCK_M, 3), self.ab_stage),
            stride=(self.BLOCK_M, (1, self.BLOCK_M * self.BLOCK_K_STAGE), 3 * self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        self.BLOCK_N_padded = self.BLOCK_N + 8
        self.epi_smem_layout = cute.make_layout(
            (3 * self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N_padded, 1)
        )
        
        sX_bytes = self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 3 * 2
        sW_bytes = self.ab_stage * self.BLOCK_N * self.BLOCK_K_STAGE * 2
        mainloop_bytes = sX_bytes + sW_bytes
        epilogue_bytes = 3 * self.BLOCK_M * self.BLOCK_N_padded * 2  # 2 bytes per element (BFloat16)
        
        self.max_smem_bytes = max(mainloop_bytes, epilogue_bytes)
        self.flat_smem_elements = cute.ceil_div(self.max_smem_bytes, 4)
        self.sW_offset_elements = sX_bytes // 4
        
        @cute.struct
        class SharedStorageBwdDw:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageBwdDw
        
        # TiledMMA Configuration
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        if self.atom_layout_mnk is not None:
            atom_layout_mnk = self.atom_layout_mnk
        else:
            if self.num_warps == 4:
                atom_layout_mnk = (2, 2, 1)
            elif self.num_warps == 2:
                atom_layout_mnk = (2, 1, 1)
            elif self.num_warps == 6:
                atom_layout_mnk = (3, 2, 1)
            elif self.num_warps == 8:
                atom_layout_mnk = (4, 2, 1)
            elif self.num_warps == 12:
                atom_layout_mnk = (4, 3, 1)
            elif self.num_warps == 16:
                atom_layout_mnk = (4, 4, 1)
            else:
                atom_layout_mnk = (self.num_warps, 1, 1)
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=atom_layout_mnk)
        
        # Copy atom for BFloat16 accumulator writeback
        copy_op_r2s = cute.nvUniversalCopyOp() if hasattr(cute, "nvUniversalCopyOp") else cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s_bf16 = cute.make_copy_atom(copy_op_r2s, BFloat16)

    @cute.jit
    def __call__(
        self,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_v: cute.Tensor,
        x: cute.Tensor,
        grad_weight: cute.Tensor,
        stream
    ):
        self._setup()
        
        num_threads = self.num_warps * 32
        
        gB = cute.make_tensor(x.iterator, cute.make_layout((x.shape[0], self.N_dim), stride=(self.N_dim, 1)))
        
        # Create 3D global layouts for split gradients A
        gA_q = cute.make_tensor(
            grad_q.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.q_strides[1], self.q_strides[2], self.q_strides[3])
            )
        )
        gA_k = cute.make_tensor(
            grad_k.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.k_strides[1], self.k_strides[2], self.k_strides[3])
            )
        )
        gA_v = cute.make_tensor(
            grad_v.iterator,
            cute.make_layout(
                (self.B * self.NUM_HEADS, self.N, self.HEAD_DIM),
                stride=(self.v_strides[1], self.v_strides[2], self.v_strides[3])
            )
        )
        
        # Make TMA load descriptors
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_N), stride=(self.BLOCK_N, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_N)
        )
        
        smem_layout_a_comp = cute.make_layout(
            (1, self.BLOCK_K_STAGE, self.BLOCK_M),
            stride=(self.BLOCK_K_STAGE * self.BLOCK_M, self.BLOCK_M, 1)
        )
        
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_q,
            smem_layout_a_comp,
            (1, self.BLOCK_K_STAGE, self.BLOCK_M)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_k,
            smem_layout_a_comp,
            (1, self.BLOCK_K_STAGE, self.BLOCK_M)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_v,
            smem_layout_a_comp,
            (1, self.BLOCK_K_STAGE, self.BLOCK_M)
        )
        
        M_dim_single = self.M_dim // 3
        swizzle_x = self.swizzle_x
        grid_x_blocks = cute.ceil_div(M_dim_single, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            grad_q, grad_k, grad_v, x, grad_weight,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.a_smem_layout_staged_row,
            self.b_smem_layout_staged_row,
            self.epi_smem_layout,
            self.copy_atom_r2s_bf16,
            tma_atom_b, tma_tensor_b,
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v
        ).launch(
            grid=[swizzled_grid_x, cute.ceil_div(self.N_dim, self.BLOCK_N), self.SPLIT_K],
            block=[num_threads, 1, 1],
            cluster=[2, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_v: cute.Tensor,
        x: cute.Tensor,
        grad_weight: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        a_smem_layout_row: cute.Layout,
        b_smem_layout_row: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s_bf16: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_tensor_b: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = self.num_warps * 32
        
        # 2D Grid Swizzling with compile-time constant group size
        swizzle_x = self.swizzle_x
        grid_dim_y_const = self.N_dim // self.BLOCK_N
        group_size = swizzle_x * grid_dim_y_const
        
        pid = block_idx_y * grid_dim_x + block_idx_x
        swizzle_group = pid // group_size
        offset_in_group = pid % group_size
        
        cta_m_raw = swizzle_group * swizzle_x + (offset_in_group % swizzle_x)
        cta_n = offset_in_group // swizzle_x
        
        M_dim_single = self.M_dim // 3
        actual_grid_dim_x = cute.ceil_div(M_dim_single, self.BLOCK_M)
        is_valid = cta_m_raw < actual_grid_dim_x
        cta_m = cutlass.min(cta_m_raw, actual_grid_dim_x - 1)
        
        B = self.B
        N = self.N
        K_dim = B * N
        N_dim = self.N_dim
        
        k_tile_cnt = K_dim // self.BLOCK_K_STAGE
        k_tiles_per_slice = (k_tile_cnt + self.SPLIT_K - 1) // self.SPLIT_K
        k_tile_start = block_idx_z * k_tiles_per_slice
        
        # Prefetch TMA descriptors
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        sX_aligned_row = cute.make_tensor(sX_aligned.iterator, a_smem_layout_row)
        sW_aligned_row = cute.make_tensor(sW_aligned.iterator, b_smem_layout_row)
        
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Partition accumulator C (registers)
        smem_accum = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), epi_smem_layout)
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
        
        gB_tile = cute.local_tile(tma_tensor_b, (self.BLOCK_K_STAGE, self.BLOCK_N), (None, cta_n))
        copy_B, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_b,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gB_tile,
            dst_tensor=sW_aligned_row,
            single_stage=False
        )
        
        m_start = cta_m * self.BLOCK_M
        h = m_start // self.HEAD_DIM
        d_tile = (m_start % self.HEAD_DIM) // self.BLOCK_M
        comp_size_X = self.BLOCK_M * self.BLOCK_K_STAGE
        
        # Define Q, K, V tiles with None for dynamic coordinates
        gA_q_tile = cute.local_tile(tma_tensor_q, (1, self.BLOCK_K_STAGE, self.BLOCK_M), (None, None, d_tile))
        gA_k_tile = cute.local_tile(tma_tensor_k, (1, self.BLOCK_K_STAGE, self.BLOCK_M), (None, None, d_tile))
        gA_v_tile = cute.local_tile(tma_tensor_v, (1, self.BLOCK_K_STAGE, self.BLOCK_M), (None, None, d_tile))
        
        # Group dynamic modes of local tile to match TMA partition shape checks (rank 4)
        gA_q_tile_grouped = cute.group_modes(gA_q_tile, 3, 5)
        gA_k_tile_grouped = cute.group_modes(gA_k_tile, 3, 5)
        gA_v_tile_grouped = cute.group_modes(gA_v_tile, 3, 5)
        
        sX_q_aligned_3d = cute.make_tensor(
            sX_aligned_row.iterator,
            cute.make_layout(
                (1, self.BLOCK_K_STAGE, self.BLOCK_M, self.ab_stage),
                stride=(comp_size_X, self.BLOCK_M, 1, 3 * comp_size_X)
            )
        )
        sX_k_aligned_3d = cute.make_tensor(
            sX_aligned_row.iterator + comp_size_X,
            cute.make_layout(
                (1, self.BLOCK_K_STAGE, self.BLOCK_M, self.ab_stage),
                stride=(comp_size_X, self.BLOCK_M, 1, 3 * comp_size_X)
            )
        )
        sX_v_aligned_3d = cute.make_tensor(
            sX_aligned_row.iterator + 2 * comp_size_X,
            cute.make_layout(
                (1, self.BLOCK_K_STAGE, self.BLOCK_M, self.ab_stage),
                stride=(comp_size_X, self.BLOCK_M, 1, 3 * comp_size_X)
            )
        )
        
        copy_A_q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_q,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_q_tile_grouped,
            dst_tensor=sX_q_aligned_3d,
            single_stage=False
        )
        copy_A_k, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_k,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_k_tile_grouped,
            dst_tensor=sX_k_aligned_3d,
            single_stage=False
        )
        copy_A_v, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_v,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_v_tile_grouped,
            dst_tensor=sX_v_aligned_3d,
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
                k_curr = k_tile_start + stage
                b = (k_curr * self.BLOCK_K_STAGE) // N
                n_start = (k_curr * self.BLOCK_K_STAGE) % N
                n_tile = n_start // self.BLOCK_K_STAGE
                b_h = b * self.NUM_HEADS + (h % self.NUM_HEADS)
                
                if tidx < 32:
                    if tidx == 0:
                        tx_bytes = 3 * self.BLOCK_M * self.BLOCK_K_STAGE * 2 + self.BLOCK_K_STAGE * self.BLOCK_N * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                    cute.arch.sync_warp()
                    copy_A_q((b_h, n_tile), stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_A_k((b_h, n_tile), stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_A_v((b_h, n_tile), stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_B(k_curr, stage, tma_bar_ptr=mbar_ptr + stage)
                    
        # Wait for stage 0 and prefetch to registers
        if k_tiles_per_slice > 0:
            cute.arch.mbarrier_wait(mbar_ptr + 0, 0)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
            
        cute.arch.setmaxregister_increase(248)
        
        # Main execution loop
        for step in cutlass.range(k_tiles_per_slice):
            pipe_read = step % self.ab_stage
            pipe_write = (step + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (step + 1) % self.ab_stage
            
            next_gmem_read = step + (self.ab_stage - 1)
            if next_gmem_read < k_tiles_per_slice:
                k_curr = k_tile_start + next_gmem_read
                b = (k_curr * self.BLOCK_K_STAGE) // N
                n_start = (k_curr * self.BLOCK_K_STAGE) % N
                n_tile = n_start // self.BLOCK_K_STAGE
                b_h = b * self.NUM_HEADS + (h % self.NUM_HEADS)
                
                if tidx < 32:
                    if tidx == 0:
                        tx_bytes = 3 * self.BLOCK_M * self.BLOCK_K_STAGE * 2 + self.BLOCK_K_STAGE * self.BLOCK_N * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                    cute.arch.sync_warp()
                    copy_A_q((b_h, n_tile), pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_A_k((b_h, n_tile), pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_A_v((b_h, n_tile), pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B(k_curr, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    
            # Perform MMA immediately (overlaps with step 1 TMA load)
            if k_tile_start + step < k_tile_cnt:
                if step % 2 == 0:
                    cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
                else:
                    cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                    
            # Wait for next stage's TMA load and copy it from SMEM to registers
            if step + 1 < k_tiles_per_slice:
                next_phase = ((step + 1) // self.ab_stage) % 2
                cute.arch.mbarrier_wait(mbar_ptr + next_pipe_read, next_phase)
                cute.arch.sync_threads()
                    
                tCsA_next = tCsX[None, None, None, next_pipe_read]
                tCsB_next = tCsW[None, None, None, next_pipe_read]
                
                if step % 2 == 0:
                    cute.autovec_copy(tCsA_next, rA_1)
                    cute.autovec_copy(tCsB_next, rB_1)
                else:
                    cute.autovec_copy(tCsA_next, rA_0)
                    cute.autovec_copy(tCsB_next, rB_0)
                    
        cute.arch.sync_threads()
        cute.arch.setmaxregister_decrease(128)
        
        # Cast Float32 accumulators to BFloat16 in registers and write BFloat16 to shared memory
        accum_bf16 = cute.make_fragment_like(accum, BFloat16)
        for idx in cutlass.range(cute.size(accum), unroll_full=True):
            accum_bf16[idx] = BFloat16(accum[idx])
        cute.copy(copy_atom_r2s_bf16, accum_bf16, tCsC)
        cute.arch.sync_threads()
        
        # Thread mapping for coalesced stores (supports BLOCK_N > num_threads)
        threads_col = cutlass.min(self.BLOCK_N, num_threads)
        threads_row = num_threads // threads_col
        
        t_row = tidx // threads_col
        t_col = tidx % threads_col
        
        cols_per_thread = self.BLOCK_N // threads_col
        rows_per_thread = (3 * self.BLOCK_M) // threads_row
        
        row_start = t_row * rows_per_thread
        col_start = t_col * cols_per_thread
        
        if is_valid:
            for r_idx in cutlass.range(rows_per_thread):
                for c_idx in cutlass.range(cols_per_thread):
                    curr_row_local = row_start + r_idx
                    comp_idx = curr_row_local // self.BLOCK_M
                    row_offset_within_comp = curr_row_local % self.BLOCK_M
                    
                    curr_row_global = comp_idx * M_dim_single + h * self.HEAD_DIM + d_tile * self.BLOCK_M + row_offset_within_comp
                    curr_col = cta_n * self.BLOCK_N + col_start + c_idx
                    
                    if curr_row_global < self.M_dim and curr_col < N_dim:
                        val = smem_accum[curr_row_local, col_start + c_idx]
                        g_ptr = grad_weight.iterator + curr_row_global * N_dim + curr_col
                        if self.SPLIT_K > 1:
                            g_ptr_workspace = grad_weight.iterator + (block_idx_z * self.M_dim + curr_row_global) * N_dim + curr_col
                            g_ptr_workspace[0] = Float32(val)
                        else:
                            if const_expr(self.accumulate):
                                g_ptr[0] = BFloat16(Float32(g_ptr[0]) + Float32(val))
                            else:
                                g_ptr[0] = val

class SplitKReductionCute:
    def __init__(self, M_dim: int, N_dim: int, SPLIT_K: int, accumulate: bool = False):
        self.M_dim = M_dim
        self.N_dim = N_dim
        self.SPLIT_K = SPLIT_K
        self.BLOCK_M = 16
        self.BLOCK_N = 64
        self.accumulate = accumulate
        
    @cute.jit
    def __call__(self, workspace: cute.Tensor, out: cute.Tensor, stream):
        num_threads = 256
        self.kernel(workspace, out).launch(
            grid=[cute.ceil_div(self.M_dim, self.BLOCK_M), cute.ceil_div(self.N_dim, self.BLOCK_N), 1],
            block=[num_threads, 1, 1],
            stream=stream
        )
        
    @cute.kernel
    def kernel(self, workspace: cute.Tensor, out: cute.Tensor):
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        tidx, _, _ = cute.arch.thread_idx()
        
        workspace_ptr = workspace.iterator
        out_ptr = out.iterator
        
        for step in cutlass.range_constexpr(4):
            linear_idx = tidx + step * 256
            r = linear_idx // self.BLOCK_N
            c = linear_idx % self.BLOCK_N
            
            row = block_idx_x * self.BLOCK_M + r
            col = block_idx_y * self.BLOCK_N + c
            
            if row < self.M_dim and col < self.N_dim:
                val = Float32(0.0)
                for k in cutlass.range(self.SPLIT_K):
                    val += workspace_ptr[k * self.M_dim * self.N_dim + row * self.N_dim + col]
                if const_expr(self.accumulate):
                    out_ptr[row * self.N_dim + col] = BFloat16(Float32(out_ptr[row * self.N_dim + col]) + val)
                else:
                    out_ptr[row * self.N_dim + col] = BFloat16(val)

_workspace_cache = {}

def get_workspace_tensor(shape, dtype, device):
    key = (shape, dtype, device)
    if key not in _workspace_cache:
        _workspace_cache[key] = torch.zeros(shape, dtype=dtype, device=device)
    else:
        _workspace_cache[key].zero_()
    return _workspace_cache[key]

_reduction_compile_cache = {}

def get_reduction_kernel(M_dim, N_dim, SPLIT_K, accumulate=False):
    key = (M_dim, N_dim, SPLIT_K, accumulate)
    if key not in _reduction_compile_cache:
        op = SplitKReductionCute(M_dim, N_dim, SPLIT_K, accumulate=accumulate)
        workspace_fake = make_fake_tensor(Float32, (SPLIT_K, M_dim, N_dim), 16)
        out_fake = make_fake_tensor(BFloat16, (M_dim, N_dim), 16)
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled_fn = cute.compile(
            op,
            workspace_fake, out_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3",
        )
        _reduction_compile_cache[key] = compiled_fn
    return _reduction_compile_cache[key]


_dx_compile_cache = {}
_dw_compile_cache = {}

def get_matmul_bwd_dx_kernel(num_warps=4, BLOCK_M=64, BLOCK_N=64, BLOCK_K_STAGE=32, ab_stage=3, B=32, N=512, K_dim=3072, N_dim=512, NUM_HEADS=16, HEAD_DIM=64, q_strides=None, k_strides=None, v_strides=None, maxrregcount=255, swizzle_x=8, atom_layout_mnk=None):
    if isinstance(atom_layout_mnk, list):
        atom_layout_mnk = tuple(atom_layout_mnk)
    key = (num_warps, BLOCK_M, BLOCK_N, BLOCK_K_STAGE, ab_stage, B, N, K_dim, N_dim, NUM_HEADS, HEAD_DIM, tuple(q_strides), tuple(k_strides), tuple(v_strides), maxrregcount, swizzle_x, atom_layout_mnk)
    if key not in _dx_compile_cache:
        op = GemmBwdDxCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K_STAGE=BLOCK_K_STAGE,
            ab_stage=ab_stage,
            B=B,
            N=N,
            K_dim=K_dim,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=q_strides,
            k_strides=k_strides,
            v_strides=v_strides,
            swizzle_x=swizzle_x,
            atom_layout_mnk=atom_layout_mnk
        )
        
        grad_q_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        grad_k_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        grad_v_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        weight_fake = make_fake_tensor(BFloat16, (K_dim, N_dim), 16)
        grad_x_fake = make_fake_tensor(BFloat16, (B * N, N_dim), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fn = cute.compile(
            op,
            grad_q_fake, grad_k_fake, grad_v_fake, weight_fake, grad_x_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _dx_compile_cache[key] = compiled_fn
    return _dx_compile_cache[key]

def get_matmul_bwd_dw_kernel(num_warps=4, BLOCK_M=64, BLOCK_N=64, BLOCK_K_STAGE=32, ab_stage=3, B=32, N=512, M_dim=3072, K_dim=512, N_dim=512, NUM_HEADS=16, HEAD_DIM=64, q_strides=None, k_strides=None, v_strides=None, maxrregcount=255, SPLIT_K=1, swizzle_x=8, accumulate=False, atom_layout_mnk=None):
    if isinstance(atom_layout_mnk, list):
        atom_layout_mnk = tuple(atom_layout_mnk)
    key = (num_warps, BLOCK_M, BLOCK_N, BLOCK_K_STAGE, ab_stage, B, N, M_dim, K_dim, N_dim, NUM_HEADS, HEAD_DIM, tuple(q_strides), tuple(k_strides), tuple(v_strides), maxrregcount, SPLIT_K, swizzle_x, accumulate, atom_layout_mnk)
    if key not in _dw_compile_cache:
        print("DEBUG JIT COMPILE: get_matmul_bwd_dw_kernel called")
        print(f"DEBUG: K_dim = {K_dim}, type = {type(K_dim)}")
        print(f"DEBUG: N_dim = {N_dim}, type = {type(N_dim)}")
        print(f"DEBUG: M_dim = {M_dim}, type = {type(M_dim)}")
        
        op = GemmBwdDwCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K_STAGE=BLOCK_K_STAGE,
            ab_stage=ab_stage,
            B=B,
            N=N,
            M_dim=M_dim,
            K_dim=K_dim,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=q_strides,
            k_strides=k_strides,
            v_strides=v_strides,
            SPLIT_K=SPLIT_K,
            swizzle_x=swizzle_x,
            accumulate=accumulate,
            atom_layout_mnk=atom_layout_mnk
        )
        print(f"DEBUG: op.K_dim set to = {op.K_dim}, type = {type(op.K_dim)}")
        
        grad_q_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        grad_k_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        grad_v_fake = make_fake_tensor(BFloat16, (B, NUM_HEADS, N, HEAD_DIM), 16)
        x_fake = make_fake_tensor(BFloat16, (K_dim, N_dim), 16)
        grad_weight_dtype = BFloat16 if SPLIT_K == 1 else Float32
        grad_weight_fake = make_fake_tensor(grad_weight_dtype, (SPLIT_K * M_dim, N_dim), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fn = cute.compile(
            op,
            grad_q_fake, grad_k_fake, grad_v_fake, x_fake, grad_weight_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _dw_compile_cache[key] = compiled_fn
    return _dw_compile_cache[key]

_bwd_dx_kernel_cache = {}
_bwd_dw_kernel_cache = {}

@torch.library.custom_op("flash_prologue::matmul_bwd_dx_cute", mutates_args=())
def matmul_bwd_dx(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    grad_q = _ensure_contiguous(grad_q)
    grad_k = _ensure_contiguous(grad_k)
    grad_v = _ensure_contiguous(grad_v)
    weight = _ensure_contiguous(weight)
    B = int(grad_q.shape[0])
    NUM_HEADS = int(grad_q.shape[1])
    N = int(grad_q.shape[2])
    HEAD_DIM = int(grad_q.shape[3])
    M_dim = B * N
    K_dim = 3 * NUM_HEADS * HEAD_DIM
    N_dim = int(weight.shape[1])
    
    if grad_q.numel() == 0 or weight.numel() == 0:
        return torch.zeros((M_dim, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        
    grad_x = torch.empty((M_dim, N_dim), dtype=torch.bfloat16, device=grad_q.device)
    
    shape_key = (B, N)
    kernel = _bwd_dx_kernel_cache.get(shape_key)
    if kernel is None:
        config = get_autotuned_config(M_dim, N_dim, K_dim, "bwd_dx", device=str(grad_q.device))
        
        block_m = config["BLOCK_M"]
        block_n = config["BLOCK_N"]
        block_k = config["BLOCK_K"]
        ab_stage = config.get("ab_stage", 3)
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        swizzle_x = config.get("swizzle_x", 8)
        
        static_stride_h = N * HEAD_DIM
        static_stride_n = HEAD_DIM
        static_stride_d = 1
        static_strides = (None, static_stride_h, static_stride_n, static_stride_d)
        
        kernel = get_matmul_bwd_dx_kernel(
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K_STAGE=block_k,
            ab_stage=ab_stage,
            B=B,
            N=N,
            K_dim=K_dim,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=maxrregcount,
            swizzle_x=swizzle_x,
            atom_layout_mnk=config.get("atom_layout_mnk", None)
        )
        _bwd_dx_kernel_cache[shape_key] = kernel
    
    kernel(grad_q, grad_k, grad_v, weight, grad_x)
    return grad_x

@matmul_bwd_dx.register_fake
def _matmul_bwd_dx_fake(grad_q, grad_k, grad_v, weight):
    B, _, N, _ = grad_q.shape
    M_dim = B * N
    _, N_dim = weight.shape
    if grad_q.numel() == 0 or weight.numel() == 0:
        return torch.zeros((M_dim, N_dim), dtype=grad_q.dtype, device=grad_q.device)
    return torch.empty((M_dim, N_dim), dtype=grad_q.dtype, device=grad_q.device)

@torch.library.custom_op("flash_prologue::matmul_bwd_dw_cute", mutates_args=())
def _matmul_bwd_dw_cute_op(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    grad_q = _ensure_contiguous(grad_q)
    grad_k = _ensure_contiguous(grad_k)
    grad_v = _ensure_contiguous(grad_v)
    x = _ensure_contiguous(x)
    B = int(grad_q.shape[0])
    NUM_HEADS = int(grad_q.shape[1])
    N = int(grad_q.shape[2])
    HEAD_DIM = int(grad_q.shape[3])
    M_dim = 3 * NUM_HEADS * HEAD_DIM
    K_dim = B * N
    N_dim = int(x.shape[2] if x.dim() == 3 else x.shape[1]) # handles both 3D [B, N, D] and 2D [B*N, D]
    
    if grad_q.numel() == 0 or x.numel() == 0:
        return torch.zeros((M_dim, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        
    x_flat = x.view(K_dim, N_dim)
    
    shape_key = (B, N, False)
    cached_val = _bwd_dw_kernel_cache.get(shape_key)
    if cached_val is None:
        config = get_autotuned_config(M_dim, N_dim, K_dim, "bwd_dw", device=str(grad_q.device))
        
        block_m = config["BLOCK_M"]
        block_n = config["BLOCK_N"]
        block_k = config["BLOCK_K"]
        ab_stage = config.get("ab_stage", 2)
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        split_k = config.get("SPLIT_K", 1)
        swizzle_x = config.get("swizzle_x", 8)
        
        static_stride_h = N * HEAD_DIM
        static_stride_n = HEAD_DIM
        static_stride_d = 1
        static_strides = (None, static_stride_h, static_stride_n, static_stride_d)
        
        kernel = get_matmul_bwd_dw_kernel(
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K_STAGE=block_k,
            ab_stage=ab_stage,
            B=B,
            N=N,
            M_dim=M_dim,
            K_dim=K_dim,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=maxrregcount,
            SPLIT_K=split_k,
            swizzle_x=swizzle_x,
            accumulate=False,
            atom_layout_mnk=config.get("atom_layout_mnk", None)
        )
        _bwd_dw_kernel_cache[shape_key] = (kernel, split_k)
    else:
        kernel, split_k = cached_val
    
    grad_weight = torch.empty((M_dim, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        
    if split_k > 1:
        # Use cached/pre-allocated workspace to prevent dynamic memory allocation overhead
        workspace = get_workspace_tensor((split_k, M_dim, N_dim), dtype=torch.float32, device=grad_q.device)
        workspace_flat = workspace.view(split_k * M_dim, N_dim)
        kernel(grad_q, grad_k, grad_v, x_flat, workspace_flat)
        
        # Use on-chip JIT split-K reduction kernel
        reduction_kernel = get_reduction_kernel(M_dim, N_dim, split_k, accumulate=False)
        reduction_kernel(workspace, grad_weight)
        return grad_weight
    else:
        # For split_k == 1, write directly to final output tensor
        kernel(grad_q, grad_k, grad_v, x_flat, grad_weight)
        return grad_weight

@_matmul_bwd_dw_cute_op.register_fake
def _matmul_bwd_dw_fake(grad_q, grad_k, grad_v, x):
    B, NUM_HEADS, N, HEAD_DIM = grad_q.shape
    M_dim = 3 * NUM_HEADS * HEAD_DIM
    N_dim = x.shape[2] if x.dim() == 3 else x.shape[1]
    if grad_q.numel() == 0 or x.numel() == 0:
        return torch.zeros((M_dim, N_dim), dtype=grad_q.dtype, device=grad_q.device)
    return torch.empty((M_dim, N_dim), dtype=grad_q.dtype, device=grad_q.device)

@torch.library.custom_op("flash_prologue::matmul_bwd_dw_accumulate_cute", mutates_args={"out"})
def _matmul_bwd_dw_accumulate_cute_op(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor
) -> None:
    grad_q = _ensure_contiguous(grad_q)
    grad_k = _ensure_contiguous(grad_k)
    grad_v = _ensure_contiguous(grad_v)
    x = _ensure_contiguous(x)
    B = int(grad_q.shape[0])
    NUM_HEADS = int(grad_q.shape[1])
    N = int(grad_q.shape[2])
    HEAD_DIM = int(grad_q.shape[3])
    M_dim = 3 * NUM_HEADS * HEAD_DIM
    K_dim = B * N
    N_dim = int(x.shape[2] if x.dim() == 3 else x.shape[1])
    
    if grad_q.numel() == 0 or x.numel() == 0:
        return
        
    x_flat = x.view(K_dim, N_dim)
    
    shape_key = (B, N, True)
    cached_val = _bwd_dw_kernel_cache.get(shape_key)
    if cached_val is None:
        config = get_autotuned_config(M_dim, N_dim, K_dim, "bwd_dw", device=str(grad_q.device))
        
        block_m = config["BLOCK_M"]
        block_n = config["BLOCK_N"]
        block_k = config["BLOCK_K"]
        ab_stage = config.get("ab_stage", 2)
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        split_k = config.get("SPLIT_K", 1)
        swizzle_x = config.get("swizzle_x", 8)
        
        static_stride_h = N * HEAD_DIM
        static_stride_n = HEAD_DIM
        static_stride_d = 1
        static_strides = (None, static_stride_h, static_stride_n, static_stride_d)
        
        kernel = get_matmul_bwd_dw_kernel(
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K_STAGE=block_k,
            ab_stage=ab_stage,
            B=B,
            N=N,
            M_dim=M_dim,
            K_dim=K_dim,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=maxrregcount,
            SPLIT_K=split_k,
            swizzle_x=swizzle_x,
            accumulate=True
        )
        _bwd_dw_kernel_cache[shape_key] = (kernel, split_k)
    else:
        kernel, split_k = cached_val
        
    if split_k > 1:
        # Use cached/pre-allocated workspace to prevent dynamic memory allocation overhead
        workspace = get_workspace_tensor((split_k, M_dim, N_dim), dtype=torch.float32, device=grad_q.device)
        workspace_flat = workspace.view(split_k * M_dim, N_dim)
        kernel(grad_q, grad_k, grad_v, x_flat, workspace_flat)
        
        # Use on-chip JIT split-K reduction kernel
        reduction_kernel = get_reduction_kernel(M_dim, N_dim, split_k, accumulate=True)
        reduction_kernel(workspace, out)
    else:
        # For split_k == 1, write directly to final output tensor
        kernel(grad_q, grad_k, grad_v, x_flat, out)

@_matmul_bwd_dw_accumulate_cute_op.register_fake
def _matmul_bwd_dw_accumulate_fake(grad_q, grad_k, grad_v, x, out):
    pass

def matmul_bwd_dw(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        return torch.ops.flash_prologue.matmul_bwd_dw_cute(grad_q, grad_k, grad_v, x)
    else:
        torch.ops.flash_prologue.matmul_bwd_dw_accumulate_cute(grad_q, grad_k, grad_v, x, out)
        return out

@torch.library.custom_op("flash_prologue::fused_qkv_backward", mutates_args=())
def fused_qkv_backward(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_q = _ensure_contiguous(grad_q)
    grad_k = _ensure_contiguous(grad_k)
    grad_v = _ensure_contiguous(grad_v)
    weight = _ensure_contiguous(weight)
    x = _ensure_contiguous(x)
    B = int(grad_q.shape[0])
    NUM_HEADS = int(grad_q.shape[1])
    N = int(grad_q.shape[2])
    HEAD_DIM = int(grad_q.shape[3])
    M_dim_dx = B * N
    K_dim_dx = 3 * NUM_HEADS * HEAD_DIM
    N_dim = int(weight.shape[1])
    
    if grad_q.numel() == 0 or x.numel() == 0 or weight.numel() == 0:
        grad_x = torch.zeros((B, N, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        grad_weight = torch.zeros((K_dim_dx, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        return grad_x, grad_weight
        
    # 1. Activation Gradients (dx)
    # Directly allocate 3D output tensor in-place
    grad_x = torch.empty((B, N, N_dim), dtype=torch.bfloat16, device=grad_q.device)
    
    static_stride_h = N * HEAD_DIM
    static_stride_n = HEAD_DIM
    static_stride_d = 1
    static_strides = (None, static_stride_h, static_stride_n, static_stride_d)
    
    shape_key = (B, N)
    dx_kernel = _bwd_dx_kernel_cache.get(shape_key)
    if dx_kernel is None:
        config_dx = get_autotuned_config(M_dim_dx, N_dim, K_dim_dx, "bwd_dx", device=str(grad_q.device))
        
        dx_kernel = get_matmul_bwd_dx_kernel(
            num_warps=config_dx.get("num_warps", 4),
            BLOCK_M=config_dx["BLOCK_M"],
            BLOCK_N=config_dx["BLOCK_N"],
            BLOCK_K_STAGE=config_dx["BLOCK_K"],
            ab_stage=config_dx.get("ab_stage", 2),
            B=B,
            N=N,
            K_dim=K_dim_dx,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=config_dx.get("maxrregcount", 255),
            swizzle_x=config_dx.get("swizzle_x", 8)
        )
        _bwd_dx_kernel_cache[shape_key] = dx_kernel
        
    dx_kernel(grad_q, grad_k, grad_v, weight, grad_x.view(M_dim_dx, N_dim))
    
    # 2. Weight Gradients (dw)
    M_dim_dw = 3 * NUM_HEADS * HEAD_DIM
    K_dim_dw = B * N
    
    shape_key_dw = (B, N, False)
    cached_dw = _bwd_dw_kernel_cache.get(shape_key_dw)
    if cached_dw is None:
        config_dw = get_autotuned_config(M_dim_dw, N_dim, K_dim_dw, "bwd_dw", device=str(grad_q.device))
        split_k = config_dw.get("SPLIT_K", 1)
        
        dw_kernel = get_matmul_bwd_dw_kernel(
            num_warps=config_dw.get("num_warps", 4),
            BLOCK_M=config_dw["BLOCK_M"],
            BLOCK_N=config_dw["BLOCK_N"],
            BLOCK_K_STAGE=config_dw["BLOCK_K"],
            ab_stage=config_dw.get("ab_stage", 2),
            B=B,
            N=N,
            M_dim=M_dim_dw,
            K_dim=K_dim_dw,
            N_dim=N_dim,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            q_strides=static_strides,
            k_strides=static_strides,
            v_strides=static_strides,
            maxrregcount=config_dw.get("maxrregcount", 255),
            SPLIT_K=split_k,
            swizzle_x=config_dw.get("swizzle_x", 8)
        )
        _bwd_dw_kernel_cache[shape_key_dw] = (dw_kernel, split_k)
    else:
        dw_kernel, split_k = cached_dw
    
    x_flat = x.view(K_dim_dw, N_dim)
    
    if split_k > 1:
        # Use cached/pre-allocated workspace to prevent dynamic memory allocation overhead
        workspace = get_workspace_tensor((split_k, M_dim_dw, N_dim), dtype=torch.float32, device=grad_q.device)
        workspace_flat = workspace.view(split_k * M_dim_dw, N_dim)
        dw_kernel(grad_q, grad_k, grad_v, x_flat, workspace_flat)
        
        grad_weight = torch.empty((M_dim_dw, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        reduction_kernel = get_reduction_kernel(M_dim_dw, N_dim, split_k)
        reduction_kernel(workspace, grad_weight)
    else:
        # For split_k == 1, write directly to final output tensor
        grad_weight = torch.empty((M_dim_dw, N_dim), dtype=torch.bfloat16, device=grad_q.device)
        dw_kernel(grad_q, grad_k, grad_v, x_flat, grad_weight)
        
    return grad_x, grad_weight

@fused_qkv_backward.register_fake
def _fused_qkv_backward_fake(grad_q, grad_k, grad_v, weight, x):
    B, NUM_HEADS, N, HEAD_DIM = grad_q.shape
    K_dim_dx = 3 * NUM_HEADS * HEAD_DIM
    N_dim = weight.shape[1]
    if grad_q.numel() == 0 or x.numel() == 0 or weight.numel() == 0:
        grad_x = torch.zeros((B, N, N_dim), dtype=grad_q.dtype, device=grad_q.device)
        grad_weight = torch.zeros((K_dim_dx, N_dim), dtype=grad_q.dtype, device=grad_q.device)
        return grad_x, grad_weight
    grad_x = torch.empty((B, N, N_dim), dtype=grad_q.dtype, device=grad_q.device)
    grad_weight = torch.empty((K_dim_dx, N_dim), dtype=grad_q.dtype, device=grad_q.device)
    return grad_x, grad_weight

