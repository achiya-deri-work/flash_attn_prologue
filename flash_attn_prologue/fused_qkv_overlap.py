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

class FusedQKVOverlap:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_K_STAGE: int = 32, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 2, hidden: int = 512, use_sync_in_loop: bool = True, ab_stage: int = None):
        self.num_warps = num_warps  # 4 warps (128 threads)
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.BLOCK_N = HEAD_GROUP * 3 * HEAD_DIM  # e.g., 192 for 1 head group (Q, K, V)
        self.BLOCK_K_STAGE = BLOCK_K_STAGE
        self.NUM_HEADS = NUM_HEADS
        self.hidden = hidden
        self.k_tile_cnt = hidden // BLOCK_K_STAGE
        self.use_sync_in_loop = use_sync_in_loop
        
        # Calculate shared memory footprint for 3 stages dynamically
        sX_elements_3 = self.BLOCK_M * self.BLOCK_K_STAGE * 3
        sW_elements_3 = self.BLOCK_N * self.BLOCK_K_STAGE * 3
        sBuf_elements = self.BLOCK_M * self.BLOCK_N
        mainloop_bytes_3 = (sX_elements_3 + sW_elements_3) * 2
        epilogue_bytes = sBuf_elements * 4
        
        if ab_stage is not None:
            self.ab_stage = ab_stage
        else:
            # Blackwell hardware SMEM limit is 99 KB
            if max(mainloop_bytes_3, epilogue_bytes) <= 99 * 1024:
                self.ab_stage = 3
            else:
                self.ab_stage = 2
        
    def _setup(self):
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(self.BLOCK_K_STAGE, 1, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        # Physical layout of sW: non-interleaved grouped layout
        # (3 components: Q, K, V are stored contiguously in SMEM)
        self.b_smem_layout_staged = cute.make_layout(
            ((self.HEAD_DIM, 3, self.HEAD_GROUP), self.BLOCK_K_STAGE, self.ab_stage),
            stride=((self.BLOCK_K_STAGE, self.HEAD_GROUP * self.HEAD_DIM * self.BLOCK_K_STAGE, self.HEAD_DIM * self.BLOCK_K_STAGE), 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        # Shared memory layout for epilogue writeback (Float32 view of sBuf)
        # Pad self.BLOCK_N by 8 elements (16 bytes) to resolve LDS bank conflicts
        self.BLOCK_N_padded = self.BLOCK_N + 8
        self.epi_smem_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N_padded, 1)
        )
        
        # Calculate maximum shared memory size needed in bytes and elements
        mainloop_bytes = (self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.ab_stage * self.BLOCK_N * self.BLOCK_K_STAGE * 2)
        epilogue_bytes = self.BLOCK_M * self.BLOCK_N_padded * 4
        self.max_smem_bytes = max(mainloop_bytes, epilogue_bytes)
        self.flat_smem_elements = cute.ceil_div(self.max_smem_bytes, 4)
        
        # Byte and Float32 element offset for sW
        self.sW_offset_elements = (self.ab_stage * self.BLOCK_M * self.BLOCK_K_STAGE * 2) // 4
        
        @cute.struct
        class SharedStorageFused:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageFused
        
        # 3. TiledMMA configuration
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        if self.BLOCK_M == 16:
            atom_layout_mnk = (1, 4, 1)
        else:
            atom_layout_mnk = (2, 2, 1)
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=atom_layout_mnk)
        
        # 4. Copy atom for Float32 accumulator writeback
        copy_op_r2s = cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, Float32)
        
        # 5. TiledCopy configuration for direct-store epilogue
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
        x: cute.Tensor,
        weight_q: cute.Tensor,
        weight_k: cute.Tensor,
        weight_v: cute.Tensor,
        q_proj: cute.Tensor,
        k_proj: cute.Tensor,
        v_proj: cute.Tensor,
        stream
    ):
        self._setup()
        
        B = x.shape[0]
        N = x.shape[1]
        
        num_threads = self.num_warps * 32
        
        # 1. Create global layouts and tensors for TMA
        gX = cute.make_tensor(x.iterator, cute.make_layout((B * N, self.hidden), stride=(self.hidden, 1)))
        
        num_heads_x_dim = self.NUM_HEADS * self.HEAD_DIM
        gW_q = cute.make_tensor(weight_q.iterator, cute.make_layout((num_heads_x_dim, self.hidden), stride=(self.hidden, 1)))
        gW_k = cute.make_tensor(weight_k.iterator, cute.make_layout((num_heads_x_dim, self.hidden), stride=(self.hidden, 1)))
        gW_v = cute.make_tensor(weight_v.iterator, cute.make_layout((num_heads_x_dim, self.hidden), stride=(self.hidden, 1)))
        
        # 2. Make TMA load descriptors (atoms/tensors)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gX,
            cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)),
            (self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        
        # Staged SMEM layout for one component of W: shape (HEAD_GROUP * HEAD_DIM, BLOCK_K_STAGE)
        smem_layout_w_comp = cute.make_layout((self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1))
        
        tma_atom_wq, tma_tensor_wq = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gW_q,
            smem_layout_w_comp,
            (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE)
        )
        
        tma_atom_wk, tma_tensor_wk = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gW_k,
            smem_layout_w_comp,
            (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE)
        )
        
        tma_atom_wv, tma_tensor_wv = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gW_v,
            smem_layout_w_comp,
            (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE)
        )
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            x, weight_q, weight_k, weight_v,
            q_proj, k_proj, v_proj,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout,
            self.copy_atom_r2s,
            self.gmem_tiled_copy_store,
            tma_atom_x, tma_tensor_x,
            tma_atom_wq, tma_tensor_wq,
            tma_atom_wk, tma_tensor_wk,
            tma_atom_wv, tma_tensor_wv,
            Boolean(self.use_sync_in_loop)
        ).launch(
            grid=[swizzled_grid_x, B, self.NUM_HEADS // self.HEAD_GROUP],
            block=[num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight_q: cute.Tensor,
        weight_k: cute.Tensor,
        weight_v: cute.Tensor,
        q_proj: cute.Tensor,
        k_proj: cute.Tensor,
        v_proj: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s: cute.CopyAtom,
        gmem_tiled_copy_store: cute.TiledCopy,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_wq: cute.CopyAtom,
        tma_tensor_wq: cute.Tensor,
        tma_atom_wk: cute.CopyAtom,
        tma_tensor_wk: cute.Tensor,
        tma_atom_wv: cute.CopyAtom,
        tma_tensor_wv: cute.Tensor,
        use_sync_in_loop: Boolean
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        grid_dim_z = cute.arch.grid_dim()[2]
        
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        
        # 2D Grid Swizzling with SWIZZLE_X = 8
        swizzle_x = 8
        N = x.shape[1]
        actual_grid_dim_x = cute.ceil_div(N, self.BLOCK_M)
        
        pid = block_idx_z * grid_dim_x + block_idx_x
        swizzle_group = pid // (swizzle_x * grid_dim_z)
        offset_in_group = pid % (swizzle_x * grid_dim_z)
        
        cta_m_raw = swizzle_group * swizzle_x + (offset_in_group % swizzle_x)
        cta_h_group = offset_in_group // swizzle_x
        cta_b = block_idx_y
        
        is_valid = cta_m_raw < actual_grid_dim_x
        cta_m = cutlass.min(cta_m_raw, actual_grid_dim_x - 1)
        
        tidx, _, _ = cute.arch.thread_idx()
        
        # Prefetch TMA descriptors on thread 0
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_x)
            cpasync.prefetch_descriptor(tma_atom_wq)
            cpasync.prefetch_descriptor(tma_atom_wk)
            cpasync.prefetch_descriptor(tma_atom_wv)
            
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        
        # Local_tile partition global tensors
        N = x.shape[1]
        hidden = x.shape[2]
        k_tile_cnt = hidden // self.BLOCK_K_STAGE
        
        # Partition global tensors inside the kernel using local_tile and tma_get_copy_fn
        N_blocks = cute.ceil_div(N, self.BLOCK_M)
        gX_tile = cute.local_tile(tma_tensor_x, (self.BLOCK_M, self.BLOCK_K_STAGE), (cta_b * N_blocks + cta_m, None))
        copy_X, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_x,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gX_tile,
            dst_tensor=sX_aligned,
            single_stage=False
        )
        
        gW_q_tile = cute.local_tile(tma_tensor_wq, (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE), (cta_h_group, None))
        gW_k_tile = cute.local_tile(tma_tensor_wk, (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE), (cta_h_group, None))
        gW_v_tile = cute.local_tile(tma_tensor_wv, (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE), (cta_h_group, None))
        
        comp_size = self.HEAD_GROUP * self.HEAD_DIM * self.BLOCK_K_STAGE
        sW_q_layout = cute.make_layout(
            (self.HEAD_GROUP * self.HEAD_DIM, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(self.BLOCK_K_STAGE, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        sW_q_aligned = cute.make_tensor(sW_aligned.iterator, sW_q_layout)
        sW_k_aligned = cute.make_tensor(sW_aligned.iterator + comp_size, sW_q_layout)
        sW_v_aligned = cute.make_tensor(sW_aligned.iterator + 2 * comp_size, sW_q_layout)
        
        copy_W_q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_wq,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gW_q_tile,
            dst_tensor=sW_q_aligned,
            single_stage=False
        )
        copy_W_k, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_wk,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gW_k_tile,
            dst_tensor=sW_k_aligned,
            single_stage=False
        )
        copy_W_v, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_wv,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gW_v_tile,
            dst_tensor=sW_v_aligned,
            single_stage=False
        )
        
        mbar_ptr = cute.arch.alloc_smem(Int64, self.ab_stage)
        if tidx == 0:
            for s in cutlass.range_constexpr(self.ab_stage):
                cute.arch.mbarrier_init(mbar_ptr + s, 1)
            cute.arch.mbarrier_init_fence()
            
        cute.arch.sync_threads()
        
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Partition accumulator C (registers)
        smem_accum = cute.make_tensor(storage.sX_raw.data_ptr().align(16), epi_smem_layout)
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        # Partition SMEM stages outside the loop
        tCsX = thr_mma.partition_A(sX_aligned)
        tCsW = thr_mma.partition_B(sW_aligned)
        
        # Pre-allocate double-buffered register fragments
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        # Epilogue indexing computed early for prefetching
        r = tidx % self.BLOCK_M
        h_local = tidx // self.BLOCK_M
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        seq = cta_m * self.BLOCK_M + r
        
        # Prefetch initial stages (up to self.ab_stage - 1)
        for stage in cutlass.range_constexpr(self.ab_stage - 1):
            if stage < k_tile_cnt:
                if tidx < 32:
                    if tidx == 0:
                        tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                    cute.arch.sync_warp()
                    copy_X(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_W_q(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_W_k(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_W_v(stage, stage, tma_bar_ptr=mbar_ptr + stage)
                
        # Wait for the first stage (stage 0) to be ready in shared memory and prefetch it to rA_0/rB_0 (vectorized)
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
                    if tidx == 0:
                        tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                    cute.arch.sync_warp()
                    copy_X(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_W_q(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_W_k(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_W_v(next_gmem_read, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                
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
                    
            # Perform MMA using current stage registers
            if k_tile % 2 == 0:
                cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
            else:
                cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
            
        # Write Float32 accumulator to sBuf
        cute.copy(copy_atom_r2s, accum, tCsC)
        cute.arch.sync_threads()

        sOut_bf16 = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.BLOCK_N), stride=(self.BLOCK_N_padded, 1)))
        
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        v_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        row_offset = r * self.BLOCK_N_padded + h_local * (3 * self.HEAD_DIM)
        
        if seq < N and h_local < self.HEAD_GROUP:
            q_ptr = smem_accum.iterator + row_offset
            k_ptr = smem_accum.iterator + row_offset + self.HEAD_DIM
            v_ptr = smem_accum.iterator + row_offset + 2 * self.HEAD_DIM
            
            q_smem = cute.make_tensor(q_ptr, cute.make_layout((self.HEAD_DIM,)))
            k_smem = cute.make_tensor(k_ptr, cute.make_layout((self.HEAD_DIM,)))
            v_smem = cute.make_tensor(v_ptr, cute.make_layout((self.HEAD_DIM,)))
            
            cute.autovec_copy(q_smem, q_reg)
            cute.autovec_copy(k_smem, k_reg)
            cute.autovec_copy(v_smem, v_reg)
            
            # Recast and write directly to sOut_bf16 in BFloat16 format
            out_q_smem = cute.make_tensor(sOut_bf16.iterator + row_offset, cute.make_layout((self.HEAD_DIM,)))
            out_k_smem = cute.make_tensor(sOut_bf16.iterator + row_offset + self.HEAD_DIM, cute.make_layout((self.HEAD_DIM,)))
            out_v_smem = cute.make_tensor(sOut_bf16.iterator + row_offset + 2 * self.HEAD_DIM, cute.make_layout((self.HEAD_DIM,)))
            
            q_bf16 = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            k_bf16 = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            v_bf16 = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                q_bf16[j] = BFloat16(q_reg[j])
                k_bf16[j] = BFloat16(k_reg[j])
                v_bf16[j] = BFloat16(v_reg[j])
                
            cute.autovec_copy(q_bf16, out_q_smem)
            cute.autovec_copy(k_bf16, out_k_smem)
            cute.autovec_copy(v_bf16, out_v_smem)
            
        cute.arch.sync_threads()
        
        thr_copy_store = gmem_tiled_copy_store.get_slice(tidx)
        
        if is_valid:
            if (cta_m + 1) * self.BLOCK_M <= N:
                for h_idx in cutlass.range_constexpr(self.HEAD_GROUP):
                    global_h = self.HEAD_GROUP * cta_h_group + h_idx
                    if global_h < self.NUM_HEADS:
                        head_offset = h_idx * (3 * self.HEAD_DIM)
                        
                        sOut_h_q = cute.make_tensor((sOut_bf16.iterator + head_offset).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        sOut_h_k = cute.make_tensor((sOut_bf16.iterator + head_offset + self.HEAD_DIM).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        sOut_h_v = cute.make_tensor((sOut_bf16.iterator + head_offset + 2 * self.HEAD_DIM).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        
                        g_offset = cta_b * (self.NUM_HEADS * N * self.HEAD_DIM) + global_h * (N * self.HEAD_DIM) + (cta_m * self.BLOCK_M) * self.HEAD_DIM
                        
                        g_tile_q = cute.make_tensor((q_proj.iterator + g_offset).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                        g_tile_k = cute.make_tensor((k_proj.iterator + g_offset).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                        g_tile_v = cute.make_tensor((v_proj.iterator + g_offset).align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                        
                        tS_q = thr_copy_store.partition_S(sOut_h_q)
                        tD_q = thr_copy_store.partition_D(g_tile_q)
                        
                        tS_k = thr_copy_store.partition_S(sOut_h_k)
                        tD_k = thr_copy_store.partition_D(g_tile_k)
                        
                        tS_v = thr_copy_store.partition_S(sOut_h_v)
                        tD_v = thr_copy_store.partition_D(g_tile_v)
                        
                        cute.copy(gmem_tiled_copy_store, tS_q, tD_q)
                        cute.copy(gmem_tiled_copy_store, tS_k, tD_k)
                        cute.copy(gmem_tiled_copy_store, tS_v, tD_v)
            else:
                h_local_thread = tidx // self.BLOCK_M
                if h_local_thread < self.HEAD_GROUP:
                    row = tidx % self.BLOCK_M
                    seq = cta_m * self.BLOCK_M + row
                    if seq < N:
                        global_h = self.HEAD_GROUP * cta_h_group + h_local_thread
                        if global_h < self.NUM_HEADS:
                            g_offset_thread = cta_b * (self.NUM_HEADS * N * self.HEAD_DIM) + global_h * (N * self.HEAD_DIM) + seq * self.HEAD_DIM
                            
                            q_g = q_proj.iterator + g_offset_thread
                            k_g = k_proj.iterator + g_offset_thread
                            v_g = v_proj.iterator + g_offset_thread
                            
                            row_offset_thread = row * self.BLOCK_N_padded + h_local_thread * (3 * self.HEAD_DIM)
                            q_s = sOut_bf16.iterator + row_offset_thread
                            k_s = sOut_bf16.iterator + row_offset_thread + self.HEAD_DIM
                            v_s = sOut_bf16.iterator + row_offset_thread + 2 * self.HEAD_DIM
                            
                            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                                q_g[j] = q_s[j]
                                k_g[j] = k_s[j]
                                v_g[j] = v_s[j]

_overlap_compile_cache = {}

def get_fused_qkv_overlap_kernel(num_warps=4, BLOCK_M=64, BLOCK_K_STAGE=32, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=2, hidden=512, use_sync_in_loop=True, ab_stage=None, maxrregcount=255):
    key = (num_warps, BLOCK_M, BLOCK_K_STAGE, NUM_HEADS, HEAD_DIM, HEAD_GROUP, hidden, use_sync_in_loop, ab_stage, maxrregcount)
    if key not in _overlap_compile_cache:
        B_sym = cute.sym_int()
        N_sym = cute.sym_int()
        
        fwd_op = FusedQKVOverlap(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_K_STAGE=BLOCK_K_STAGE,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            HEAD_GROUP=HEAD_GROUP,
            hidden=hidden,
            use_sync_in_loop=use_sync_in_loop,
            ab_stage=ab_stage
        )
        
        x_fake = make_fake_tensor(BFloat16, (B_sym, N_sym, hidden), 16)
        weight_q_fake = make_fake_tensor(BFloat16, (NUM_HEADS * HEAD_DIM, hidden), 16)
        weight_k_fake = make_fake_tensor(BFloat16, (NUM_HEADS * HEAD_DIM, hidden), 16)
        weight_v_fake = make_fake_tensor(BFloat16, (NUM_HEADS * HEAD_DIM, hidden), 16)
        
        q_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        k_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        v_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            x_fake, weight_q_fake, weight_k_fake, weight_v_fake,
            q_proj_fake, k_proj_fake, v_proj_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _overlap_compile_cache[key] = compiled_fwd
    return _overlap_compile_cache[key]
 
@torch.library.custom_op("flash_prologue::fused_qkv_overlap", mutates_args=())
def fused_qkv_overlap_op(
    x: torch.Tensor,
    weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, hidden = x.shape
    num_heads = weight.shape[0] // 3 // 64
    head_dim = 64
    
    # Query Autotuner for optimal shape configs
    from flash_attn_prologue.autotune import get_autotuned_config
    config = get_autotuned_config(B * N, 3 * num_heads * head_dim, hidden, "overlap", device=str(x.device))
    
    block_m = config["BLOCK_M"]
    block_k = config["BLOCK_K"]
    head_group = config.get("HEAD_GROUP", 2)
    ab_stage = config.get("ab_stage", None)
    num_warps = config.get("num_warps", 4)
    maxrregcount = config.get("maxrregcount", 255)
    
    q_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    k_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    v_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    
    fwd_kernel = get_fused_qkv_overlap_kernel(
        num_warps=num_warps,
        BLOCK_M=block_m,
        BLOCK_K_STAGE=block_k,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        HEAD_GROUP=head_group,
        hidden=hidden,
        use_sync_in_loop=True,
        ab_stage=ab_stage,
        maxrregcount=maxrregcount
    )
    
    num_heads_x_dim = num_heads * head_dim
    weight_q = weight[0 : num_heads_x_dim, :]
    weight_k = weight[num_heads_x_dim : 2 * num_heads_x_dim, :]
    weight_v = weight[2 * num_heads_x_dim : 3 * num_heads_x_dim, :]
    
    fwd_kernel(
        x, weight_q, weight_k, weight_v,
        q_proj_bf16, k_proj_bf16, v_proj_bf16
    )
    
    return q_proj_bf16, k_proj_bf16, v_proj_bf16

@fused_qkv_overlap_op.register_fake
def _fused_qkv_overlap_fake(x, weight):
    B, N, hidden = x.shape
    num_heads = weight.shape[0] // 3 // 64
    head_dim = 64
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    v_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    return q_proj, k_proj, v_proj
