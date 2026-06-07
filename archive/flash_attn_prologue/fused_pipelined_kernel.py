import os
import torch
import math

# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Boolean, const_expr
from quack.compile_utils import make_fake_tensor

class FusedQKVPipelinedCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_K_STAGE: int = 32, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 2, hidden: int = 512, use_sync_in_loop: bool = True):
        self.num_warps = num_warps  # 4 warps (128 threads)
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.BLOCK_N = HEAD_GROUP * 3 * HEAD_DIM  # e.g., 384 features for a pair of heads' Q, K, V
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
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(self.BLOCK_K_STAGE, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
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
        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        if self.BLOCK_M == 16:
            atom_layout_mnk = (1, 4, 1)
        else:
            atom_layout_mnk = (2, 2, 1)
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=atom_layout_mnk)
        
        # 4. Copy atom for Float32 accumulator writeback
        copy_op_r2s = cute.nvgpu.CopyUniversalOp()
        self.copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, Float32)
 
        # 5. TiledCopy configurations using cp.async (cooperative, tiled for all 128 threads)
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom = cute.make_copy_atom(copy_op, BFloat16, num_bits_per_copy=128)
        
        # For X (shape: BLOCK_M x BLOCK_K_STAGE)
        copy_x_val_k = min(8, self.BLOCK_K_STAGE)
        copy_x_thr_k = self.BLOCK_K_STAGE // copy_x_val_k
        copy_x_thr_m = min(128 // copy_x_thr_k, self.BLOCK_M)
        self.copy_x_threads = copy_x_thr_m * copy_x_thr_k
        self.gmem_tiled_copy_x = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((copy_x_thr_m, copy_x_thr_k), order=(1, 0)),
            cute.make_layout((1, copy_x_val_k))
        )
        
        # For W (shape: BLOCK_N x BLOCK_K_STAGE)
        copy_w_val_k = min(8, self.BLOCK_K_STAGE)
        copy_w_thr_k = self.BLOCK_K_STAGE // copy_w_val_k
        copy_w_thr_n = min(128 // copy_w_thr_k, self.BLOCK_N)
        self.copy_w_threads = copy_w_thr_n * copy_w_thr_k
        self.gmem_tiled_copy_w = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((copy_w_thr_n, copy_w_thr_k), order=(1, 0)),
            cute.make_layout((1, copy_w_val_k))
        )
        
        store_op = cute.nvgpu.CopyUniversalOp()
        store_atom = cute.make_copy_atom(store_op, BFloat16, num_bits_per_copy=128)
        store_thr_m = min(16, self.BLOCK_M)
        store_thr_n = 128 // store_thr_m
        self.gmem_tiled_copy_store = cute.make_tiled_copy_tv(
            store_atom,
            cute.make_ordered_layout((store_thr_m, store_thr_n), order=(1, 0)),
            cute.make_layout((1, 8))
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        q_proj: cute.Tensor,
        k_proj: cute.Tensor,
        v_proj: cute.Tensor,
        stream
    ):
        self._setup()
        
        B = x.shape[0]
        N = x.shape[1]
        
        num_threads = self.num_warps * 32
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            x, weight,
            rmsnorm_weight, scales, sin, cos,
            q_proj, k_proj, v_proj,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout,
            self.copy_atom_r2s,
            self.gmem_tiled_copy_x,
            self.gmem_tiled_copy_w,
            self.gmem_tiled_copy_store,
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
        weight: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        q_proj: cute.Tensor,
        k_proj: cute.Tensor,
        v_proj: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        copy_atom_r2s: cute.CopyAtom,
        gmem_tiled_copy_x: cute.TiledCopy,
        gmem_tiled_copy_w: cute.TiledCopy,
        gmem_tiled_copy_store: cute.TiledCopy,
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
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        
        # Local_tile partition global tensors
        N = x.shape[1]
        hidden = x.shape[2]
        k_tile_cnt = hidden // self.BLOCK_K_STAGE
        
        # Thread slice of sW layout
        sW_sub_layout = cute.make_layout((self.BLOCK_N, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1))
        
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
        seq_clamped = cutlass.min(seq, N - 1)
        
        # Preload scales and RMSNorm weights AOT (Ahead-of-Time)
        scale_val = Float32(0.0)
        q_scale_factor = Float32(1.0)
        w_q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        w_k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        # Load scales and weights if they are valid
        if global_h < self.NUM_HEADS:
            scale_val = Float32(scales[global_h, 0, 0])
            q_scale_factor = cute.math.exp(scale_val)
            
            w_q_ptr = rmsnorm_weight[0, 0, None].iterator.align(16)
            w_k_ptr = rmsnorm_weight[1, 0, None].iterator.align(16)
            gW_q = cute.make_tensor(w_q_ptr, cute.make_layout((self.HEAD_DIM,)))
            gW_k = cute.make_tensor(w_k_ptr, cute.make_layout((self.HEAD_DIM,)))
            cute.autovec_copy(gW_q, w_q_reg)
            cute.autovec_copy(gW_k, w_k_reg)
        else:
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                w_q_reg[j] = Float32(1.0)
                w_k_reg[j] = Float32(1.0)
                
        # Layouts for global reads
        gX_layout = cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(hidden, 1))
        gW_layout = cute.make_layout((self.BLOCK_N, self.BLOCK_K_STAGE), stride=(hidden, 1))
        
        # Prefetch initial stages (up to self.ab_stage - 1)
        for stage in range(self.ab_stage - 1):
            if stage < k_tile_cnt:
                k_offset_init = stage * self.BLOCK_K_STAGE
                
                # Load X
                if tidx < self.copy_x_threads:
                    thr_copy_x = gmem_tiled_copy_x.get_slice(tidx)
                    tSsX_stage = thr_copy_x.partition_D(sX_aligned)
                    x_ptr_init = (x[cta_b, cta_m * self.BLOCK_M, None].iterator + k_offset_init).align(16)
                    gX_init = cute.make_tensor(x_ptr_init, gX_layout)
                    tSgX_init = thr_copy_x.partition_S(gX_init)
                    cute.copy(gmem_tiled_copy_x, tSgX_init, tSsX_stage[None, None, None, stage])
                
                # Load W: Q, K, V weights for current head group contiguously
                if tidx < self.copy_w_threads:
                    thr_copy_w = gmem_tiled_copy_w.get_slice(tidx)
                    sW_ptr_init = sW_aligned.iterator + stage * (self.BLOCK_N * self.BLOCK_K_STAGE)
                    sW_write_init = cute.make_tensor(sW_ptr_init.align(16), sW_sub_layout)
                    tSsW_init = thr_copy_w.partition_D(sW_write_init)
                    
                    weight_ptr_init = (weight[cta_h_group * self.BLOCK_N, None].iterator + k_offset_init).align(16)
                    gW_init = cute.make_tensor(weight_ptr_init, gW_layout)
                    tSgW_init = thr_copy_w.partition_S(gW_init)
                    cute.copy(gmem_tiled_copy_w, tSgW_init, tSsW_init)
                
                cute.arch.cp_async_commit_group()
                
        # Wait for the first stage (stage 0) to be ready in shared memory and prefetch it to rA_0/rB_0 (vectorized)
        if k_tile_cnt > 0:
            if self.ab_stage - 1 < k_tile_cnt:
                cute.arch.cp_async_wait_group(self.ab_stage - 2)
            else:
                cute.arch.cp_async_wait_group(0)
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
                k_offset_loop = next_gmem_read * self.BLOCK_K_STAGE
                
                # Load X
                if tidx < self.copy_x_threads:
                    thr_copy_x = gmem_tiled_copy_x.get_slice(tidx)
                    tSsX_stage = thr_copy_x.partition_D(sX_aligned)
                    x_ptr_loop = (x[cta_b, cta_m * self.BLOCK_M, None].iterator + k_offset_loop).align(16)
                    gX_loop = cute.make_tensor(x_ptr_loop, gX_layout)
                    tSgX_loop = thr_copy_x.partition_S(gX_loop)
                    cute.copy(gmem_tiled_copy_x, tSgX_loop, tSsX_stage[None, None, None, pipe_write])
                
                # Load W
                if tidx < self.copy_w_threads:
                    thr_copy_w = gmem_tiled_copy_w.get_slice(tidx)
                    sW_ptr_loop = sW_aligned.iterator + pipe_write * (self.BLOCK_N * self.BLOCK_K_STAGE)
                    sW_write_loop = cute.make_tensor(sW_ptr_loop.align(16), sW_sub_layout)
                    tSsW_loop = thr_copy_w.partition_D(sW_write_loop)
                    
                    weight_ptr_loop = (weight[cta_h_group * self.BLOCK_N, None].iterator + k_offset_loop).align(16)
                    gW_loop = cute.make_tensor(weight_ptr_loop, gW_layout)
                    tSgW_loop = thr_copy_w.partition_S(gW_loop)
                    cute.copy(gmem_tiled_copy_w, tSgW_loop, tSsW_loop)
                
                cute.arch.cp_async_commit_group()
                
            # Prefetch next stage's registers
            if k_tile + 1 < k_tile_cnt:
                if next_gmem_read < k_tile_cnt:
                    cute.arch.cp_async_wait_group(self.ab_stage - 2)
                else:
                    cute.arch.cp_async_wait_group(0)
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

        r = tidx % self.BLOCK_M
        h_local = tidx // self.BLOCK_M
        
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        seq = cta_m * self.BLOCK_M + r
        
        row_offset = r * self.BLOCK_N_padded + h_local * (3 * self.HEAD_DIM)
        
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        v_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        if seq < N:
            q_ptr = smem_accum.iterator + row_offset
            k_ptr = smem_accum.iterator + row_offset + self.HEAD_DIM
            v_ptr = smem_accum.iterator + row_offset + 2 * self.HEAD_DIM
            
            q_smem = cute.make_tensor(q_ptr, cute.make_layout((self.HEAD_DIM,)))
            k_smem = cute.make_tensor(k_ptr, cute.make_layout((self.HEAD_DIM,)))
            v_smem = cute.make_tensor(v_ptr, cute.make_layout((self.HEAD_DIM,)))
            
            cute.autovec_copy(q_smem, q_reg)
            cute.autovec_copy(k_smem, k_reg)
            cute.autovec_copy(v_smem, v_reg)
        else:
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                q_reg[j] = Float32(0.0)
                k_reg[j] = Float32(0.0)
                v_reg[j] = Float32(0.0)
                
        cute.arch.sync_threads()
        
        q_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        v_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        
        if seq < N:
            cos_ptr = cos[cta_b, 0, seq, None].iterator.align(16)
            sin_ptr = sin[cta_b, 0, seq, None].iterator.align(16)
            gCos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM,)))
            gSin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM,)))
            cos_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
            sin_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
            cute.autovec_copy(gCos, cos_reg)
            cute.autovec_copy(gSin, sin_reg)

            q_sum_sq = Float32(0.0)
            k_sum_sq = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                val_q = q_reg[j]
                val_k = k_reg[j]
                q_sum_sq += val_q * val_q
                k_sum_sq += val_k * val_k
                
            q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=True)
            k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=True)
            
            half_dim = self.HEAD_DIM // 2
            for j in cutlass.range(half_dim, unroll_full=True):
                c0 = cos_reg[j]
                c1 = cos_reg[j + half_dim]
                s0 = sin_reg[j]
                s1 = sin_reg[j + half_dim]
                
                w_q_0 = w_q_reg[j]
                w_q_1 = w_q_reg[j + half_dim]
                w_k_0 = w_k_reg[j]
                w_k_1 = w_k_reg[j + half_dim]
                
                qn0 = q_reg[j] * q_rsqrt * w_q_0
                qn1 = q_reg[j + half_dim] * q_rsqrt * w_q_1
                kn0 = k_reg[j] * k_rsqrt * w_k_0
                kn1 = k_reg[j + half_dim] * k_rsqrt * w_k_1
                
                qr0 = qn0 * c0 - qn1 * s0
                qr1 = qn1 * c1 + qn0 * s1
                kr0 = kn0 * c0 - kn1 * s0
                kr1 = kn1 * c1 + kn0 * s1
                
                q_out_reg[j] = BFloat16(qr0 * q_scale_factor)
                q_out_reg[j + half_dim] = BFloat16(qr1 * q_scale_factor)
                k_out_reg[j] = BFloat16(kr0)
                k_out_reg[j + half_dim] = BFloat16(kr1)
                
                v_out_reg[j] = BFloat16(v_reg[j])
                v_out_reg[j + half_dim] = BFloat16(v_reg[j + half_dim])
        else:
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                q_out_reg[j] = BFloat16(0.0)
                k_out_reg[j] = BFloat16(0.0)
                v_out_reg[j] = BFloat16(0.0)
                
        sOut_bf16 = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.BLOCK_N), stride=(self.BLOCK_N_padded, 1)))
        
        out_q_smem = cute.make_tensor(sOut_bf16.iterator + row_offset, cute.make_layout((self.HEAD_DIM,)))
        out_k_smem = cute.make_tensor(sOut_bf16.iterator + row_offset + self.HEAD_DIM, cute.make_layout((self.HEAD_DIM,)))
        out_v_smem = cute.make_tensor(sOut_bf16.iterator + row_offset + 2 * self.HEAD_DIM, cute.make_layout((self.HEAD_DIM,)))
        
        cute.autovec_copy(q_out_reg, out_q_smem)
        cute.autovec_copy(k_out_reg, out_k_smem)
        cute.autovec_copy(v_out_reg, out_v_smem)
        
        cute.arch.sync_threads()
        
        thr_copy_store = gmem_tiled_copy_store.get_slice(tidx)
        
        if is_valid:
            if (cta_m + 1) * self.BLOCK_M <= N:
                for h_idx in cutlass.range(self.HEAD_GROUP, unroll_full=True):
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


_fused_pipelined_compile_cache = {}

def get_fused_pipelined_fwd_kernel(num_warps=4, BLOCK_M=128, BLOCK_K_STAGE=64, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=2, hidden=512, use_sync_in_loop=True):
    key = (num_warps, BLOCK_M, BLOCK_K_STAGE, NUM_HEADS, HEAD_DIM, HEAD_GROUP, hidden, use_sync_in_loop)
    if key not in _fused_pipelined_compile_cache:
        B_sym = cute.sym_int()
        N_sym = cute.sym_int()
        
        fwd_op = FusedQKVPipelinedCute(num_warps=num_warps, BLOCK_M=BLOCK_M, BLOCK_K_STAGE=BLOCK_K_STAGE, NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM, HEAD_GROUP=HEAD_GROUP, hidden=hidden, use_sync_in_loop=use_sync_in_loop)
        
        x_fake = make_fake_tensor(BFloat16, (B_sym, N_sym, hidden), 16)
        weight_fake = make_fake_tensor(BFloat16, (3 * NUM_HEADS * HEAD_DIM, hidden), 16)
        rmsnorm_weight_fake = make_fake_tensor(Float32, (2, 1, HEAD_DIM), 16)
        scales_fake = make_fake_tensor(Float32, (NUM_HEADS, 1, 1), 1)
        sin_fake = make_fake_tensor(Float32, (B_sym, 1, N_sym, HEAD_DIM), 16)
        cos_fake = make_fake_tensor(Float32, (B_sym, 1, N_sym, HEAD_DIM), 16)
        
        q_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        k_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        v_proj_fake = make_fake_tensor(BFloat16, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            x_fake, weight_fake, rmsnorm_weight_fake, scales_fake, sin_fake, cos_fake,
            q_proj_fake, k_proj_fake, v_proj_fake,
            fake_stream,
            options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount=255'",
        )
        _fused_pipelined_compile_cache[key] = compiled_fwd
    return _fused_pipelined_compile_cache[key]
 
@torch.library.custom_op("flash_prologue::fused_qkv_pipelined_cute", mutates_args=())
def fused_qkv_pipelined_cute_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    
    x_bf16 = x.to(torch.bfloat16).contiguous()
    
    # Query Autotuner for optimal shape configs
    from flash_attn_prologue.autotune import get_autotuned_config
    config = get_autotuned_config(B * N, 3 * num_heads * head_dim, hidden, is_fused=True, device=str(x.device))
    
    block_m = config["BLOCK_M"]
    block_k = config["BLOCK_K"]
    head_group = config.get("HEAD_GROUP", 2)
    
    # Pre-permute weight to make Q, K, V weights for head groups contiguous in memory.
    hg = head_group
    w_reshaped = weight.to(torch.bfloat16).view(3, num_heads, head_dim, hidden)
    w_split = w_reshaped.view(3, num_heads // hg, hg, head_dim, hidden)
    w_permuted = w_split.permute(1, 2, 0, 3, 4)
    weight_bf16 = w_permuted.reshape(num_heads * 3 * head_dim, hidden).contiguous()
    
    rmsnorm_weight_32 = rmsnorm_weight.to(torch.float32).contiguous()
    scales_32 = scales.to(torch.float32).contiguous()
    sin_32 = sin.to(torch.float32).contiguous()
    cos_32 = cos.to(torch.float32).contiguous()
    
    q_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    k_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    v_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    
    fwd_kernel = get_fused_pipelined_fwd_kernel(
        num_warps=4,
        BLOCK_M=block_m,
        BLOCK_K_STAGE=block_k,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        HEAD_GROUP=head_group,
        hidden=hidden,
        use_sync_in_loop=True
    )
    
    fwd_kernel(
        x_bf16, weight_bf16, rmsnorm_weight_32, scales_32, sin_32, cos_32,
        q_proj_bf16, k_proj_bf16, v_proj_bf16
    )
    
    return q_proj_bf16, k_proj_bf16, v_proj_bf16

@fused_qkv_pipelined_cute_op.register_fake
def _fused_qkv_pipelined_cute_fake(x, weight, rmsnorm_weight, scales, sin, cos):
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    v_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    return q_proj, k_proj, v_proj

def setup_fused_pipelined_context(ctx, inputs, output):
    x, weight, rmsnorm_weight, scales, sin, cos = inputs
    q, k, v = output
    ctx.save_for_backward(x, weight, rmsnorm_weight, scales, sin, cos)

def fused_pipelined_backward_hook(ctx, grad_q, grad_k, grad_v):
    x, weight, rmsnorm_weight, scales, sin, cos = ctx.saved_tensors
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    eps = 1e-5
    
    # Recompute forward intermediates using eager custom op
    _, _, _, q_rsqrt, k_rsqrt = torch.ops.flash_prologue.fused_qkv(
        x, weight, rmsnorm_weight, scales, sin, cos, num_heads, head_dim, eps
    )
    
    # Execute eager backward custom op
    grad_x, grad_weight, grad_rmsnorm_weight, grad_scales = torch.ops.flash_prologue.fused_qkv_backward(
        grad_q, grad_k, grad_v, x, weight, rmsnorm_weight, scales, sin, cos,
        q_rsqrt, k_rsqrt, num_heads, head_dim, eps
    )
    
    return grad_x, grad_weight, grad_rmsnorm_weight, grad_scales, None, None

torch.library.register_autograd(
    "flash_prologue::fused_qkv_pipelined_cute",
    fused_pipelined_backward_hook,
    setup_context=setup_fused_pipelined_context
)
