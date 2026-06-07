import os
import torch

# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32

class FusedQKVCuteSplitN:
    def __init__(self, num_warps: int = 12, BLOCK_M: int = 64, BLOCK_K: int = 512, NUM_HEADS: int = 16, HEAD_DIM: int = 64):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_K = BLOCK_K
        self.NUM_HEADS = NUM_HEADS
        self.HEAD_DIM = HEAD_DIM
        self.BLOCK_K_TILE = 32
        self.ab_stage = 2
        
    def _setup(self):
        # 1. Setup shared memory layouts
        self.smem_x_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_TILE, self.ab_stage),
            stride=(self.BLOCK_K_TILE, 1, self.BLOCK_M * self.BLOCK_K_TILE)
        )
        # Each sub-slice of sW has shape (HEAD_DIM, BLOCK_K_TILE, ab_stage)
        self.smem_w_layout_sub = cute.make_layout(
            (self.HEAD_DIM, self.BLOCK_K_TILE, self.ab_stage),
            stride=(self.BLOCK_K_TILE, 1, self.HEAD_DIM * self.BLOCK_K_TILE)
        )
        # Each sub-slice of sBuf has shape (BLOCK_M, HEAD_DIM)
        self.smem_accum_layout_sub = cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1))
        
        # 2. Shared storage definition (Fits in 81.5 KB)
        @cute.struct
        class SharedStorageSplitN:
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.BLOCK_M * self.BLOCK_K_TILE], 128]
            sW_q: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.HEAD_DIM * self.BLOCK_K_TILE], 128]
            sW_k: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.HEAD_DIM * self.BLOCK_K_TILE], 128]
            sW_v: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.HEAD_DIM * self.BLOCK_K_TILE], 128]
            sBuf_q: cute.struct.Align[cute.struct.MemRange[Float32, self.BLOCK_M * self.HEAD_DIM], 128]
            sBuf_k: cute.struct.Align[cute.struct.MemRange[Float32, self.BLOCK_M * self.HEAD_DIM], 128]
            sBuf_v: cute.struct.Align[cute.struct.MemRange[Float32, self.BLOCK_M * self.HEAD_DIM], 128]
        self.shared_storage = SharedStorageSplitN
        
        # 3. TiledCopy configurations using cp.async (uses 4 warps = 128 threads in parallel to match 8-element vector copy size)
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom = cute.make_copy_atom(copy_op, BFloat16, num_bits_per_copy=128)
        
        self.gmem_tiled_copy_x = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((32, 4), order=(1, 0)),
            cute.make_layout((1, 8)) # 8 elements of BFloat16 = 128 bits
        )
        self.gmem_tiled_copy_w = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((32, 4), order=(1, 0)),
            cute.make_layout((1, 8)) # 8 elements of BFloat16 = 128 bits
        )
        
        # 4. TiledMMA configuration (uses 128 threads / 4 warps)
        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=(2, 2, 1))
        
        # 5. TiledCopy configuration for cooperative stores in the epilogue
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
        num_threads = self.num_warps * cute.arch.WARP_SIZE
        
        self.kernel(
            x, weight, rmsnorm_weight, scales, sin, cos, q_proj, k_proj, v_proj,
            self.smem_x_layout, self.smem_w_layout_sub, self.smem_accum_layout_sub,
            self.tiled_mma, self.gmem_tiled_copy_x, self.gmem_tiled_copy_w, self.gmem_tiled_copy_store
        ).launch(
            grid=[cute.ceil_div(N, self.BLOCK_M), B, self.NUM_HEADS],
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
        smem_x_layout: cute.Layout,
        smem_w_layout_sub: cute.Layout,
        smem_accum_layout_sub: cute.Layout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_x: cute.TiledCopy,
        gmem_tiled_copy_w: cute.TiledCopy,
        gmem_tiled_copy_store: cute.TiledCopy
    ):
        cta_m, cta_b, cta_h = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        seq_start = cta_m * self.BLOCK_M
        N = x.shape[1]
        
        # Thread indexing
        warp_idx = tid // 32
        wg_id = warp_idx // 4
        
        # Select active buffers using explicit ternary conditions
        w_ptr = storage.sW_q.data_ptr() if wg_id == 0 else (storage.sW_k.data_ptr() if wg_id == 1 else storage.sW_v.data_ptr())
        accum_ptr = storage.sBuf_q.data_ptr() if wg_id == 0 else (storage.sBuf_k.data_ptr() if wg_id == 1 else storage.sBuf_v.data_ptr())
        
        smem_x = storage.sX.get_tensor(smem_x_layout)
        smem_w = cute.make_tensor(w_ptr.align(16), smem_w_layout_sub)
        smem_accum = cute.make_tensor(accum_ptr.align(16), smem_accum_layout_sub)
        
        # Slices for epilogue (used by the first 64 threads)
        smem_accum_q = storage.sBuf_q.get_tensor(smem_accum_layout_sub)
        smem_accum_k = storage.sBuf_k.get_tensor(smem_accum_layout_sub)
        smem_accum_v = storage.sBuf_v.get_tensor(smem_accum_layout_sub)
        
        # TiledCopy slicing for the 128 copy threads
        thr_copy_x = gmem_tiled_copy_x.get_slice(tid % 128)
        thr_copy_w = gmem_tiled_copy_w.get_slice(tid % 128)
        
        # Slices for weight loading (only first 128 threads copy)
        smem_w_q = storage.sW_q.get_tensor(smem_w_layout_sub)
        smem_w_k = storage.sW_k.get_tensor(smem_w_layout_sub)
        smem_w_v = storage.sW_v.get_tensor(smem_w_layout_sub)
        
        tSsX = thr_copy_x.partition_D(smem_x)
        tSsW_q = thr_copy_w.partition_D(smem_w_q)
        tSsW_k = thr_copy_w.partition_D(smem_w_k)
        tSsW_v = thr_copy_w.partition_D(smem_w_v)
        
        # Computational Partitioning for Warpgroups 0, 1, 2
        local_tid = tid % 128
        thr_mma = tiled_mma.get_slice(local_tid)
        
        # Partition Accumulators in register fragments
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        # Partition SMEM stages outside the loop
        tCsX = thr_mma.partition_A(smem_x)
        tCsW = thr_mma.partition_B(smem_w)
        
        # Pre-allocate double-buffered register fragments
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        k_tile_cnt = self.BLOCK_K // self.BLOCK_K_TILE
        
        # Prefetch stage 0
        if k_tile_cnt > 0:
            if tid < 128:
                # Load X
                x_ptr = (x[cta_b, seq_start, None].iterator + 0).align(16)
                gX = cute.make_tensor(x_ptr, cute.make_layout((self.BLOCK_M, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                tSgX = thr_copy_x.partition_S(gX)
                cute.copy(gmem_tiled_copy_x, tSgX, tSsX[None, None, None, 0])
                
                # Load Q Weight
                weight_ptr_q = (weight[cta_h * self.HEAD_DIM, None].iterator + 0).align(16)
                gW_q = cute.make_tensor(weight_ptr_q, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                tSgW_q = thr_copy_w.partition_S(gW_q)
                cute.copy(gmem_tiled_copy_w, tSgW_q, tSsW_q[None, None, None, 0])
                
                # Load K Weight
                weight_ptr_k = (weight[(self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + 0).align(16)
                gW_k = cute.make_tensor(weight_ptr_k, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                tSgW_k = thr_copy_w.partition_S(gW_k)
                cute.copy(gmem_tiled_copy_w, tSgW_k, tSsW_k[None, None, None, 0])
                
                # Load V Weight
                weight_ptr_v = (weight[(2 * self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + 0).align(16)
                gW_v = cute.make_tensor(weight_ptr_v, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                tSgW_v = thr_copy_w.partition_S(gW_v)
                cute.copy(gmem_tiled_copy_w, tSgW_v, tSsW_v[None, None, None, 0])
                
                cute.arch.cp_async_commit_group()
        
        # Wait for stage 0 and prefetch to registers
        if k_tile_cnt > 0:
            if tid < 128:
                cute.arch.cp_async_wait_group(self.ab_stage - 2)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
                
        # Main execution loop
        for k_tile_idx in cutlass.range_constexpr(k_tile_cnt):
            pipe_read = k_tile_idx % self.ab_stage
            pipe_write = (k_tile_idx + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (k_tile_idx + 1) % self.ab_stage
            
            next_gmem_read_idx = k_tile_idx + (self.ab_stage - 1)
            if next_gmem_read_idx < k_tile_cnt:
                k_tile_offset = next_gmem_read_idx * self.BLOCK_K_TILE
                
                if tid < 128:
                    # Load X
                    x_ptr = (x[cta_b, seq_start, None].iterator + k_tile_offset).align(16)
                    gX = cute.make_tensor(x_ptr, cute.make_layout((self.BLOCK_M, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                    tSgX = thr_copy_x.partition_S(gX)
                    cute.copy(gmem_tiled_copy_x, tSgX, tSsX[None, None, None, pipe_write])
                    
                    # Load Q Weight
                    weight_ptr_q = (weight[cta_h * self.HEAD_DIM, None].iterator + k_tile_offset).align(16)
                    gW_q = cute.make_tensor(weight_ptr_q, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                    tSgW_q = thr_copy_w.partition_S(gW_q)
                    cute.copy(gmem_tiled_copy_w, tSgW_q, tSsW_q[None, None, None, pipe_write])
                    
                    # Load K Weight
                    weight_ptr_k = (weight[(self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + k_tile_offset).align(16)
                    gW_k = cute.make_tensor(weight_ptr_k, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                    tSgW_k = thr_copy_w.partition_S(gW_k)
                    cute.copy(gmem_tiled_copy_w, tSgW_k, tSsW_k[None, None, None, pipe_write])
                    
                    # Load V Weight
                    weight_ptr_v = (weight[(2 * self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + k_tile_offset).align(16)
                    gW_v = cute.make_tensor(weight_ptr_v, cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1)))
                    tSgW_v = thr_copy_w.partition_S(gW_v)
                    cute.copy(gmem_tiled_copy_w, tSgW_v, tSsW_v[None, None, None, pipe_write])
                    
                    cute.arch.cp_async_commit_group()
                    
            # Prefetch next stage's registers
            if k_tile_idx + 1 < k_tile_cnt:
                if tid < 128:
                    cute.arch.cp_async_wait_group(self.ab_stage - 2)
                cute.arch.sync_threads()
                
                tCsA_next = tCsX[None, None, None, next_pipe_read]
                tCsB_next = tCsW[None, None, None, next_pipe_read]
                
                if k_tile_idx % 2 == 0:
                    cute.autovec_copy(tCsA_next, rA_1)
                    cute.autovec_copy(tCsB_next, rB_1)
                else:
                    cute.autovec_copy(tCsA_next, rA_0)
                    cute.autovec_copy(tCsB_next, rB_0)
                        
            # Perform MMA using current stage registers
            if k_tile_idx % 2 == 0:
                cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
            else:
                cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                    
        # Write back accumulators to SMEM at the end of the GEMM loop
        for i in range(cute.size(accum)):
            tCsC[i] = accum[i]
                
        cute.arch.sync_threads()
        
        # Recast buffers to BFloat16 views for writeback
        sOut_bf16_q = cute.make_tensor(cute.recast_ptr(storage.sBuf_q.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
        sOut_bf16_k = cute.make_tensor(cute.recast_ptr(storage.sBuf_k.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
        sOut_bf16_v = cute.make_tensor(cute.recast_ptr(storage.sBuf_v.data_ptr().align(16), dtype=BFloat16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
        
        # Register fragments
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        v_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        if tid < self.BLOCK_M and seq_start + tid < N:
            # Slices of shared memory
            q_ptr = smem_accum_q.iterator + tid * self.HEAD_DIM
            k_ptr = smem_accum_k.iterator + tid * self.HEAD_DIM
            v_ptr = smem_accum_v.iterator + tid * self.HEAD_DIM
            
            q_smem = cute.make_tensor(q_ptr, cute.make_layout((self.HEAD_DIM,)))
            k_smem = cute.make_tensor(k_ptr, cute.make_layout((self.HEAD_DIM,)))
            v_smem = cute.make_tensor(v_ptr, cute.make_layout((self.HEAD_DIM,)))
            
            # Vectorized Shared-to-Register bulk load
            cute.autovec_copy(q_smem, q_reg)
            cute.autovec_copy(k_smem, k_reg)
            cute.autovec_copy(v_smem, v_reg)
        else:
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                q_reg[j] = Float32(0.0)
                k_reg[j] = Float32(0.0)
                v_reg[j] = Float32(0.0)
                
        # Synchronize threads block-wide to prevent overwriting smem_accum before all threads have loaded their inputs
        cute.arch.sync_threads()
        
        # Only the first 64 threads execute the epilogue computation (1 thread per row)
        if tid < self.BLOCK_M and seq_start + tid < N:
            half_dim = self.HEAD_DIM // 2
            
            # Vectorized RMSNorm Reductions
            q_sum_sq = Float32(0.0)
            k_sum_sq = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                val_q = q_reg[j]
                val_k = k_reg[j]
                q_sum_sq += val_q * val_q
                k_sum_sq += val_k * val_k
            
            q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=True)
            k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=True)
            
            # Scale and Apply RoPE
            scale_val = Float32(scales[cta_h, 0, 0])
            q_scale_factor = cute.math.exp(scale_val)
            
            q_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            k_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            v_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
            
            for j in cutlass.range(half_dim, unroll_full=True):
                # Load cos and sin on-the-fly
                c0 = Float32(cos[cta_b, 0, seq_start + tid, j])
                c1 = Float32(cos[cta_b, 0, seq_start + tid, j + half_dim])
                s0 = Float32(sin[cta_b, 0, seq_start + tid, j])
                s1 = Float32(sin[cta_b, 0, seq_start + tid, j + half_dim])
                
                # Load RMSNorm weights on-the-fly
                w_q_0 = Float32(rmsnorm_weight[0, 0, j])
                w_q_1 = Float32(rmsnorm_weight[0, 0, j + half_dim])
                w_k_0 = Float32(rmsnorm_weight[1, 0, j])
                w_k_1 = Float32(rmsnorm_weight[1, 0, j + half_dim])
                
                # Apply RMSNorm
                qn0 = q_reg[j] * q_rsqrt * w_q_0
                qn1 = q_reg[j + half_dim] * q_rsqrt * w_q_1
                kn0 = k_reg[j] * k_rsqrt * w_k_0
                kn1 = k_reg[j + half_dim] * k_rsqrt * w_k_1
                
                # Apply RoPE
                qr0 = qn0 * c0 - qn1 * s0
                qr1 = qn1 * c1 + qn0 * s1
                kr0 = kn0 * c0 - kn1 * s0
                kr1 = kn1 * c1 + kn0 * s1
                
                # Store back to registers
                q_out_reg[j] = BFloat16(qr0 * q_scale_factor)
                q_out_reg[j + half_dim] = BFloat16(qr1 * q_scale_factor)
                k_out_reg[j] = BFloat16(kr0)
                k_out_reg[j + half_dim] = BFloat16(kr1)
                
                v_out_reg[j] = BFloat16(v_reg[j])
                v_out_reg[j + half_dim] = BFloat16(v_reg[j + half_dim])
                
            # Write out to sOut_bf16 (vectorized)
            row_offset = tid * self.HEAD_DIM
            out_q_smem = cute.make_tensor(sOut_bf16_q.iterator + row_offset, cute.make_layout((self.HEAD_DIM,)))
            out_k_smem = cute.make_tensor(sOut_bf16_k.iterator + row_offset, cute.make_layout((self.HEAD_DIM,)))
            out_v_smem = cute.make_tensor(sOut_bf16_v.iterator + row_offset, cute.make_layout((self.HEAD_DIM,)))
            
            cute.autovec_copy(q_out_reg, out_q_smem)
            cute.autovec_copy(k_out_reg, out_k_smem)
            cute.autovec_copy(v_out_reg, out_v_smem)
            
        # Synchronize threads block-wide
        cute.arch.sync_threads()
        
        # Cooperatively store results to Global Memory
        if seq_start + self.BLOCK_M <= N:
            if tid < 128:
                thr_copy_store = gmem_tiled_copy_store.get_slice(tid)
                
                g_tile_q = cute.make_tensor(q_proj[cta_b, cta_h, seq_start, None].iterator.align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                g_tile_k = cute.make_tensor(k_proj[cta_b, cta_h, seq_start, None].iterator.align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                g_tile_v = cute.make_tensor(v_proj[cta_b, cta_h, seq_start, None].iterator.align(16), cute.make_layout((self.BLOCK_M, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)))
                
                tS_q = thr_copy_store.partition_S(sOut_bf16_q)
                tD_q = thr_copy_store.partition_D(g_tile_q)
                cute.copy(gmem_tiled_copy_store, tS_q, tD_q)
                
                tS_k = thr_copy_store.partition_S(sOut_bf16_k)
                tD_k = thr_copy_store.partition_D(g_tile_k)
                cute.copy(gmem_tiled_copy_store, tS_k, tD_k)
                
                tS_v = thr_copy_store.partition_S(sOut_bf16_v)
                tD_v = thr_copy_store.partition_D(g_tile_v)
                cute.copy(gmem_tiled_copy_store, tS_v, tD_v)
        else:
            if tid < self.BLOCK_M:
                row = tid
                seq = seq_start + row
                if seq < N:
                    q_g = q_proj.iterator + cta_b * (self.NUM_HEADS * N * self.HEAD_DIM) + cta_h * (N * self.HEAD_DIM) + seq * self.HEAD_DIM
                    k_g = k_proj.iterator + cta_b * (self.NUM_HEADS * N * self.HEAD_DIM) + cta_h * (N * self.HEAD_DIM) + seq * self.HEAD_DIM
                    v_g = v_proj.iterator + cta_b * (self.NUM_HEADS * N * self.HEAD_DIM) + cta_h * (N * self.HEAD_DIM) + seq * self.HEAD_DIM
                    
                    row_offset_thread = row * self.HEAD_DIM
                    q_s = sOut_bf16_q.iterator + row_offset_thread
                    k_s = sOut_bf16_k.iterator + row_offset_thread
                    v_s = sOut_bf16_v.iterator + row_offset_thread
                    
                    for j in cutlass.range(self.HEAD_DIM // 8, unroll_full=True):
                        reg_q = cute.make_rmem_tensor((8,), BFloat16)
                        reg_k = cute.make_rmem_tensor((8,), BFloat16)
                        reg_v = cute.make_rmem_tensor((8,), BFloat16)
                        
                        cute.autovec_copy(cute.make_tensor(q_s + j * 8, cute.make_layout((8,))), reg_q)
                        cute.autovec_copy(cute.make_tensor(k_s + j * 8, cute.make_layout((8,))), reg_k)
                        cute.autovec_copy(cute.make_tensor(v_s + j * 8, cute.make_layout((8,))), reg_v)
                        
                        cute.autovec_copy(reg_q, cute.make_tensor(q_g + j * 8, cute.make_layout((8,))))
                        cute.autovec_copy(reg_k, cute.make_tensor(k_g + j * 8, cute.make_layout((8,))))
                        cute.autovec_copy(reg_v, cute.make_tensor(v_g + j * 8, cute.make_layout((8,))))

_fwd_splitn_compile_cache = {}

def get_fused_qkv_splitn_fwd_kernel(num_warps=12, BLOCK_M=64, BLOCK_K=512, NUM_HEADS=16, HEAD_DIM=64):
    key = (num_warps, BLOCK_M, BLOCK_K, NUM_HEADS, HEAD_DIM)
    if key not in _fwd_splitn_compile_cache:
        from quack.compile_utils import make_fake_tensor
        
        B_sym = cute.sym_int()
        N_sym = cute.sym_int()
        
        fwd_op = FusedQKVCuteSplitN(num_warps=num_warps, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM)
        
        x_fake = make_fake_tensor(BFloat16, (B_sym, N_sym, BLOCK_K), 16)
        weight_fake = make_fake_tensor(BFloat16, (3 * NUM_HEADS * HEAD_DIM, BLOCK_K), 16)
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
        _fwd_splitn_compile_cache[key] = compiled_fwd
    return _fwd_splitn_compile_cache[key]

@torch.library.custom_op("flash_prologue::fused_qkv_cute_splitn", mutates_args=())
def fused_qkv_cute_splitn_op(
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
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    
    rmsnorm_weight_32 = rmsnorm_weight.to(torch.float32).contiguous()
    scales_32 = scales.to(torch.float32).contiguous()
    sin_32 = sin.to(torch.float32).contiguous()
    cos_32 = cos.to(torch.float32).contiguous()
    
    q_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    k_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    v_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    
    fwd_kernel = get_fused_qkv_splitn_fwd_kernel(
        num_warps=12,
        BLOCK_M=64,
        BLOCK_K=hidden,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim
    )
    
    fwd_kernel(
        x_bf16, weight_bf16, rmsnorm_weight_32, scales_32, sin_32, cos_32,
        q_proj_bf16, k_proj_bf16, v_proj_bf16
    )
    
    return q_proj_bf16, k_proj_bf16, v_proj_bf16

@fused_qkv_cute_splitn_op.register_fake
def _fused_qkv_cute_splitn_fake(x, weight, rmsnorm_weight, scales, sin, cos):
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    v_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    return q_proj, k_proj, v_proj

def setup_fused_qkv_cute_splitn_context(ctx, inputs, output):
    x, weight, rmsnorm_weight, scales, sin, cos = inputs
    q, k, v = output
    ctx.save_for_backward(x, weight, rmsnorm_weight, scales, sin, cos)

def fused_qkv_cute_splitn_backward_hook(ctx, grad_q, grad_k, grad_v):
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
    "flash_prologue::fused_qkv_cute_splitn",
    fused_qkv_cute_splitn_backward_hook,
    setup_context=setup_fused_qkv_cute_splitn_context
)
