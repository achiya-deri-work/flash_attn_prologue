import os
import torch
import math
from typing import Tuple


# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor
from flash_attn_prologue.autotune import get_autotuned_config

def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 5:
        if t.shape[2] == 1:
            return t.squeeze(2)
        elif t.shape[1] == 1:
            return t.squeeze(1)
    elif t.ndim == 3:
        return t.unsqueeze(1)
    elif t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(1)
    return t

class FusedEpilogueCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 1, ab_stage: int = 1, q_in_strides=None, k_in_strides=None, q_out_strides=None, k_out_strides=None, sin_strides=None, cos_strides=None):
        self.num_warps = num_warps  # 4 warps (128 threads)
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.BLOCK_N = HEAD_GROUP * HEAD_DIM
        self.BLOCK_N_padded = self.BLOCK_N + 8
        self.NUM_HEADS = NUM_HEADS
        self.ab_stage = ab_stage
        self.SUB_M = BLOCK_M // ab_stage
        self.q_in_strides = q_in_strides
        self.k_in_strides = k_in_strides
        self.q_out_strides = q_out_strides
        self.k_out_strides = k_out_strides
        self.sin_strides = sin_strides
        self.cos_strides = cos_strides
        
        # Shared memory footprint calculation with padding to avoid LDS bank conflicts
        sQ_size = self.SUB_M * self.BLOCK_N_padded * self.ab_stage * 2
        sK_size = self.SUB_M * self.BLOCK_N_padded * self.ab_stage * 2
        
        self.max_smem_bytes = sQ_size + sK_size
        self.flat_smem_elements = (self.max_smem_bytes + 3) // 4
        
        # Element offsets in Float32 for flat layouts
        self.sK_offset_f32 = sQ_size // 4

    def _setup(self):
        self.val_smem_layout = cute.make_layout(
            (self.SUB_M, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N_padded, 1, self.SUB_M * self.BLOCK_N_padded)
        )
        
        # 3. TiledCopy configurations using cp.async
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom_bf16 = cute.make_copy_atom(copy_op, BFloat16, num_bits_per_copy=128)
        
        # Copy layout for Q, K (a single head of shape (SUB_M, HEAD_DIM))
        num_threads = self.num_warps * 32
        copy_val_val_k = min(8, self.HEAD_DIM)
        copy_val_thr_k = self.HEAD_DIM // copy_val_val_k
        max_thr_m = num_threads // copy_val_thr_k
        copy_val_thr_m = 1
        for d in range(1, self.SUB_M + 1):
            if self.SUB_M % d == 0 and d <= max_thr_m:
                copy_val_thr_m = d
        self.copy_val_threads = copy_val_thr_m * copy_val_thr_k
        self.gmem_tiled_copy_val = cute.make_tiled_copy_tv(
            copy_atom_bf16,
            cute.make_ordered_layout((copy_val_thr_m, copy_val_thr_k), order=(1, 0)),
            cute.make_layout((1, copy_val_val_k))
        )
        
        # Global layouts for inputs Q and K using their loading strides
        self.g_q_in_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.q_in_strides[2], self.q_in_strides[3])
        )
        self.g_k_in_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.k_in_strides[2], self.k_in_strides[3])
        )
        
        # Storage definition
        @cute.struct
        class SharedStorageEpilogue:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageEpilogue

    @cute.jit
    def load_stage(
        self,
        stage_idx,
        seq_start,
        N,
        q_in,
        k_in,
        sQ,
        sK,
        tidx,
        is_valid,
        cta_b,
        cta_h_group,
        gmem_tiled_copy_val,
        g_q_in_layout,
        g_k_in_layout
    ):
        stage_seq_start = seq_start + stage_idx * self.SUB_M
        clamped_seq = cutlass.min(stage_seq_start, N - 1)
        stage_offset = stage_idx * self.SUB_M * self.BLOCK_N_padded
        
        if is_valid:
            if tidx < self.copy_val_threads:
                if self.HEAD_GROUP == 1:
                    curr_h0 = cta_h_group
                    if curr_h0 < self.NUM_HEADS:
                        g_offset0_q = cta_b * self.q_in_strides[0] + curr_h0 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        # Q0
                        q_g_ptr0 = (q_in.iterator + g_offset0_q).align(16)
                        gQ0 = cute.make_tensor(q_g_ptr0, g_q_in_layout)
                        tSgQ0 = thr_copy_val.partition_S(gQ0)
                        
                        sQ_slice0 = cute.make_tensor((sQ.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ0 = thr_copy_val.partition_D(sQ_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgQ0, tSsQ0)
                        
                        # K0
                        g_offset0_k = cta_b * self.k_in_strides[0] + curr_h0 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr0 = (k_in.iterator + g_offset0_k).align(16)
                        gK0 = cute.make_tensor(k_g_ptr0, g_k_in_layout)
                        tSgK0 = thr_copy_val.partition_S(gK0)
                        
                        sK_slice0 = cute.make_tensor((sK.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK0 = thr_copy_val.partition_D(sK_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgK0, tSsK0)
                else:
                    # HEAD_GROUP == 2
                    curr_h0 = 2 * cta_h_group
                    if curr_h0 < self.NUM_HEADS:
                        g_offset0_q = cta_b * self.q_in_strides[0] + curr_h0 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        # Q0
                        q_g_ptr0 = (q_in.iterator + g_offset0_q).align(16)
                        gQ0 = cute.make_tensor(q_g_ptr0, g_q_in_layout)
                        tSgQ0 = thr_copy_val.partition_S(gQ0)
                        
                        sQ_slice0 = cute.make_tensor((sQ.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ0 = thr_copy_val.partition_D(sQ_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgQ0, tSsQ0)
                        
                        # K0
                        g_offset0_k = cta_b * self.k_in_strides[0] + curr_h0 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr0 = (k_in.iterator + g_offset0_k).align(16)
                        gK0 = cute.make_tensor(k_g_ptr0, g_k_in_layout)
                        tSgK0 = thr_copy_val.partition_S(gK0)
                        
                        sK_slice0 = cute.make_tensor((sK.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK0 = thr_copy_val.partition_D(sK_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgK0, tSsK0)
                        
                    curr_h1 = 2 * cta_h_group + 1
                    if curr_h1 < self.NUM_HEADS:
                        g_offset1_q = cta_b * self.q_in_strides[0] + curr_h1 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        # Q1
                        q_g_ptr1 = (q_in.iterator + g_offset1_q).align(16)
                        gQ1 = cute.make_tensor(q_g_ptr1, g_q_in_layout)
                        tSgQ1 = thr_copy_val.partition_S(gQ1)
                        
                        sQ_slice1 = cute.make_tensor((sQ.iterator + stage_offset + 1 * self.HEAD_DIM).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ1 = thr_copy_val.partition_D(sQ_slice1)
                        cute.copy(gmem_tiled_copy_val, tSgQ1, tSsQ1)
                        
                        # K1
                        g_offset1_k = cta_b * self.k_in_strides[0] + curr_h1 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr1 = (k_in.iterator + g_offset1_k).align(16)
                        gK1 = cute.make_tensor(k_g_ptr1, g_k_in_layout)
                        tSgK1 = thr_copy_val.partition_S(gK1)
                        
                        sK_slice1 = cute.make_tensor((sK.iterator + stage_offset + 1 * self.HEAD_DIM).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK1 = thr_copy_val.partition_D(sK_slice1)
                        cute.copy(gmem_tiled_copy_val, tSgK1, tSsK1)

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        q_out_gmem: cute.Tensor,
        k_out_gmem: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        stream
    ):
        self.g_q_out_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.q_out_strides[2], self.q_out_strides[3])
        )
        self.g_k_out_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.k_out_strides[2], self.k_out_strides[3])
        )
        self._setup()
        
        B = q_in.shape[0]
        N = q_in.shape[2]
        
        num_threads = self.num_warps * 32
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            q_in, k_in, q_out_gmem, k_out_gmem, rmsnorm_weight, scales, sin, cos,
            self.val_smem_layout, self.g_q_in_layout, self.g_k_in_layout,
            self.g_q_out_layout, self.g_k_out_layout,
            self.gmem_tiled_copy_val
        ).launch(
            grid=[swizzled_grid_x, B, self.NUM_HEADS // self.HEAD_GROUP],
            block=[num_threads, 1, 1],
            stream=stream
        )
 
    @cute.kernel
    def kernel(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        q_out_gmem: cute.Tensor,
        k_out_gmem: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        val_smem_layout: cute.Layout,
        g_q_in_layout: cute.Layout,
        g_k_in_layout: cute.Layout,
        g_q_out_layout: cute.Layout,
        g_k_out_layout: cute.Layout,
        gmem_tiled_copy_val: cute.TiledCopy
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        grid_dim_z = cute.arch.grid_dim()[2]
        
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        
        # 2D Grid Swizzling
        swizzle_x = 8
        N = q_in.shape[2]
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
        
        # Segment and align flat shared memory
        sQ = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), val_smem_layout)
        sK = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sK_offset_f32).align(16), dtype=BFloat16), val_smem_layout)
        
        # Early indices setup mapping to the current sub-tile
        r = tidx % self.SUB_M
        h_local = tidx // self.SUB_M
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        
        # Load weights and scales Ahead of Time (AOT) to registers
        scale_val = Float32(0.0)
        q_scale_factor = Float32(1.0)
        w_q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        w_k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        
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
                w_q_reg[j] = rmsnorm_weight.element_type(1.0)
                w_k_reg[j] = rmsnorm_weight.element_type(1.0)
                
        # Load sequence block start
        seq_start = cta_m * self.BLOCK_M
        
        # Process epilogue operations in registers
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        sin_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), sin.element_type)
        cos_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), cos.element_type)
        
        q_out = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_out = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        
        # Manual unrolling of the double-buffered stages based on self.ab_stage
        if self.ab_stage == 1:
            # Prologue: Async load stage 0 Q, K
            self.load_stage(
                0, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            
            # --- Inline Compute Stage 0 ---
            seq = seq_start + r
            if is_valid and seq < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 0
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)
                
        elif self.ab_stage == 2:
            # Prologue: Async load stage 0 Q, K
            self.load_stage(
                0, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            # Load stage 1 Q, K
            self.load_stage(
                1, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            # Wait for stage 0
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()
            
            # --- Inline Compute Stage 0 ---
            seq0 = seq_start + r
            if is_valid and seq0 < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 0
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq0 * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq0 * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq0 * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq0 * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)
                
            # Wait for stage 1
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            
            # --- Inline Compute Stage 1 ---
            seq1 = seq_start + self.SUB_M + r
            if is_valid and seq1 < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 1 * self.SUB_M * self.BLOCK_N_padded
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq1 * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq1 * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq1 * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq1 * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)

_epilogue_compile_cache = {}

def get_fused_epilogue_kernel(
    num_warps=4, BLOCK_M=64, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=1, ab_stage=3,
    q_in_strides=None, k_in_strides=None, q_out_strides=None, k_out_strides=None, sin_strides=None, cos_strides=None,
    maxrregcount=255, B=32, N=512, sin_shape=None, cos_shape=None,
    q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, scales_dtype=torch.bfloat16, sin_dtype=torch.float32, cos_dtype=torch.float32
):
    B_sym = B
    N_sym = N
    if q_in_strides is None:
        q_in_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_in_strides = tuple(q_in_strides)
    if k_in_strides is None:
        k_in_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_in_strides = tuple(k_in_strides)
    if q_out_strides is None:
        q_out_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_out_strides = tuple(q_out_strides)
    if k_out_strides is None:
        k_out_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_out_strides = tuple(k_out_strides)
    if sin_strides is None:
        sin_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        sin_strides = tuple(sin_strides)
    if cos_strides is None:
        cos_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        cos_strides = tuple(cos_strides)
        
    if sin_shape is None:
        sin_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        sin_shape = tuple(sin_shape)
    if cos_shape is None:
        cos_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        cos_shape = tuple(cos_shape)
        
    if sin_shape[0] == 1:
        sin_strides = (0, sin_strides[1], sin_strides[2], sin_strides[3])
    if cos_shape[0] == 1:
        cos_strides = (0, cos_strides[1], cos_strides[2], cos_strides[3])
        
    key = (
        B, N, num_warps, BLOCK_M, NUM_HEADS, HEAD_DIM, HEAD_GROUP, ab_stage,
        q_in_strides, k_in_strides, q_out_strides, k_out_strides, sin_strides, cos_strides, maxrregcount, sin_shape, cos_shape,
        q_dtype, k_dtype, weight_dtype, scales_dtype, sin_dtype, cos_dtype
    )
    if key not in _epilogue_compile_cache:
        fwd_op = FusedEpilogueCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_in_strides=q_in_strides,
            k_in_strides=k_in_strides,
            q_out_strides=q_out_strides,
            k_out_strides=k_out_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides
        )
        
        # Map torch.dtype to CuTe types for JIT compilation
        q_cute_dtype = _torch_to_cute_dtype(q_dtype)
        k_cute_dtype = _torch_to_cute_dtype(k_dtype)
        weight_cute_dtype = _torch_to_cute_dtype(weight_dtype)
        scales_cute_dtype = _torch_to_cute_dtype(scales_dtype)
        sin_cute_dtype = _torch_to_cute_dtype(sin_dtype)
        cos_cute_dtype = _torch_to_cute_dtype(cos_dtype)
        
        q_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_in_strides, assumed_align=16)
        k_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_in_strides, assumed_align=16)
        
        q_out_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_out_strides, assumed_align=16)
        k_out_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_out_strides, assumed_align=16)
        
        rmsnorm_weight_fake = make_fake_tensor(weight_cute_dtype, (2, 1, HEAD_DIM), 16)
        scales_fake = make_fake_tensor(scales_cute_dtype, (NUM_HEADS, 1, 1), 1)
        sin_fake = cute.runtime.make_fake_tensor(sin_cute_dtype, sin_shape, stride=sin_strides, assumed_align=16)
        cos_fake = cute.runtime.make_fake_tensor(cos_cute_dtype, cos_shape, stride=cos_strides, assumed_align=16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            q_fake, k_fake, q_out_fake, k_out_fake, rmsnorm_weight_fake, scales_fake, sin_fake, cos_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _epilogue_compile_cache[key] = compiled_fwd
    return _epilogue_compile_cache[key]

def _torch_to_cute_dtype(dtype):
    if dtype == torch.bfloat16:
        return BFloat16
    elif dtype == torch.float32:
        return Float32
    elif dtype == torch.float16:
        from cutlass import Float16
        return Float16
    raise ValueError(f"Unsupported dtype: {dtype}")

def _get_compact_strides(shape, strides):
    sorted_dims = sorted(range(len(shape)), key=lambda i: (int(strides[i]), i))
    compact = list(strides)
    cur_stride = 1
    for d in sorted_dims:
        if strides[d] == 0:
            compact[d] = 0
        else:
            compact[d] = cur_stride
            cur_stride *= shape[d]
    return tuple(compact)

_epilogue_fwd_kernel_cache = {}

@torch.library.custom_op("flash_prologue::qknorm_rope_scale_cute", mutates_args=())
def qknorm_rope_scale_cute_op(
    q: torch.Tensor,
    k: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    sin = _ensure_4d(sin)
    cos = _ensure_4d(cos)
    B = int(q.shape[0])
    num_heads = int(q.shape[1])
    N = int(q.shape[2])
    head_dim = int(q.shape[3])
    
    q_out_strides = _get_compact_strides(q.shape, q.stride())
    k_out_strides = _get_compact_strides(k.shape, k.stride())
    
    q_in_strides = tuple(int(s) for s in q.stride())
    k_in_strides = tuple(int(s) for s in k.stride())
    q_out_strides_t = tuple(int(s) for s in q_out_strides)
    k_out_strides_t = tuple(int(s) for s in k_out_strides)
    
    sin_c = sin if sin.is_contiguous() else sin.contiguous()
    cos_c = cos if cos.is_contiguous() else cos.contiguous()
    sin_strides = tuple(int(s) for s in sin_c.stride())
    cos_strides = tuple(int(s) for s in cos_c.stride())
    
    cache_key = (
        B, N, num_heads, head_dim,
        q_in_strides, k_in_strides, q_out_strides_t, k_out_strides_t,
        sin_strides, cos_strides,
        tuple(sin_c.shape), tuple(cos_c.shape),
        q.dtype, k.dtype, rmsnorm_weight.dtype, scales.dtype, sin_c.dtype, cos_c.dtype
    )
    fwd_kernel = _epilogue_fwd_kernel_cache.get(cache_key)
    if fwd_kernel is None:
        config = get_autotuned_config(M=N, N=num_heads, K=head_dim, is_fused="epilogue_fwd")
        BLOCK_M = config["BLOCK_M"]
        HEAD_GROUP = config["HEAD_GROUP"]
        ab_stage = config["ab_stage"]
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        
        fwd_kernel = get_fused_epilogue_kernel(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_in_strides=q_in_strides,
            k_in_strides=k_in_strides,
            q_out_strides=q_out_strides_t,
            k_out_strides=k_out_strides_t,
            sin_strides=sin_strides,
            cos_strides=cos_strides,
            maxrregcount=maxrregcount,
            B=B,
            N=N,
            sin_shape=tuple(sin_c.shape),
            cos_shape=tuple(cos_c.shape),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            weight_dtype=rmsnorm_weight.dtype,
            scales_dtype=scales.dtype,
            sin_dtype=sin_c.dtype,
            cos_dtype=cos_c.dtype
        )
        _epilogue_fwd_kernel_cache[cache_key] = fwd_kernel
        
    rmsnorm_weight_c = rmsnorm_weight if rmsnorm_weight.is_contiguous() else rmsnorm_weight.contiguous()
    scales_c = scales if scales.is_contiguous() else scales.contiguous()
    
    q_out = torch.empty_strided(q.shape, q_out_strides, dtype=q.dtype, device=q.device)
    k_out = torch.empty_strided(k.shape, k_out_strides, dtype=k.dtype, device=k.device)
    
    try:
        fwd_kernel(
            q, k, q_out, k_out, rmsnorm_weight_c, scales_c, sin_c, cos_c
        )
    except Exception as e:
        import sys
        import tvm_ffi
        sys.stderr.write("--- DEBUG FFI FAILURE ---\n")
        sys.stderr.write(f"q shape: {q.shape}, stride: {q.stride()}, dtype: {q.dtype}\n")
        sys.stderr.write(f"k shape: {k.shape}, stride: {k.stride()}, dtype: {k.dtype}\n")
        sys.stderr.write(f"q_out shape: {q_out.shape}, stride: {q_out.stride()}, dtype: {q_out.dtype}\n")
        sys.stderr.write(f"k_out shape: {k_out.shape}, stride: {k_out.stride()}, dtype: {k_out.dtype}\n")
        sys.stderr.write(f"rmsnorm_weight shape: {rmsnorm_weight_c.shape}, stride: {rmsnorm_weight_c.stride()}, dtype: {rmsnorm_weight_c.dtype}\n")
        sys.stderr.write(f"scales shape: {scales_c.shape}, stride: {scales_c.stride()}, dtype: {scales_c.dtype}\n")
        sys.stderr.write(f"sin shape: {sin_c.shape}, stride: {sin_c.stride()}, dtype: {sin_c.dtype}\n")
        sys.stderr.write(f"cos shape: {cos_c.shape}, stride: {cos_c.stride()}, dtype: {cos_c.dtype}\n")
        sys.stderr.flush()
        raise e
    # Preserve input lifetimes to prevent eager allocator from reusing memory while FFI kernel runs
    inputs_tuple = (q, k, rmsnorm_weight_c, scales_c, sin_c, cos_c)
    q_out._base_inputs = inputs_tuple
    k_out._base_inputs = inputs_tuple
    return q_out, k_out

@qknorm_rope_scale_cute_op.register_fake
def _qknorm_rope_scale_cute_fake(q, k, rmsnorm_weight, scales, sin, cos, eps=1e-5):
    q_out_strides = _get_compact_strides(q.shape, q.stride())
    k_out_strides = _get_compact_strides(k.shape, k.stride())
    return (
        torch.empty_strided(q.shape, q_out_strides, dtype=q.dtype, device=q.device),
        torch.empty_strided(k.shape, k_out_strides, dtype=k.dtype, device=k.device)
    )


class FusedEpilogueBackwardCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 1, ab_stage: int = 1, q_strides=None, k_strides=None, sin_strides=None, cos_strides=None):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.NUM_HEADS = NUM_HEADS
        self.ab_stage = 1
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.sin_strides = sin_strides
        self.cos_strides = cos_strides
        
        # Calculate shared memory footprint for reduction
        # s_reduction: shape (BLOCK_M, 2, HEAD_GROUP, HEAD_DIM) -> BLOCK_M * 2 * HEAD_GROUP * HEAD_DIM Float32 elements
        # s_scales_reduction: shape (BLOCK_M, HEAD_GROUP) -> BLOCK_M * HEAD_GROUP Float32 elements
        s_rmsnorm_reduction_elems = self.BLOCK_M * 2 * self.HEAD_GROUP * self.HEAD_DIM
        # Align offset of scales to multiple of 32 Float32 elements (128 bytes)
        self.s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
        
        total_elems = self.s_scales_offset + self.BLOCK_M * self.HEAD_GROUP
        # Align total elements to multiple of 32 Float32 elements
        self.flat_smem_elements = ((total_elems + 31) // 32) * 32
        self.max_smem_bytes = self.flat_smem_elements * 4

    def _setup(self):
        self.g_q_layout = cute.make_layout(
            (self.BLOCK_M, self.HEAD_DIM),
            stride=(self.q_strides[2], self.q_strides[3])
        )
        self.g_k_layout = cute.make_layout(
            (self.BLOCK_M, self.HEAD_DIM),
            stride=(self.k_strides[2], self.k_strides[3])
        )
        
        @cute.struct
        class SharedStorageEpilogueBack:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageEpilogueBack
 
    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        grad_q_out: cute.Tensor,
        grad_k_out: cute.Tensor,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_rmsnorm_weight: cute.Tensor,
        grad_scales: cute.Tensor,
        stream
    ):
        self._setup()
        
        B = q.shape[0]
        N = q.shape[2]
        
        num_threads = self.num_warps * 32
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        
        self.kernel(
            q, k, rmsnorm_weight, scales, sin, cos,
            grad_q_out, grad_k_out, grad_q, grad_k,
            grad_rmsnorm_weight, grad_scales,
            self.g_q_layout, self.g_k_layout
        ).launch(
            grid=[grid_x_blocks, B, self.NUM_HEADS // self.HEAD_GROUP],
            block=[num_threads, 1, 1],
            stream=stream
        )
 
    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        grad_q_out: cute.Tensor,
        grad_k_out: cute.Tensor,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_rmsnorm_weight: cute.Tensor,
        grad_scales: cute.Tensor,
        g_q_layout: cute.Layout,
        g_k_layout: cute.Layout
    ):
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        
        cta_m = block_idx_x
        cta_b = block_idx_y
        cta_h_group = block_idx_z
        
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = self.num_warps * 32
        
        # Shared memory reduction buffers allocation and zeroing
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        s_reduction = cute.make_tensor(
            cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=Float32),
            cute.make_layout(
                (self.BLOCK_M, 2, self.HEAD_GROUP, self.HEAD_DIM),
                stride=(2 * self.HEAD_GROUP * self.HEAD_DIM, self.HEAD_GROUP * self.HEAD_DIM, self.HEAD_DIM, 1)
            )
        )
        s_scales_reduction = cute.make_tensor(
            cute.recast_ptr((storage.sX_raw.data_ptr() + self.s_scales_offset).align(16), dtype=Float32),
            cute.make_layout((self.BLOCK_M, self.HEAD_GROUP), stride=(self.HEAD_GROUP, 1))
        )
        
        s_flat = cute.make_tensor(
            cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=Float32),
            cute.make_layout((self.flat_smem_elements,))
        )
        
        for idx in cutlass.range(tidx, self.flat_smem_elements, num_threads):
            s_flat[idx] = Float32(0.0)
            
        cute.arch.sync_threads()
        
        # Early indices setup
        r = tidx % self.BLOCK_M
        h_local = tidx // self.BLOCK_M
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        
        # Load weights and scales Ahead of Time (AOT) to registers
        scale_val = Float32(0.0)
        q_scale_factor = Float32(1.0)
        w_q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        w_k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        
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
                w_q_reg[j] = rmsnorm_weight.element_type(1.0)
                w_k_reg[j] = rmsnorm_weight.element_type(1.0)
                
        N = q.shape[2]
        
        # Process epilogue operations in registers: preallocated outside the loop to avoid stack spilling
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dq_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dk_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        sin_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), sin.element_type)
        cos_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), cos.element_type)
        
        q_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        k_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dq_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dk_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dq_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dk_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        q_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        dq_in = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dk_in = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)

        # Setup sequence indices for the assigned block tile
        seq_start = cta_m * self.BLOCK_M
        seq = seq_start + r
        
        if seq < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
            # Direct GMEM load for q, k, grad_q_out, grad_k_out
            q_offset = cta_b * self.q_strides[0] + global_h * self.q_strides[1] + seq * self.q_strides[2]
            k_offset = cta_b * self.k_strides[0] + global_h * self.k_strides[1] + seq * self.k_strides[2]
            
            q_ptr = (q.iterator + q_offset).align(16)
            k_ptr = (k.iterator + k_offset).align(16)
            dq_out_ptr = (grad_q_out.iterator + q_offset).align(16)
            dk_out_ptr = (grad_k_out.iterator + k_offset).align(16)
            
            g_q = cute.make_tensor(q_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_k = cute.make_tensor(k_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            g_dq_out = cute.make_tensor(dq_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_dk_out = cute.make_tensor(dk_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            
            cute.autovec_copy(g_q, q_reg)
            cute.autovec_copy(g_k, k_reg)
            cute.autovec_copy(g_dq_out, dq_out_reg)
            cute.autovec_copy(g_dk_out, dk_out_reg)
            
            # Direct GMEM load for sin/cos
            sin_offset = cta_b * self.sin_strides[0] + seq * self.sin_strides[2]
            cos_offset = cta_b * self.cos_strides[0] + seq * self.cos_strides[2]
            
            sin_ptr = (sin.iterator + sin_offset).align(16)
            cos_ptr = (cos.iterator + cos_offset).align(16)
            
            g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
            g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
            
            cute.autovec_copy(g_sin, sin_reg)
            cute.autovec_copy(g_cos, cos_reg)
            
            # 1. Compute on-the-fly RMSNorm stats (rsqrt) for Q and K
            q_sum_sq = Float32(0.0)
            k_sum_sq = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                val_q = Float32(q_reg[j])
                val_k = Float32(k_reg[j])
                q_sum_sq += val_q * val_q
                k_sum_sq += val_k * val_k
                
            q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
            k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
            
            # 2. Recompute normalized Q and K (normed) in registers
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                q_normed[j] = Float32(q_reg[j]) * q_rsqrt
                k_normed[j] = Float32(k_reg[j]) * k_rsqrt
                
            # 3. Propagate grad_q_out through the query scale factor
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                dq_roped[j] = Float32(dq_out_reg[j]) * q_scale_factor
                dk_roped[j] = Float32(dk_out_reg[j])
                
            # 4. Propagate through RoPE backward (inverse rotation)
            half_dim = self.HEAD_DIM // 2
            
            for j in cutlass.range(half_dim, unroll_full=True):
                cos_val = Float32(cos_reg[j])
                sin_val = Float32(sin_reg[j])
                
                dq_normed[j] = dq_roped[j] * cos_val + dq_roped[j + half_dim] * sin_val
                dq_normed[j + half_dim] = dq_roped[j + half_dim] * cos_val - dq_roped[j] * sin_val
                
                dk_normed[j] = dk_roped[j] * cos_val + dk_roped[j + half_dim] * sin_val
                dk_normed[j + half_dim] = dk_roped[j + half_dim] * cos_val - dk_roped[j] * sin_val
                
            # 5. Compute parameter gradients (weights and scales)
            # Scales gradient
            # First recompute q_roped
            for j in cutlass.range(half_dim, unroll_full=True):
                cos_val = Float32(cos_reg[j])
                sin_val = Float32(sin_reg[j])
                
                qn1 = q_normed[j] * Float32(w_q_reg[j])
                qn2 = q_normed[j + half_dim] * Float32(w_q_reg[j + half_dim])
                q_roped[j] = qn1 * cos_val - qn2 * sin_val
                q_roped[j + half_dim] = qn2 * cos_val + qn1 * sin_val
                
            grad_scale_local = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                grad_scale_local += dq_roped[j] * q_roped[j]
                
            # Store in shared memory reduction buffers
            s_scales_reduction[r, h_local] = grad_scale_local
            
            # RMSNorm weights gradient
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                s_reduction[r, 0, h_local, j] = dq_normed[j] * q_normed[j]
                s_reduction[r, 1, h_local, j] = dk_normed[j] * k_normed[j]
                
            # 6. Compute input gradients (grad_q and grad_k)
            sum_dq_q = Float32(0.0)
            sum_dk_k = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                sum_dq_q += dq_normed[j] * Float32(w_q_reg[j]) * q_normed[j]
                sum_dk_k += dk_normed[j] * Float32(w_k_reg[j]) * k_normed[j]
                
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                dq_in_val = q_rsqrt * (dq_normed[j] * Float32(w_q_reg[j]) - q_normed[j] * (sum_dq_q / Float32(self.HEAD_DIM)))
                dk_in_val = k_rsqrt * (dk_normed[j] * Float32(w_k_reg[j]) - k_normed[j] * (sum_dk_k / Float32(self.HEAD_DIM)))
                dq_in[j] = BFloat16(dq_in_val)
                dk_in[j] = BFloat16(dk_in_val)
                
            # Vectorized global store of input gradients
            dq_in_ptr = (grad_q.iterator + q_offset).align(16)
            dk_in_ptr = (grad_k.iterator + k_offset).align(16)
            
            g_out_dq = cute.make_tensor(dq_in_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_out_dk = cute.make_tensor(dk_in_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            
            cute.autovec_copy(dq_in, g_out_dq)
            cute.autovec_copy(dk_in, g_out_dk)
                
        # Synchronize before cooperative global memory writeback
        cute.arch.sync_threads()
        
        # Cooperative writeback to global memory
        # Parallel tree reduction in shared memory for scales gradient
        step = self.BLOCK_M // 2
        while step > 0:
            total_work = step * self.HEAD_GROUP
            for work_idx in cutlass.range(tidx, total_work, num_threads):
                r_idx = work_idx % step
                hl = work_idx // step
                s_scales_reduction[r_idx, hl] += s_scales_reduction[r_idx + step, hl]
            
            cute.arch.sync_threads()
            step = step // 2

        # Write back the final scales gradient
        for hl in cutlass.range(tidx, self.HEAD_GROUP, num_threads):
            curr_global_h = self.HEAD_GROUP * cta_h_group + hl
            if curr_global_h < self.NUM_HEADS:
                sum_scales = s_scales_reduction[0, hl]
                cute.arch.atomic_add(grad_scales[curr_global_h, 0, None].iterator, sum_scales)
                
        # Cooperative reduction and write back of rmsnorm weights gradient (vectorized 128-bit LDS)
        reduce_groups = 2 * self.HEAD_GROUP * (self.HEAD_DIM // 4)
        num_iters = (reduce_groups + num_threads - 1) // num_threads
        for iter_idx in cutlass.range_constexpr(num_iters):
            idx = tidx + iter_idx * num_threads
            if idx < reduce_groups:
                reg_sum = cute.make_rmem_tensor((4,), Float32)
                reg_val = cute.make_rmem_tensor((4,), Float32)
                
                weight_idx = idx // (self.HEAD_GROUP * (self.HEAD_DIM // 4))
                rem = idx % (self.HEAD_GROUP * (self.HEAD_DIM // 4))
                hl = rem // (self.HEAD_DIM // 4)
                j_group = rem % (self.HEAD_DIM // 4)
                j_start = j_group * 4
                
                reg_sum[0] = Float32(0.0)
                reg_sum[1] = Float32(0.0)
                reg_sum[2] = Float32(0.0)
                reg_sum[3] = Float32(0.0)
                
                for r_idx in cutlass.range(self.BLOCK_M, unroll_full=True):
                    s_offset = (
                        r_idx * (2 * self.HEAD_GROUP * self.HEAD_DIM) +
                        weight_idx * (self.HEAD_GROUP * self.HEAD_DIM) +
                        hl * self.HEAD_DIM +
                        j_start
                    )
                    
                    s_slice_ptr = s_reduction.iterator + s_offset
                    s_slice = cute.make_tensor(s_slice_ptr.align(16), cute.make_layout((4,)))
                    
                    # Vectorized 128-bit LDS load
                    cute.autovec_copy(s_slice, reg_val)
                    
                    reg_sum[0] += reg_val[0]
                    reg_sum[1] += reg_val[1]
                    reg_sum[2] += reg_val[2]
                    reg_sum[3] += reg_val[3]
                        
                curr_global_h = self.HEAD_GROUP * cta_h_group + hl
                if curr_global_h < self.NUM_HEADS:
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 0,
                        reg_sum[0]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 1,
                        reg_sum[1]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 2,
                        reg_sum[2]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 3,
                        reg_sum[3]
                    )

_epilogue_back_compile_cache = {}

def get_fused_epilogue_back_kernel(
    num_warps=4, BLOCK_M=64, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=1, ab_stage=1,
    q_strides=None, k_strides=None, sin_strides=None, cos_strides=None,
    maxrregcount=255, B=32, N=512, sin_shape=None, cos_shape=None,
    q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, scales_dtype=torch.bfloat16, sin_dtype=torch.float32, cos_dtype=torch.float32
):
    B_sym = B
    N_sym = N
    if q_strides is None:
        q_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_strides = tuple(q_strides)
    if k_strides is None:
        k_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_strides = tuple(k_strides)
    if sin_strides is None:
        sin_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        sin_strides = tuple(sin_strides)
    if cos_strides is None:
        cos_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        cos_strides = tuple(cos_strides)
        
    if sin_shape is None:
        sin_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        sin_shape = tuple(sin_shape)
    if cos_shape is None:
        cos_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        cos_shape = tuple(cos_shape)
        
    if sin_shape[0] == 1:
        sin_strides = (0, sin_strides[1], sin_strides[2], sin_strides[3])
    if cos_shape[0] == 1:
        cos_strides = (0, cos_strides[1], cos_strides[2], cos_strides[3])
        
    key = (
        B, N, num_warps, BLOCK_M, NUM_HEADS, HEAD_DIM, HEAD_GROUP, ab_stage,
        q_strides, k_strides, sin_strides, cos_strides, maxrregcount, sin_shape, cos_shape,
        q_dtype, k_dtype, weight_dtype, scales_dtype, sin_dtype, cos_dtype
    )
    if key not in _epilogue_back_compile_cache:
        back_op = FusedEpilogueBackwardCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_strides=q_strides,
            k_strides=k_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides
        )
        
        q_cute_dtype = _torch_to_cute_dtype(q_dtype)
        k_cute_dtype = _torch_to_cute_dtype(k_dtype)
        weight_cute_dtype = _torch_to_cute_dtype(weight_dtype)
        scales_cute_dtype = _torch_to_cute_dtype(scales_dtype)
        sin_cute_dtype = _torch_to_cute_dtype(sin_dtype)
        cos_cute_dtype = _torch_to_cute_dtype(cos_dtype)
        
        q_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        k_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        rmsnorm_weight_fake = make_fake_tensor(weight_cute_dtype, (2, 1, HEAD_DIM), 16)
        scales_fake = make_fake_tensor(scales_cute_dtype, (NUM_HEADS, 1, 1), 1)
        sin_fake = cute.runtime.make_fake_tensor(sin_cute_dtype, sin_shape, stride=sin_strides, assumed_align=16)
        cos_fake = cute.runtime.make_fake_tensor(cos_cute_dtype, cos_shape, stride=cos_strides, assumed_align=16)
        grad_q_out_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        grad_k_out_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        grad_q_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        grad_k_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        grad_rmsnorm_weight_fake = make_fake_tensor(Float32, (2, 1, HEAD_DIM), 16)
        grad_scales_fake = make_fake_tensor(Float32, (NUM_HEADS, 1, 1), 1)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_back = cute.compile(
            back_op,
            q_fake, k_fake, rmsnorm_weight_fake, scales_fake, sin_fake, cos_fake,
            grad_q_out_fake, grad_k_out_fake, grad_q_fake, grad_k_fake,
            grad_rmsnorm_weight_fake, grad_scales_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _epilogue_back_compile_cache[key] = compiled_back
    return _epilogue_back_compile_cache[key]

_epilogue_bwd_kernel_cache = {}

@torch.library.custom_op("flash_prologue::qknorm_rope_scale_cute_backward", mutates_args=())
def qknorm_rope_scale_cute_backward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sin = _ensure_4d(sin)
    cos = _ensure_4d(cos)
    B = int(q.shape[0])
    num_heads = int(q.shape[1])
    N = int(q.shape[2])
    head_dim = int(q.shape[3])
    q_strides_compact = _get_compact_strides(q.shape, q.stride())
    k_strides_compact = _get_compact_strides(k.shape, k.stride())


    q_matched = q if q.stride() == q_strides_compact else torch.empty_strided(q.shape, q_strides_compact, dtype=q.dtype, device=q.device).copy_(q)
    k_matched = k if k.stride() == k_strides_compact else torch.empty_strided(k.shape, k_strides_compact, dtype=k.dtype, device=k.device).copy_(k)
    
    grad_q_out_matched = grad_q_out if grad_q_out.stride() == q_strides_compact else torch.empty_strided(q.shape, q_strides_compact, dtype=grad_q_out.dtype, device=grad_q_out.device).copy_(grad_q_out)
    grad_k_out_matched = grad_k_out if grad_k_out.stride() == k_strides_compact else torch.empty_strided(k.shape, k_strides_compact, dtype=grad_k_out.dtype, device=grad_k_out.device).copy_(grad_k_out)

    # Zero out potential garbage in the padded region of grad_q_out/grad_k_out
    active_mask = (cos.abs().sum(dim=(0, 1, 3)) > 1e-6).to(dtype=grad_q_out_matched.dtype, device=grad_q_out_matched.device)
    active_mask_4d = active_mask.view(1, 1, -1, 1)
    grad_q_out_matched = grad_q_out_matched * active_mask_4d
    grad_k_out_matched = grad_k_out_matched * active_mask_4d

    rmsnorm_weight_c = rmsnorm_weight if rmsnorm_weight.is_contiguous() else rmsnorm_weight.contiguous()
    scales_c = scales if scales.is_contiguous() else scales.contiguous()
    sin_c = sin if sin.is_contiguous() else sin.contiguous()
    cos_c = cos if cos.is_contiguous() else cos.contiguous()



    q_strides = tuple(int(s) for s in q_strides_compact)
    k_strides = tuple(int(s) for s in k_strides_compact)
    
    sin_strides = tuple(int(s) for s in sin_c.stride())
    cos_strides = tuple(int(s) for s in cos_c.stride())
    
    cache_key = (
        B, N, num_heads, head_dim,
        q_strides, k_strides, sin_strides, cos_strides,
        tuple(sin_c.shape), tuple(cos_c.shape),
        q.dtype, k.dtype, rmsnorm_weight.dtype, scales.dtype, sin_c.dtype, cos_c.dtype
    )
    back_kernel = _epilogue_bwd_kernel_cache.get(cache_key)
    if back_kernel is None:
        config = get_autotuned_config(M=N, N=num_heads, K=head_dim, is_fused="epilogue_bwd")
        BLOCK_M = config["BLOCK_M"]
        HEAD_GROUP = config["HEAD_GROUP"]
        ab_stage = config["ab_stage"]
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        
        back_kernel = get_fused_epilogue_back_kernel(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_strides=q_strides,
            k_strides=k_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides,
            maxrregcount=maxrregcount,
            B=B,
            N=N,
            sin_shape=tuple(sin_c.shape),
            cos_shape=tuple(cos_c.shape),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            weight_dtype=rmsnorm_weight.dtype,
            scales_dtype=scales.dtype,
            sin_dtype=sin_c.dtype,
            cos_dtype=cos_c.dtype
        )
        _epilogue_bwd_kernel_cache[cache_key] = back_kernel
        
    grad_q = torch.empty_strided(q.shape, q_strides_compact, dtype=q.dtype, device=q.device)
    grad_k = torch.empty_strided(k.shape, k_strides_compact, dtype=k.dtype, device=k.device)
    grad_rmsnorm_weight = torch.zeros((2, 1, head_dim), device=q.device, dtype=torch.float32)
    grad_scales = torch.zeros((num_heads, 1, 1), device=q.device, dtype=torch.float32)
    
    rmsnorm_weight_c = rmsnorm_weight if rmsnorm_weight.is_contiguous() else rmsnorm_weight.contiguous()
    scales_c = scales if scales.is_contiguous() else scales.contiguous()
    

    
    back_kernel(
        q_matched, k_matched, rmsnorm_weight_c, scales_c, sin_c, cos_c,
        grad_q_out_matched, grad_k_out_matched, grad_q, grad_k,
        grad_rmsnorm_weight, grad_scales
    )
    

    
    # Preserve input lifetimes to prevent eager allocator from reusing memory while FFI kernel runs
    inputs_tuple = (q_matched, k_matched, rmsnorm_weight_c, scales_c, sin_c, cos_c, grad_q_out_matched, grad_k_out_matched)
    grad_q._base_inputs = inputs_tuple
    grad_k._base_inputs = inputs_tuple
    grad_rmsnorm_weight._base_inputs = inputs_tuple
    grad_scales._base_inputs = inputs_tuple
    return grad_q, grad_k, grad_rmsnorm_weight, grad_scales

@qknorm_rope_scale_cute_backward_op.register_fake
def _qknorm_rope_scale_cute_backward_fake(q, k, rmsnorm_weight, scales, sin, cos, grad_q_out, grad_k_out, eps=1e-5):
    q_strides_compact = _get_compact_strides(q.shape, q.stride())
    k_strides_compact = _get_compact_strides(k.shape, k.stride())
    grad_q = torch.empty_strided(q.shape, q_strides_compact, dtype=q.dtype, device=q.device)
    grad_k = torch.empty_strided(k.shape, k_strides_compact, dtype=k.dtype, device=k.device)
    grad_rmsnorm_weight = torch.empty((2, 1, q.shape[-1]), device=q.device, dtype=torch.float32)
    grad_scales = torch.empty((q.shape[1], 1, 1), device=q.device, dtype=torch.float32)
    return grad_q, grad_k, grad_rmsnorm_weight, grad_scales


def setup_qknorm_rope_scale_cute_context(ctx, inputs, output):
    q, k, rmsnorm_weight, scales, sin, cos, eps = inputs
    ctx.save_for_backward(q, k, rmsnorm_weight, scales, sin, cos)
    ctx.eps = eps

def qknorm_rope_scale_cute_backward_hook(ctx, grad_q, grad_k):
    q, k, rmsnorm_weight, scales, sin, cos = ctx.saved_tensors
    eps = ctx.eps
    
    dq, dk, dw, ds = qknorm_rope_scale_cute_backward_op(
        q, k, rmsnorm_weight, scales, sin, cos, grad_q, grad_k, eps
    )
    return dq, dk, dw, ds, None, None, None

torch.library.register_autograd(
    "flash_prologue::qknorm_rope_scale_cute",
    qknorm_rope_scale_cute_backward_hook,
    setup_context=setup_qknorm_rope_scale_cute_context
)


# --- Inverse Epilogue Custom Operator Implementation ---

class FusedEpilogueInverseCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 1, ab_stage: int = 1, q_in_strides=None, k_in_strides=None, q_out_strides=None, k_out_strides=None, sin_strides=None, cos_strides=None):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.BLOCK_N = HEAD_GROUP * HEAD_DIM
        self.BLOCK_N_padded = self.BLOCK_N + 8
        self.NUM_HEADS = NUM_HEADS
        self.ab_stage = ab_stage
        self.SUB_M = BLOCK_M // ab_stage
        self.q_in_strides = q_in_strides
        self.k_in_strides = k_in_strides
        self.q_out_strides = q_out_strides
        self.k_out_strides = k_out_strides
        self.sin_strides = sin_strides
        self.cos_strides = cos_strides
        
        sQ_size = self.SUB_M * self.BLOCK_N_padded * self.ab_stage * 2
        sK_size = self.SUB_M * self.BLOCK_N_padded * self.ab_stage * 2
        
        self.max_smem_bytes = sQ_size + sK_size
        self.flat_smem_elements = (self.max_smem_bytes + 3) // 4
        self.sK_offset_f32 = sQ_size // 4

    def _setup(self):
        self.val_smem_layout = cute.make_layout(
            (self.SUB_M, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N_padded, 1, self.SUB_M * self.BLOCK_N_padded)
        )
        
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom_bf16 = cute.make_copy_atom(copy_op, BFloat16, num_bits_per_copy=128)
        
        num_threads = self.num_warps * 32
        copy_val_val_k = min(8, self.HEAD_DIM)
        copy_val_thr_k = self.HEAD_DIM // copy_val_val_k
        max_thr_m = num_threads // copy_val_thr_k
        copy_val_thr_m = 1
        for d in range(1, self.SUB_M + 1):
            if self.SUB_M % d == 0 and d <= max_thr_m:
                copy_val_thr_m = d
        self.copy_val_threads = copy_val_thr_m * copy_val_thr_k
        self.gmem_tiled_copy_val = cute.make_tiled_copy_tv(
            copy_atom_bf16,
            cute.make_ordered_layout((copy_val_thr_m, copy_val_thr_k), order=(1, 0)),
            cute.make_layout((1, copy_val_val_k))
        )
        
        self.g_q_in_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.q_in_strides[2], self.q_in_strides[3])
        )
        self.g_k_in_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.k_in_strides[2], self.k_in_strides[3])
        )
        
        @cute.struct
        class SharedStorageEpilogue:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageEpilogue
 
    @cute.jit
    def load_stage(
        self,
        stage_idx,
        seq_start,
        N,
        q_in,
        k_in,
        sQ,
        sK,
        tidx,
        is_valid,
        cta_b,
        cta_h_group,
        gmem_tiled_copy_val,
        g_q_in_layout,
        g_k_in_layout
    ):
        stage_seq_start = seq_start + stage_idx * self.SUB_M
        clamped_seq = cutlass.min(stage_seq_start, N - 1)
        stage_offset = stage_idx * self.SUB_M * self.BLOCK_N_padded
        
        if is_valid:
            if tidx < self.copy_val_threads:
                if self.HEAD_GROUP == 1:
                    curr_h0 = cta_h_group
                    if curr_h0 < self.NUM_HEADS:
                        g_offset0_q = cta_b * self.q_in_strides[0] + curr_h0 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        q_g_ptr0 = (q_in.iterator + g_offset0_q).align(16)
                        gQ0 = cute.make_tensor(q_g_ptr0, g_q_in_layout)
                        tSgQ0 = thr_copy_val.partition_S(gQ0)
                        
                        sQ_slice0 = cute.make_tensor((sQ.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ0 = thr_copy_val.partition_D(sQ_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgQ0, tSsQ0)
                        
                        g_offset0_k = cta_b * self.k_in_strides[0] + curr_h0 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr0 = (k_in.iterator + g_offset0_k).align(16)
                        gK0 = cute.make_tensor(k_g_ptr0, g_k_in_layout)
                        tSgK0 = thr_copy_val.partition_S(gK0)
                        
                        sK_slice0 = cute.make_tensor((sK.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK0 = thr_copy_val.partition_D(sK_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgK0, tSsK0)
                else:
                    curr_h0 = 2 * cta_h_group
                    if curr_h0 < self.NUM_HEADS:
                        g_offset0_q = cta_b * self.q_in_strides[0] + curr_h0 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        q_g_ptr0 = (q_in.iterator + g_offset0_q).align(16)
                        gQ0 = cute.make_tensor(q_g_ptr0, g_q_in_layout)
                        tSgQ0 = thr_copy_val.partition_S(gQ0)
                        
                        sQ_slice0 = cute.make_tensor((sQ.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ0 = thr_copy_val.partition_D(sQ_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgQ0, tSsQ0)
                        
                        g_offset0_k = cta_b * self.k_in_strides[0] + curr_h0 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr0 = (k_in.iterator + g_offset0_k).align(16)
                        gK0 = cute.make_tensor(k_g_ptr0, g_k_in_layout)
                        tSgK0 = thr_copy_val.partition_S(gK0)
                        
                        sK_slice0 = cute.make_tensor((sK.iterator + stage_offset).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK0 = thr_copy_val.partition_D(sK_slice0)
                        cute.copy(gmem_tiled_copy_val, tSgK0, tSsK0)
                        
                    curr_h1 = 2 * cta_h_group + 1
                    if curr_h1 < self.NUM_HEADS:
                        g_offset1_q = cta_b * self.q_in_strides[0] + curr_h1 * self.q_in_strides[1] + clamped_seq * self.q_in_strides[2]
                        thr_copy_val = gmem_tiled_copy_val.get_slice(tidx)
                        
                        q_g_ptr1 = (q_in.iterator + g_offset1_q).align(16)
                        gQ1 = cute.make_tensor(q_g_ptr1, g_q_in_layout)
                        tSgQ1 = thr_copy_val.partition_S(gQ1)
                        
                        sQ_slice1 = cute.make_tensor((sQ.iterator + stage_offset + 1 * self.HEAD_DIM).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsQ1 = thr_copy_val.partition_D(sQ_slice1)
                        cute.copy(gmem_tiled_copy_val, tSgQ1, tSsQ1)
                        
                        g_offset1_k = cta_b * self.k_in_strides[0] + curr_h1 * self.k_in_strides[1] + clamped_seq * self.k_in_strides[2]
                        k_g_ptr1 = (k_in.iterator + g_offset1_k).align(16)
                        gK1 = cute.make_tensor(k_g_ptr1, g_k_in_layout)
                        tSgK1 = thr_copy_val.partition_S(gK1)
                        
                        sK_slice1 = cute.make_tensor((sK.iterator + stage_offset + 1 * self.HEAD_DIM).align(16), cute.make_layout((self.SUB_M, self.HEAD_DIM), stride=(self.BLOCK_N_padded, 1)))
                        tSsK1 = thr_copy_val.partition_D(sK_slice1)
                        cute.copy(gmem_tiled_copy_val, tSgK1, tSsK1)

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        q_out_gmem: cute.Tensor,
        k_out_gmem: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        rms_q: cute.Tensor,
        rms_k: cute.Tensor,
        stream
    ):
        self.g_q_out_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.q_out_strides[2], self.q_out_strides[3])
        )
        self.g_k_out_layout = cute.make_layout(
            (self.SUB_M, self.HEAD_DIM),
            stride=(self.k_out_strides[2], self.k_out_strides[3])
        )
        self._setup()
        
        B = q_in.shape[0]
        N = q_in.shape[2]
        
        num_threads = self.num_warps * 32
        
        swizzle_x = 8
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        swizzled_grid_x = cute.ceil_div(grid_x_blocks, swizzle_x) * swizzle_x
        
        self.kernel(
            q_in, k_in, q_out_gmem, k_out_gmem, rmsnorm_weight, scales, sin, cos, rms_q, rms_k,
            self.val_smem_layout, self.g_q_in_layout, self.g_k_in_layout,
            self.g_q_out_layout, self.g_k_out_layout,
            self.gmem_tiled_copy_val
        ).launch(
            grid=[swizzled_grid_x, B, self.NUM_HEADS // self.HEAD_GROUP],
            block=[num_threads, 1, 1],
            stream=stream
        )
 
    @cute.kernel
    def kernel(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        q_out_gmem: cute.Tensor,
        k_out_gmem: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        rms_q: cute.Tensor,
        rms_k: cute.Tensor,
        val_smem_layout: cute.Layout,
        g_q_in_layout: cute.Layout,
        g_k_in_layout: cute.Layout,
        g_q_out_layout: cute.Layout,
        g_k_out_layout: cute.Layout,
        gmem_tiled_copy_val: cute.TiledCopy
    ):
        grid_dim_x = cute.arch.grid_dim()[0]
        grid_dim_z = cute.arch.grid_dim()[2]
        
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        
        swizzle_x = 8
        N = q_in.shape[2]
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
        
        sQ = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), val_smem_layout)
        sK = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sK_offset_f32).align(16), dtype=BFloat16), val_smem_layout)
        
        r = tidx % self.SUB_M
        h_local = tidx // self.SUB_M
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        
        scale_val = Float32(0.0)
        q_scale_factor = Float32(1.0)
        w_q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        w_k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        
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
                w_q_reg[j] = rmsnorm_weight.element_type(1.0)
                w_k_reg[j] = rmsnorm_weight.element_type(1.0)
                
        seq_start = cta_m * self.BLOCK_M
        
        q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        sin_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), sin.element_type)
        cos_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), cos.element_type)
        
        q_out = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_out = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        
        if self.ab_stage == 1:
            self.load_stage(
                0, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            
            seq = seq_start + r
            if is_valid and seq < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 0
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                rms_q[cta_b, global_h, seq, 0] = q_rsqrt
                rms_k[cta_b, global_h, seq, 0] = k_rsqrt
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)
                
        elif self.ab_stage == 2:
            self.load_stage(
                0, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            self.load_stage(
                1, seq_start, N, q_in, k_in, sQ, sK, tidx, is_valid, cta_b, cta_h_group,
                gmem_tiled_copy_val, g_q_in_layout, g_k_in_layout
            )
            cute.arch.cp_async_commit_group()
            
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()
            
            seq0 = seq_start + r
            if is_valid and seq0 < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 0
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq0 * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq0 * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                rms_q[cta_b, global_h, seq0, 0] = q_rsqrt
                rms_k[cta_b, global_h, seq0, 0] = k_rsqrt
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq0 * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq0 * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)
                
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            
            seq1 = seq_start + self.SUB_M + r
            if is_valid and seq1 < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
                stage_offset = 1 * self.SUB_M * self.BLOCK_N_padded
                row_offset_val = stage_offset + r * self.BLOCK_N_padded + h_local * self.HEAD_DIM
                
                q_s = cute.make_tensor(sQ.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                k_s = cute.make_tensor(sK.iterator + row_offset_val, cute.make_layout((self.HEAD_DIM,)))
                
                sin_offset = cta_b * self.sin_strides[0] + seq1 * self.sin_strides[2]
                cos_offset = cta_b * self.cos_strides[0] + seq1 * self.cos_strides[2]
                
                sin_ptr = (sin.iterator + sin_offset).align(16)
                cos_ptr = (cos.iterator + cos_offset).align(16)
                
                g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
                g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
                
                cute.autovec_copy(q_s, q_reg)
                cute.autovec_copy(k_s, k_reg)
                cute.autovec_copy(g_sin, sin_reg)
                cute.autovec_copy(g_cos, cos_reg)
                
                q_sum_sq = Float32(0.0)
                k_sum_sq = Float32(0.0)
                for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                    val_q = Float32(q_reg[j])
                    val_k = Float32(k_reg[j])
                    q_sum_sq += val_q * val_q
                    k_sum_sq += val_k * val_k
                    
                q_rsqrt = cute.math.rsqrt((q_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                k_rsqrt = cute.math.rsqrt((k_sum_sq / Float32(self.HEAD_DIM)) + Float32(1e-5), fastmath=False)
                
                rms_q[cta_b, global_h, seq1, 0] = q_rsqrt
                rms_k[cta_b, global_h, seq1, 0] = k_rsqrt
                
                half_dim = self.HEAD_DIM // 2
                for j in cutlass.range(half_dim, unroll_full=True):
                    qn1 = Float32(q_reg[j]) * q_rsqrt * Float32(w_q_reg[j])
                    qn2 = Float32(q_reg[j + half_dim]) * q_rsqrt * Float32(w_q_reg[j + half_dim])
                    
                    kn1 = Float32(k_reg[j]) * k_rsqrt * Float32(w_k_reg[j])
                    kn2 = Float32(k_reg[j + half_dim]) * k_rsqrt * Float32(w_k_reg[j + half_dim])
                    
                    cos_val = Float32(cos_reg[j])
                    sin_val = Float32(sin_reg[j])
                    
                    q_final1 = (qn1 * cos_val - qn2 * sin_val) * q_scale_factor
                    q_final2 = (qn2 * cos_val + qn1 * sin_val) * q_scale_factor
                    
                    k_final1 = kn1 * cos_val - kn2 * sin_val
                    k_final2 = kn2 * cos_val + kn1 * sin_val
                    
                    q_out[j] = BFloat16(q_final1)
                    q_out[j + half_dim] = BFloat16(q_final2)
                    k_out[j] = BFloat16(k_final1)
                    k_out[j + half_dim] = BFloat16(k_final2)
                    
                q_out_offset = cta_b * self.q_out_strides[0] + global_h * self.q_out_strides[1] + seq1 * self.q_out_strides[2]
                k_out_offset = cta_b * self.k_out_strides[0] + global_h * self.k_out_strides[1] + seq1 * self.k_out_strides[2]
                
                q_out_ptr = (q_out_gmem.iterator + q_out_offset).align(16)
                k_out_ptr = (k_out_gmem.iterator + k_out_offset).align(16)
                
                g_out_layout_q = cute.make_layout((self.HEAD_DIM,), stride=(self.q_out_strides[3],))
                g_out_layout_k = cute.make_layout((self.HEAD_DIM,), stride=(self.k_out_strides[3],))
                g_out_q = cute.make_tensor(q_out_ptr, g_out_layout_q)
                g_out_k = cute.make_tensor(k_out_ptr, g_out_layout_k)
                
                cute.autovec_copy(q_out, g_out_q)
                cute.autovec_copy(k_out, g_out_k)


class FusedEpilogueInverseBackwardCute:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, NUM_HEADS: int = 16, HEAD_DIM: int = 64, HEAD_GROUP: int = 1, ab_stage: int = 1, q_strides=None, k_strides=None, sin_strides=None, cos_strides=None):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.HEAD_DIM = HEAD_DIM
        self.HEAD_GROUP = HEAD_GROUP
        self.NUM_HEADS = NUM_HEADS
        self.ab_stage = 1
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.sin_strides = sin_strides
        self.cos_strides = cos_strides
        
        s_rmsnorm_reduction_elems = self.BLOCK_M * 2 * self.HEAD_GROUP * self.HEAD_DIM
        self.s_scales_offset = ((s_rmsnorm_reduction_elems + 31) // 32) * 32
        
        total_elems = self.s_scales_offset + self.BLOCK_M * self.HEAD_GROUP
        self.flat_smem_elements = ((total_elems + 31) // 32) * 32
        self.max_smem_bytes = self.flat_smem_elements * 4

    def _setup(self):
        self.g_q_layout = cute.make_layout(
            (self.BLOCK_M, self.HEAD_DIM),
            stride=(self.q_strides[2], self.q_strides[3])
        )
        self.g_k_layout = cute.make_layout(
            (self.BLOCK_M, self.HEAD_DIM),
            stride=(self.k_strides[2], self.k_strides[3])
        )
        
        @cute.struct
        class SharedStorageEpilogueBack:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, self.flat_smem_elements], 1024]
        self.shared_storage = SharedStorageEpilogueBack

    @cute.jit
    def __call__(
        self,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        grad_q_out: cute.Tensor,
        grad_k_out: cute.Tensor,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_rmsnorm_weight: cute.Tensor,
        grad_scales: cute.Tensor,
        rms_q: cute.Tensor,
        rms_k: cute.Tensor,
        stream
    ):
        self._setup()
        
        B = q_out.shape[0]
        N = q_out.shape[2]
        
        num_threads = self.num_warps * 32
        grid_x_blocks = cute.ceil_div(N, self.BLOCK_M)
        
        self.kernel(
            q_out, k_out, rmsnorm_weight, scales, sin, cos,
            grad_q_out, grad_k_out, grad_q, grad_k,
            grad_rmsnorm_weight, grad_scales,
            rms_q, rms_k,
            self.g_q_layout, self.g_k_layout
        ).launch(
            grid=[grid_x_blocks, B, self.NUM_HEADS // self.HEAD_GROUP],
            block=[num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        scales: cute.Tensor,
        sin: cute.Tensor,
        cos: cute.Tensor,
        grad_q_out: cute.Tensor,
        grad_k_out: cute.Tensor,
        grad_q: cute.Tensor,
        grad_k: cute.Tensor,
        grad_rmsnorm_weight: cute.Tensor,
        grad_scales: cute.Tensor,
        rms_q: cute.Tensor,
        rms_k: cute.Tensor,
        g_q_layout: cute.Layout,
        g_k_layout: cute.Layout
    ):
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        block_idx_z = cute.arch.block_idx()[2]
        
        cta_m = block_idx_x
        cta_b = block_idx_y
        cta_h_group = block_idx_z
        
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = self.num_warps * 32
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        s_reduction = cute.make_tensor(
            cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=Float32),
            cute.make_layout(
                (self.BLOCK_M, 2, self.HEAD_GROUP, self.HEAD_DIM),
                stride=(2 * self.HEAD_GROUP * self.HEAD_DIM, self.HEAD_GROUP * self.HEAD_DIM, self.HEAD_DIM, 1)
            )
        )
        s_scales_reduction = cute.make_tensor(
            cute.recast_ptr((storage.sX_raw.data_ptr() + self.s_scales_offset).align(16), dtype=Float32),
            cute.make_layout((self.BLOCK_M, self.HEAD_GROUP), stride=(self.HEAD_GROUP, 1))
        )
        
        s_flat = cute.make_tensor(
            cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=Float32),
            cute.make_layout((self.flat_smem_elements,))
        )
        
        for idx in cutlass.range(tidx, self.flat_smem_elements, num_threads):
            s_flat[idx] = Float32(0.0)
            
        cute.arch.sync_threads()
        
        r = tidx % self.BLOCK_M
        h_local = tidx // self.BLOCK_M
        global_h = self.HEAD_GROUP * cta_h_group + h_local
        
        scale_val = Float32(0.0)
        q_scale_factor = Float32(1.0)
        w_q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        w_k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), rmsnorm_weight.element_type)
        
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
                w_q_reg[j] = rmsnorm_weight.element_type(1.0)
                w_k_reg[j] = rmsnorm_weight.element_type(1.0)
                
        N = q_out.shape[2]
        
        q_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        k_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dq_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dk_out_reg = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        sin_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), sin.element_type)
        cos_reg = cute.make_rmem_tensor((self.HEAD_DIM // 2,), cos.element_type)
        
        q_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        k_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dq_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dk_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dq_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        dk_normed = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        q_roped = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
        
        dq_in = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)
        dk_in = cute.make_rmem_tensor((self.HEAD_DIM,), BFloat16)

        seq_start = cta_m * self.BLOCK_M
        seq = seq_start + r
        
        if seq < N and h_local < self.HEAD_GROUP and global_h < self.NUM_HEADS:
            q_offset = cta_b * self.q_strides[0] + global_h * self.q_strides[1] + seq * self.q_strides[2]
            k_offset = cta_b * self.k_strides[0] + global_h * self.k_strides[1] + seq * self.k_strides[2]
            
            q_out_ptr = (q_out.iterator + q_offset).align(16)
            k_out_ptr = (k_out.iterator + k_offset).align(16)
            dq_out_ptr = (grad_q_out.iterator + q_offset).align(16)
            dk_out_ptr = (grad_k_out.iterator + k_offset).align(16)
            
            g_q_out = cute.make_tensor(q_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_k_out = cute.make_tensor(k_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            g_dq_out = cute.make_tensor(dq_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_dk_out = cute.make_tensor(dk_out_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            
            cute.autovec_copy(g_q_out, q_out_reg)
            cute.autovec_copy(g_k_out, k_out_reg)
            cute.autovec_copy(g_dq_out, dq_out_reg)
            cute.autovec_copy(g_dk_out, dk_out_reg)
            
            sin_offset = cta_b * self.sin_strides[0] + seq * self.sin_strides[2]
            cos_offset = cta_b * self.cos_strides[0] + seq * self.cos_strides[2]
            
            sin_ptr = (sin.iterator + sin_offset).align(16)
            cos_ptr = (cos.iterator + cos_offset).align(16)
            
            g_sin = cute.make_tensor(sin_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.sin_strides[3],)))
            g_cos = cute.make_tensor(cos_ptr, cute.make_layout((self.HEAD_DIM // 2,), stride=(self.cos_strides[3],)))
            
            cute.autovec_copy(g_sin, sin_reg)
            cute.autovec_copy(g_cos, cos_reg)
            
            # Load rms_q and rms_k values from global memory
            q_rsqrt = Float32(rms_q[cta_b, global_h, seq, 0])
            k_rsqrt = Float32(rms_k[cta_b, global_h, seq, 0])
            
            # Reconstruct q_normed and k_normed
            half_dim = self.HEAD_DIM // 2
            for j in cutlass.range(half_dim, unroll_full=True):
                cos_val = Float32(cos_reg[j])
                sin_val = Float32(sin_reg[j])
                
                # Unscale Q
                q_f1_unscaled = Float32(q_out_reg[j]) / q_scale_factor
                q_f2_unscaled = Float32(q_out_reg[j + half_dim]) / q_scale_factor
                
                # Inverse RoPE for Q
                qn1 = q_f1_unscaled * cos_val + q_f2_unscaled * sin_val
                qn2 = q_f2_unscaled * cos_val - q_f1_unscaled * sin_val
                
                # Inverse RoPE for K
                kn1 = Float32(k_out_reg[j]) * cos_val + Float32(k_out_reg[j + half_dim]) * sin_val
                kn2 = Float32(k_out_reg[j + half_dim]) * cos_val - Float32(k_out_reg[j]) * sin_val
                
                # Compute q_normed and k_normed
                q_normed[j] = qn1 / Float32(w_q_reg[j])
                q_normed[j + half_dim] = qn2 / Float32(w_q_reg[j + half_dim])
                
                k_normed[j] = kn1 / Float32(w_k_reg[j])
                k_normed[j + half_dim] = kn2 / Float32(w_k_reg[j + half_dim])
                
            # Compute gradients
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                dq_roped[j] = Float32(dq_out_reg[j]) * q_scale_factor
                dk_roped[j] = Float32(dk_out_reg[j])
                
            for j in cutlass.range(half_dim, unroll_full=True):
                cos_val = Float32(cos_reg[j])
                sin_val = Float32(sin_reg[j])
                
                dq_normed[j] = dq_roped[j] * cos_val + dq_roped[j + half_dim] * sin_val
                dq_normed[j + half_dim] = dq_roped[j + half_dim] * cos_val - dq_roped[j] * sin_val
                
                dk_normed[j] = dk_roped[j] * cos_val + dk_roped[j + half_dim] * sin_val
                dk_normed[j + half_dim] = dk_roped[j + half_dim] * cos_val - dk_roped[j] * sin_val
                
            for j in cutlass.range(half_dim, unroll_full=True):
                cos_val = Float32(cos_reg[j])
                sin_val = Float32(sin_reg[j])
                
                qn1 = q_normed[j] * Float32(w_q_reg[j])
                qn2 = q_normed[j + half_dim] * Float32(w_q_reg[j + half_dim])
                q_roped[j] = qn1 * cos_val - qn2 * sin_val
                q_roped[j + half_dim] = qn2 * cos_val + qn1 * sin_val
                
            grad_scale_local = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                grad_scale_local += dq_roped[j] * q_roped[j]
                
            s_scales_reduction[r, h_local] = grad_scale_local
            
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                s_reduction[r, 0, h_local, j] = dq_normed[j] * q_normed[j]
                s_reduction[r, 1, h_local, j] = dk_normed[j] * k_normed[j]
                
            sum_dq_q = Float32(0.0)
            sum_dk_k = Float32(0.0)
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                sum_dq_q += dq_normed[j] * Float32(w_q_reg[j]) * q_normed[j]
                sum_dk_k += dk_normed[j] * Float32(w_k_reg[j]) * k_normed[j]
                
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                dq_in_val = q_rsqrt * (dq_normed[j] * Float32(w_q_reg[j]) - q_normed[j] * (sum_dq_q / Float32(self.HEAD_DIM)))
                dk_in_val = k_rsqrt * (dk_normed[j] * Float32(w_k_reg[j]) - k_normed[j] * (sum_dk_k / Float32(self.HEAD_DIM)))
                dq_in[j] = BFloat16(dq_in_val)
                dk_in[j] = BFloat16(dk_in_val)
                
            dq_in_ptr = (grad_q.iterator + q_offset).align(16)
            dk_in_ptr = (grad_k.iterator + k_offset).align(16)
            
            g_out_dq = cute.make_tensor(dq_in_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.q_strides[3],)))
            g_out_dk = cute.make_tensor(dk_in_ptr, cute.make_layout((self.HEAD_DIM,), stride=(self.k_strides[3],)))
            
            cute.autovec_copy(dq_in, g_out_dq)
            cute.autovec_copy(dk_in, g_out_dk)
                
        cute.arch.sync_threads()
        
        step = self.BLOCK_M // 2
        while step > 0:
            total_work = step * self.HEAD_GROUP
            for work_idx in cutlass.range(tidx, total_work, num_threads):
                r_idx = work_idx % step
                hl = work_idx // step
                s_scales_reduction[r_idx, hl] += s_scales_reduction[r_idx + step, hl]
            
            cute.arch.sync_threads()
            step = step // 2

        for hl in cutlass.range(tidx, self.HEAD_GROUP, num_threads):
            curr_global_h = self.HEAD_GROUP * cta_h_group + hl
            if curr_global_h < self.NUM_HEADS:
                sum_scales = s_scales_reduction[0, hl]
                cute.arch.atomic_add(grad_scales[curr_global_h, 0, None].iterator, sum_scales)
                
        reduce_groups = 2 * self.HEAD_GROUP * (self.HEAD_DIM // 4)
        num_iters = (reduce_groups + num_threads - 1) // num_threads
        for iter_idx in cutlass.range_constexpr(num_iters):
            idx = tidx + iter_idx * num_threads
            if idx < reduce_groups:
                reg_sum = cute.make_rmem_tensor((4,), Float32)
                reg_val = cute.make_rmem_tensor((4,), Float32)
                
                weight_idx = idx // (self.HEAD_GROUP * (self.HEAD_DIM // 4))
                rem = idx % (self.HEAD_GROUP * (self.HEAD_DIM // 4))
                hl = rem // (self.HEAD_DIM // 4)
                j_group = rem % (self.HEAD_DIM // 4)
                j_start = j_group * 4
                
                reg_sum[0] = Float32(0.0)
                reg_sum[1] = Float32(0.0)
                reg_sum[2] = Float32(0.0)
                reg_sum[3] = Float32(0.0)
                
                for r_idx in cutlass.range(self.BLOCK_M, unroll_full=True):
                    s_offset = (
                        r_idx * (2 * self.HEAD_GROUP * self.HEAD_DIM) +
                        weight_idx * (self.HEAD_GROUP * self.HEAD_DIM) +
                        hl * self.HEAD_DIM +
                        j_start
                    )
                    
                    s_slice_ptr = s_reduction.iterator + s_offset
                    s_slice = cute.make_tensor(s_slice_ptr.align(16), cute.make_layout((4,)))
                    
                    cute.autovec_copy(s_slice, reg_val)
                    
                    reg_sum[0] += reg_val[0]
                    reg_sum[1] += reg_val[1]
                    reg_sum[2] += reg_val[2]
                    reg_sum[3] += reg_val[3]
                        
                curr_global_h = self.HEAD_GROUP * cta_h_group + hl
                if curr_global_h < self.NUM_HEADS:
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 0,
                        reg_sum[0]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 1,
                        reg_sum[1]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 2,
                        reg_sum[2]
                    )
                    cute.arch.atomic_add(
                        grad_rmsnorm_weight[weight_idx, 0, None].iterator + j_start + 3,
                        reg_sum[3]
                    )


_epilogue_inverse_compile_cache = {}

def get_fused_epilogue_inverse_kernel(
    num_warps=4, BLOCK_M=64, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=1, ab_stage=3,
    q_in_strides=None, k_in_strides=None, q_out_strides=None, k_out_strides=None, sin_strides=None, cos_strides=None,
    maxrregcount=255, B=32, N=512, sin_shape=None, cos_shape=None,
    q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, scales_dtype=torch.bfloat16, sin_dtype=torch.float32, cos_dtype=torch.float32
):
    B_sym = B
    N_sym = N
    if q_in_strides is None:
        q_in_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_in_strides = tuple(q_in_strides)
    if k_in_strides is None:
        k_in_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_in_strides = tuple(k_in_strides)
    if q_out_strides is None:
        q_out_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_out_strides = tuple(q_out_strides)
    if k_out_strides is None:
        k_out_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_out_strides = tuple(k_out_strides)
    if sin_strides is None:
        sin_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        sin_strides = tuple(sin_strides)
    if cos_strides is None:
        cos_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        cos_strides = tuple(cos_strides)
        
    if sin_shape is None:
        sin_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        sin_shape = tuple(sin_shape)
    if cos_shape is None:
        cos_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        cos_shape = tuple(cos_shape)
        
    if sin_shape[0] == 1:
        sin_strides = (0, sin_strides[1], sin_strides[2], sin_strides[3])
    if cos_shape[0] == 1:
        cos_strides = (0, cos_strides[1], cos_strides[2], cos_strides[3])
        
    key = (
        B, N, num_warps, BLOCK_M, NUM_HEADS, HEAD_DIM, HEAD_GROUP, ab_stage,
        q_in_strides, k_in_strides, q_out_strides, k_out_strides, sin_strides, cos_strides, maxrregcount, sin_shape, cos_shape,
        q_dtype, k_dtype, weight_dtype, scales_dtype, sin_dtype, cos_dtype
    )
    if key not in _epilogue_inverse_compile_cache:
        fwd_op = FusedEpilogueInverseCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_in_strides=q_in_strides,
            k_in_strides=k_in_strides,
            q_out_strides=q_out_strides,
            k_out_strides=k_out_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides
        )
        
        q_cute_dtype = _torch_to_cute_dtype(q_dtype)
        k_cute_dtype = _torch_to_cute_dtype(k_dtype)
        weight_cute_dtype = _torch_to_cute_dtype(weight_dtype)
        scales_cute_dtype = _torch_to_cute_dtype(scales_dtype)
        sin_cute_dtype = _torch_to_cute_dtype(sin_dtype)
        cos_cute_dtype = _torch_to_cute_dtype(cos_dtype)
        
        q_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_in_strides, assumed_align=16)
        k_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_in_strides, assumed_align=16)
        q_out_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_out_strides, assumed_align=16)
        k_out_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_out_strides, assumed_align=16)
        rmsnorm_weight_fake = make_fake_tensor(weight_cute_dtype, (2, 1, HEAD_DIM), 16)
        scales_fake = make_fake_tensor(scales_cute_dtype, (NUM_HEADS, 1, 1), 1)
        sin_fake = cute.runtime.make_fake_tensor(sin_cute_dtype, sin_shape, stride=sin_strides, assumed_align=16)
        cos_fake = cute.runtime.make_fake_tensor(cos_cute_dtype, cos_shape, stride=cos_strides, assumed_align=16)
        rms_q_fake = cute.runtime.make_fake_tensor(Float32, (B_sym, NUM_HEADS, N_sym, 1), stride=(NUM_HEADS * N_sym, N_sym, 1, 1), assumed_align=4)
        rms_k_fake = cute.runtime.make_fake_tensor(Float32, (B_sym, NUM_HEADS, N_sym, 1), stride=(NUM_HEADS * N_sym, N_sym, 1, 1), assumed_align=4)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            q_fake, k_fake, q_out_fake, k_out_fake, rmsnorm_weight_fake, scales_fake, sin_fake, cos_fake,
            rms_q_fake, rms_k_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _epilogue_inverse_compile_cache[key] = compiled_fwd
    return _epilogue_inverse_compile_cache[key]


_epilogue_inverse_back_compile_cache = {}

def get_fused_epilogue_inverse_back_kernel(
    num_warps=4, BLOCK_M=64, NUM_HEADS=16, HEAD_DIM=64, HEAD_GROUP=1, ab_stage=1,
    q_strides=None, k_strides=None, sin_strides=None, cos_strides=None,
    maxrregcount=255, B=32, N=512, sin_shape=None, cos_shape=None,
    q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, scales_dtype=torch.bfloat16, sin_dtype=torch.float32, cos_dtype=torch.float32
):
    B_sym = B
    N_sym = N
    if q_strides is None:
        q_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        q_strides = tuple(q_strides)
    if k_strides is None:
        k_strides = (NUM_HEADS * N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        k_strides = tuple(k_strides)
    if sin_strides is None:
        sin_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        sin_strides = tuple(sin_strides)
    if cos_strides is None:
        cos_strides = (N_sym * HEAD_DIM, N_sym * HEAD_DIM, HEAD_DIM, 1)
    else:
        cos_strides = tuple(cos_strides)
        
    if sin_shape is None:
        sin_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        sin_shape = tuple(sin_shape)
    if cos_shape is None:
        cos_shape = (B_sym, 1, N_sym, HEAD_DIM // 2)
    else:
        cos_shape = tuple(cos_shape)
        
    if sin_shape[0] == 1:
        sin_strides = (0, sin_strides[1], sin_strides[2], sin_strides[3])
    if cos_shape[0] == 1:
        cos_strides = (0, cos_strides[1], cos_strides[2], cos_strides[3])
        
    key = (
        B, N, num_warps, BLOCK_M, NUM_HEADS, HEAD_DIM, HEAD_GROUP, ab_stage,
        q_strides, k_strides, sin_strides, cos_strides, maxrregcount, sin_shape, cos_shape,
        q_dtype, k_dtype, weight_dtype, scales_dtype, sin_dtype, cos_dtype
    )
    if key not in _epilogue_inverse_back_compile_cache:
        back_op = FusedEpilogueInverseBackwardCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=NUM_HEADS,
            HEAD_DIM=HEAD_DIM,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_strides=q_strides,
            k_strides=k_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides
        )
        
        q_cute_dtype = _torch_to_cute_dtype(q_dtype)
        k_cute_dtype = _torch_to_cute_dtype(k_dtype)
        weight_cute_dtype = _torch_to_cute_dtype(weight_dtype)
        scales_cute_dtype = _torch_to_cute_dtype(scales_dtype)
        sin_cute_dtype = _torch_to_cute_dtype(sin_dtype)
        cos_cute_dtype = _torch_to_cute_dtype(cos_dtype)
        
        q_out_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        k_out_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        rmsnorm_weight_fake = make_fake_tensor(weight_cute_dtype, (2, 1, HEAD_DIM), 16)
        scales_fake = make_fake_tensor(scales_cute_dtype, (NUM_HEADS, 1, 1), 1)
        sin_fake = cute.runtime.make_fake_tensor(sin_cute_dtype, sin_shape, stride=sin_strides, assumed_align=16)
        cos_fake = cute.runtime.make_fake_tensor(cos_cute_dtype, cos_shape, stride=cos_strides, assumed_align=16)
        grad_q_out_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        grad_k_out_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        grad_q_fake = cute.runtime.make_fake_tensor(q_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=q_strides, assumed_align=16)
        grad_k_fake = cute.runtime.make_fake_tensor(k_cute_dtype, (B_sym, NUM_HEADS, N_sym, HEAD_DIM), stride=k_strides, assumed_align=16)
        grad_rmsnorm_weight_fake = make_fake_tensor(Float32, (2, 1, HEAD_DIM), 16)
        grad_scales_fake = make_fake_tensor(Float32, (NUM_HEADS, 1, 1), 1)
        rms_q_fake = cute.runtime.make_fake_tensor(Float32, (B_sym, NUM_HEADS, N_sym, 1), stride=(NUM_HEADS * N_sym, N_sym, 1, 1), assumed_align=4)
        rms_k_fake = cute.runtime.make_fake_tensor(Float32, (B_sym, NUM_HEADS, N_sym, 1), stride=(NUM_HEADS * N_sym, N_sym, 1, 1), assumed_align=4)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_back = cute.compile(
            back_op,
            q_out_fake, k_out_fake, rmsnorm_weight_fake, scales_fake, sin_fake, cos_fake,
            grad_q_out_fake, grad_k_out_fake, grad_q_fake, grad_k_fake,
            grad_rmsnorm_weight_fake, grad_scales_fake,
            rms_q_fake, rms_k_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount={maxrregcount}'",
        )
        _epilogue_inverse_back_compile_cache[key] = compiled_back
    return _epilogue_inverse_back_compile_cache[key]


_epilogue_inverse_fwd_kernel_cache = {}

@torch.library.custom_op("flash_prologue::qknorm_rope_scale_cute_inverse", mutates_args=())
def qknorm_rope_scale_cute_inverse_op(
    q: torch.Tensor,
    k: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sin = _ensure_4d(sin)
    cos = _ensure_4d(cos)
    B = int(q.shape[0])
    num_heads = int(q.shape[1])
    N = int(q.shape[2])
    head_dim = int(q.shape[3])
    
    q_out_strides = _get_compact_strides(q.shape, q.stride())
    k_out_strides = _get_compact_strides(k.shape, k.stride())
    
    q_in_strides = tuple(int(s) for s in q.stride())
    k_in_strides = tuple(int(s) for s in k.stride())
    q_out_strides_t = tuple(int(s) for s in q_out_strides)
    k_out_strides_t = tuple(int(s) for s in k_out_strides)
    
    sin_c = sin if sin.is_contiguous() else sin.contiguous()
    cos_c = cos if cos.is_contiguous() else cos.contiguous()
    sin_strides = tuple(int(s) for s in sin_c.stride())
    cos_strides = tuple(int(s) for s in cos_c.stride())
    
    cache_key = (
        B, N, num_heads, head_dim,
        q_in_strides, k_in_strides, q_out_strides_t, k_out_strides_t,
        sin_strides, cos_strides,
        tuple(sin_c.shape), tuple(cos_c.shape),
        q.dtype, k.dtype, rmsnorm_weight.dtype, scales.dtype, sin_c.dtype, cos_c.dtype
    )
    fwd_kernel = _epilogue_inverse_fwd_kernel_cache.get(cache_key)
    if fwd_kernel is None:
        config = get_autotuned_config(M=N, N=num_heads, K=head_dim, is_fused="epilogue_fwd")
        BLOCK_M = config["BLOCK_M"]
        HEAD_GROUP = config["HEAD_GROUP"]
        ab_stage = config["ab_stage"]
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        
        fwd_kernel = get_fused_epilogue_inverse_kernel(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_in_strides=q_in_strides,
            k_in_strides=k_in_strides,
            q_out_strides=q_out_strides_t,
            k_out_strides=k_out_strides_t,
            sin_strides=sin_strides,
            cos_strides=cos_strides,
            maxrregcount=maxrregcount,
            B=B,
            N=N,
            sin_shape=tuple(sin_c.shape),
            cos_shape=tuple(cos_c.shape),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            weight_dtype=rmsnorm_weight.dtype,
            scales_dtype=scales.dtype,
            sin_dtype=sin_c.dtype,
            cos_dtype=cos_c.dtype
        )
        _epilogue_inverse_fwd_kernel_cache[cache_key] = fwd_kernel
        
    rmsnorm_weight_c = rmsnorm_weight if rmsnorm_weight.is_contiguous() else rmsnorm_weight.contiguous()
    scales_c = scales if scales.is_contiguous() else scales.contiguous()
    
    q_out = torch.empty_strided(q.shape, q_out_strides, dtype=q.dtype, device=q.device)
    k_out = torch.empty_strided(k.shape, k_out_strides, dtype=k.dtype, device=k.device)
    
    rms_q_shape = (B, num_heads, N, 1)
    rms_q_strides = (num_heads * N, N, 1, 1)
    rms_q = torch.empty_strided(rms_q_shape, rms_q_strides, dtype=torch.float32, device=q.device)
    rms_k = torch.empty_strided(rms_q_shape, rms_q_strides, dtype=torch.float32, device=q.device)
    
    fwd_kernel(
        q, k, q_out, k_out, rmsnorm_weight_c, scales_c, sin_c, cos_c, rms_q, rms_k
    )
    # Preserve input lifetimes to prevent eager allocator from reusing memory while FFI kernel runs
    inputs_tuple = (q, k, rmsnorm_weight_c, scales_c, sin_c, cos_c)
    q_out._base_inputs = inputs_tuple
    k_out._base_inputs = inputs_tuple
    rms_q._base_inputs = inputs_tuple
    rms_k._base_inputs = inputs_tuple
    return q_out, k_out, rms_q, rms_k

@qknorm_rope_scale_cute_inverse_op.register_fake
def _qknorm_rope_scale_cute_inverse_fake(q, k, rmsnorm_weight, scales, sin, cos, eps=1e-5):
    q_out_strides = _get_compact_strides(q.shape, q.stride())
    k_out_strides = _get_compact_strides(k.shape, k.stride())
    rms_q_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
    rms_q_strides = (q.shape[1] * q.shape[2], q.shape[2], 1, 1)
    return (
        torch.empty_strided(q.shape, q_out_strides, dtype=q.dtype, device=q.device),
        torch.empty_strided(k.shape, k_out_strides, dtype=k.dtype, device=k.device),
        torch.empty_strided(rms_q_shape, rms_q_strides, dtype=torch.float32, device=q.device),
        torch.empty_strided(rms_q_shape, rms_q_strides, dtype=torch.float32, device=q.device)
    )


_epilogue_inverse_bwd_kernel_cache = {}

@torch.library.custom_op("flash_prologue::qknorm_rope_scale_cute_inverse_backward", mutates_args=())
def qknorm_rope_scale_cute_inverse_backward_op(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    scales: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    rms_q: torch.Tensor,
    rms_k: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sin = _ensure_4d(sin)
    cos = _ensure_4d(cos)
    B = int(q_out.shape[0])
    num_heads = int(q_out.shape[1])
    N = int(q_out.shape[2])
    head_dim = int(q_out.shape[3])
    
    q_strides_compact = _get_compact_strides(q_out.shape, q_out.stride())
    k_strides_compact = _get_compact_strides(k_out.shape, k_out.stride())
    
    q_matched = q_out if q_out.stride() == q_strides_compact else torch.empty_strided(q_out.shape, q_strides_compact, dtype=q_out.dtype, device=q_out.device).copy_(q_out)
    k_matched = k_out if k_out.stride() == k_strides_compact else torch.empty_strided(k_out.shape, k_strides_compact, dtype=k_out.dtype, device=k_out.device).copy_(k_out)
    
    grad_q_out_matched = grad_q_out if grad_q_out.stride() == q_strides_compact else torch.empty_strided(q_out.shape, q_strides_compact, dtype=grad_q_out.dtype, device=grad_q_out.device).copy_(grad_q_out)
    grad_k_out_matched = grad_k_out if grad_k_out.stride() == k_strides_compact else torch.empty_strided(k_out.shape, k_strides_compact, dtype=grad_k_out.dtype, device=grad_k_out.device).copy_(grad_k_out)

    # Zero out potential garbage in the padded region of grad_q_out/grad_k_out
    active_mask = (cos.abs().sum(dim=(0, 1, 3)) > 1e-6).to(dtype=grad_q_out_matched.dtype, device=grad_q_out_matched.device)
    active_mask_4d = active_mask.view(1, 1, -1, 1)
    grad_q_out_matched = grad_q_out_matched * active_mask_4d
    grad_k_out_matched = grad_k_out_matched * active_mask_4d
    
    q_strides = tuple(int(s) for s in q_strides_compact)
    k_strides = tuple(int(s) for s in k_strides_compact)
    
    sin_c = sin if sin.is_contiguous() else sin.contiguous()
    cos_c = cos if cos.is_contiguous() else cos.contiguous()
    sin_strides = tuple(int(s) for s in sin_c.stride())
    cos_strides = tuple(int(s) for s in cos_c.stride())
    
    cache_key = (
        B, N, num_heads, head_dim,
        q_strides, k_strides, sin_strides, cos_strides,
        tuple(sin_c.shape), tuple(cos_c.shape),
        q_out.dtype, k_out.dtype, rmsnorm_weight.dtype, scales.dtype, sin_c.dtype, cos_c.dtype
    )
    back_kernel = _epilogue_inverse_bwd_kernel_cache.get(cache_key)
    if back_kernel is None:
        config = get_autotuned_config(M=N, N=num_heads, K=head_dim, is_fused="epilogue_bwd")
        BLOCK_M = config["BLOCK_M"]
        HEAD_GROUP = config["HEAD_GROUP"]
        ab_stage = config["ab_stage"]
        num_warps = config.get("num_warps", 4)
        maxrregcount = config.get("maxrregcount", 255)
        
        back_kernel = get_fused_epilogue_inverse_back_kernel(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HEAD_GROUP=HEAD_GROUP,
            ab_stage=ab_stage,
            q_strides=q_strides,
            k_strides=k_strides,
            sin_strides=sin_strides,
            cos_strides=cos_strides,
            maxrregcount=maxrregcount,
            B=B,
            N=N,
            sin_shape=tuple(sin_c.shape),
            cos_shape=tuple(cos_c.shape),
            q_dtype=q_out.dtype,
            k_dtype=k_out.dtype,
            weight_dtype=rmsnorm_weight.dtype,
            scales_dtype=scales.dtype,
            sin_dtype=sin_c.dtype,
            cos_dtype=cos_c.dtype
        )
        _epilogue_inverse_bwd_kernel_cache[cache_key] = back_kernel
        
    grad_q = torch.empty_strided(q_out.shape, q_strides_compact, dtype=q_out.dtype, device=q_out.device)
    grad_k = torch.empty_strided(k_out.shape, k_strides_compact, dtype=k_out.dtype, device=k_out.device)
    grad_rmsnorm_weight = torch.zeros((2, 1, head_dim), device=q_out.device, dtype=torch.float32)
    grad_scales = torch.zeros((num_heads, 1, 1), device=q_out.device, dtype=torch.float32)
    
    rmsnorm_weight_c = rmsnorm_weight if rmsnorm_weight.is_contiguous() else rmsnorm_weight.contiguous()
    scales_c = scales if scales.is_contiguous() else scales.contiguous()
    rms_q_c = rms_q if rms_q.is_contiguous() else rms_q.contiguous()
    rms_k_c = rms_k if rms_k.is_contiguous() else rms_k.contiguous()
    
    back_kernel(
        q_matched, k_matched, rmsnorm_weight_c, scales_c, sin_c, cos_c,
        grad_q_out_matched, grad_k_out_matched, grad_q, grad_k,
        grad_rmsnorm_weight, grad_scales,
        rms_q_c, rms_k_c
    )
    
    # Preserve input lifetimes to prevent eager allocator from reusing memory while FFI kernel runs
    inputs_tuple = (q_matched, k_matched, rmsnorm_weight_c, scales_c, sin_c, cos_c, grad_q_out_matched, grad_k_out_matched, rms_q_c, rms_k_c)
    grad_q._base_inputs = inputs_tuple
    grad_k._base_inputs = inputs_tuple
    grad_rmsnorm_weight._base_inputs = inputs_tuple
    grad_scales._base_inputs = inputs_tuple
    return grad_q, grad_k, grad_rmsnorm_weight, grad_scales

@qknorm_rope_scale_cute_inverse_backward_op.register_fake
def _qknorm_rope_scale_cute_inverse_backward_fake(q_out, k_out, rmsnorm_weight, scales, sin, cos, grad_q_out, grad_k_out, rms_q, rms_k, eps=1e-5):
    q_strides_compact = _get_compact_strides(q_out.shape, q_out.stride())
    k_strides_compact = _get_compact_strides(k_out.shape, k_out.stride())
    grad_q = torch.empty_strided(q_out.shape, q_strides_compact, dtype=q_out.dtype, device=q_out.device)
    grad_k = torch.empty_strided(k_out.shape, k_strides_compact, dtype=k_out.dtype, device=k_out.device)
    grad_rmsnorm_weight = torch.empty((2, 1, q_out.shape[-1]), device=q_out.device, dtype=torch.float32)
    grad_scales = torch.empty((q_out.shape[1], 1, 1), device=q_out.device, dtype=torch.float32)
    return grad_q, grad_k, grad_rmsnorm_weight, grad_scales


def setup_qknorm_rope_scale_cute_inverse_context(ctx, inputs, output):
    q, k, rmsnorm_weight, scales, sin, cos, eps = inputs
    q_out, k_out, rms_q, rms_k = output
    ctx.save_for_backward(q_out, k_out, rmsnorm_weight, scales, sin, cos, rms_q, rms_k)
    ctx.eps = eps

def qknorm_rope_scale_cute_inverse_backward_hook(ctx, grad_q_out, grad_k_out, grad_rms_q, grad_rms_k):
    q_out, k_out, rmsnorm_weight, scales, sin, cos, rms_q, rms_k = ctx.saved_tensors
    eps = ctx.eps
    
    dq, dk, dw, ds = qknorm_rope_scale_cute_inverse_backward_op(
        q_out, k_out, rmsnorm_weight, scales, sin, cos, grad_q_out, grad_k_out, rms_q, rms_k, eps
    )
    return dq, dk, dw, ds, None, None, None

torch.library.register_autograd(
    "flash_prologue::qknorm_rope_scale_cute_inverse",
    qknorm_rope_scale_cute_inverse_backward_hook,
    setup_context=setup_qknorm_rope_scale_cute_inverse_context
)

