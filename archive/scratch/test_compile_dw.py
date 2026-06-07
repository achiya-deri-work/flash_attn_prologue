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

class GemmBwdDwTest:
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
        # Column-major layouts for the main loop MMA
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_M, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_N, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        # Transposed row-major layouts for the TMA load destination in shared memory
        self.a_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_M, self.ab_stage),
            stride=(self.BLOCK_M, 1, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
        self.b_smem_layout_staged_row = cute.make_layout(
            (self.BLOCK_K_STAGE, self.BLOCK_N, self.ab_stage),
            stride=(self.BLOCK_N, 1, self.BLOCK_N * self.BLOCK_K_STAGE)
        )
        
        # Contiguous (no padding) epilogue layout to avoid CUDA misalignment errors
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
        
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=(2, 2, 1))
        
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
        
        # Define the row-major view of gA and gB (transposed shapes) for TMA load
        gA_row = cute.make_tensor(grad_output_t.iterator, cute.make_layout((self.K_dim, self.M_dim), stride=(self.M_dim, 1)))
        gB_row = cute.make_tensor(x.iterator, cute.make_layout((self.K_dim, self.N_dim), stride=(self.N_dim, 1)))
        gO = cute.make_tensor(grad_weight.iterator, cute.make_layout((self.M_dim, self.N_dim), stride=(self.N_dim, 1)))
        
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_row,
            cute.make_layout((self.BLOCK_K_STAGE, self.BLOCK_M), stride=(self.BLOCK_M, 1)),
            (self.BLOCK_K_STAGE, self.BLOCK_M)
        )
        
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gB_row,
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
        K_dim = self.K_dim
        N_dim = self.N_dim
        
        k_tile_cnt = self.k_tile_cnt
        k_tiles_per_slice = (k_tile_cnt + self.SPLIT_K - 1) // self.SPLIT_K
        k_tile_start = block_idx_z * k_tiles_per_slice
        
        if tidx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_o)
            
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        # Column-major views of shared memory for registers partition & MMA
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        sW_aligned = cute.make_tensor(cute.recast_ptr((storage.sX_raw.data_ptr() + self.sW_offset_elements).align(16), dtype=BFloat16), b_smem_layout)
        
        # Row-major views of shared memory for TMA destination partition
        sX_aligned_row = cute.make_tensor(sX_aligned.iterator, a_smem_layout_row)
        sW_aligned_row = cute.make_tensor(sW_aligned.iterator, b_smem_layout_row)
        
        thr_mma = tiled_mma.get_slice(tidx)
        
        smem_accum = cute.make_tensor(storage.sX_raw.data_ptr().align(16), epi_smem_layout)
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        # MMA reads column-major shared memory
        tCsX = thr_mma.partition_A(sX_aligned)
        tCsW = thr_mma.partition_B(sW_aligned)
        
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        # TMA partitions use the transposed row-major shapes
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
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(self.ab_stage):
                    cute.arch.mbarrier_init(mbar_ptr + s, 1)
                cute.arch.mbarrier_init_fence()
            
        cute.arch.sync_threads()
        
        # Prefetch initial stages
        for stage in cutlass.range_constexpr(self.ab_stage - 1):
            if stage < k_tiles_per_slice:
                k_curr = cutlass.min(k_tile_start + stage, k_tile_cnt - 1)
                if tidx == 0:
                    with cute.arch.elect_one():
                        tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
                if tidx < 32:
                    copy_A(k_curr, stage, tma_bar_ptr=mbar_ptr + stage)
                    copy_B(k_curr, stage, tma_bar_ptr=mbar_ptr + stage)
                    
        if k_tiles_per_slice > 0:
            cute.arch.mbarrier_wait(mbar_ptr + 0, 0)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
            
        for step in cutlass.range_constexpr(k_tiles_per_slice):
            pipe_read = step % self.ab_stage
            pipe_write = (step + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (step + 1) % self.ab_stage
            
            next_gmem_read = step + (self.ab_stage - 1)
            if next_gmem_read < k_tiles_per_slice:
                k_curr = cutlass.min(k_tile_start + next_gmem_read, k_tile_cnt - 1)
                if tidx == 0:
                    with cute.arch.elect_one():
                        tx_bytes = (self.BLOCK_M * self.BLOCK_K_STAGE * 2) + (self.BLOCK_N * self.BLOCK_K_STAGE * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + pipe_write, tx_bytes)
                if tidx < 32:
                    copy_A(k_curr, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    copy_B(k_curr, pipe_write, tma_bar_ptr=mbar_ptr + pipe_write)
                    
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
                    
            if k_tile_start + step < k_tile_cnt:
                if step % 2 == 0:
                    cute.gemm(tiled_mma, accum, rA_0, rB_0, accum)
                else:
                    cute.gemm(tiled_mma, accum, rA_1, rB_1, accum)
                    
        cute.copy(copy_atom_r2s, accum, tCsC)
        cute.arch.sync_threads()
        
        # TMA Epilogue store
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

def test():
    M_dim = 3072
    K_dim = 16384
    N_dim = 512
    
    op = GemmBwdDwTest(
        num_warps=4,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K_STAGE=32,
        ab_stage=3,
        M_dim=M_dim,
        K_dim=K_dim,
        N_dim=N_dim,
        SPLIT_K=1
    )
    
    print("Making fake tensors...")
    grad_output_t_fake = make_fake_tensor(BFloat16, (M_dim, K_dim), 16, leading_dim=0)
    x_fake = make_fake_tensor(BFloat16, (K_dim, N_dim), 16)
    grad_weight_fake = make_fake_tensor(Float32, (M_dim, N_dim), 16)
    
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    
    print("Compiling GemmBwdDwTest with opt-level 3 and ptxas -O3...")
    compiled_fn = cute.compile(
        op,
        grad_output_t_fake, x_fake, grad_weight_fake,
        fake_stream,
        options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount=255'",
    )
    print("Compilation successful!")
    
    print("Creating real tensors on GPU...")
    grad_output_t = torch.randn(K_dim, M_dim, device="cuda", dtype=torch.bfloat16).t()
    x = torch.randn(K_dim, N_dim, device="cuda", dtype=torch.bfloat16)
    grad_weight = torch.zeros(M_dim, N_dim, device="cuda", dtype=torch.float32)
    
    print("Executing compiled kernel...")
    compiled_fn(grad_output_t, x, grad_weight)
    torch.cuda.synchronize()
    print("Execution successful!")

if __name__ == "__main__":
    test()
