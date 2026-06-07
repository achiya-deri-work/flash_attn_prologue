import os
import torch
import math

# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int64, Boolean, const_expr
from quack.compile_utils import make_fake_tensor

class PipelinedGemmCute:
    def __init__(
        self,
        num_warps: int = 4,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 64,
        ab_stage: int = 2,
        atom_layout_mnk: tuple = (2, 2, 1)
    ):
        self.atom_layout_mnk = atom_layout_mnk
        self.num_warps = math.prod(self.atom_layout_mnk) if atom_layout_mnk is not None else num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.ab_stage = ab_stage

        
    def _setup(self):
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K, self.ab_stage),
            stride=(self.BLOCK_K, 1, self.BLOCK_M * self.BLOCK_K)
        )
        self.b_smem_layout_staged = cute.make_layout(
            (self.BLOCK_N, self.BLOCK_K, self.ab_stage),
            stride=(self.BLOCK_K, 1, self.BLOCK_N * self.BLOCK_K)
        )
        
        # Shared memory for epilogue writeback (aliased)
        self.epi_smem_layout = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_N),
            stride=(self.BLOCK_N, 1)
        )
        
        # 2. Shared storage (double-buffered)
        @cute.struct
        class SharedStoragePipelined:
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.BLOCK_M * self.BLOCK_K], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, self.ab_stage * self.BLOCK_N * self.BLOCK_K], 1024]
        self.shared_storage = SharedStoragePipelined
        
        # 3. TiledMMA configuration
        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=self.atom_layout_mnk)

        # 4. TiledCopy configurations using cp.async (cooperative, tiled for all 128 threads)
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom = cute.make_copy_atom(copy_op, BFloat16, num_bits_per_copy=128)
        
        self.gmem_tiled_copy_x = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((32, 4), order=(1, 0)),
            cute.make_layout((1, 8))
        )
        self.gmem_tiled_copy_w = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((32, 4), order=(1, 0)),
            cute.make_layout((1, 8))
        )
        
        # Statically define write iterations
        self.num_write_iters = self.BLOCK_N // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qkv_out: cute.Tensor,
        stream
    ):
        self._setup()
        
        B = x.shape[0]
        N = x.shape[1]
        hidden = x.shape[2]
        total_m = B * N
        
        x_2d = cute.make_tensor(x.iterator, cute.make_layout((total_m, hidden), stride=(hidden, 1)))
        qkv_out_2d = cute.make_tensor(qkv_out.iterator, cute.make_layout((total_m, qkv_out.shape[2]), stride=(qkv_out.shape[2], 1)))
        
        num_threads = self.num_warps * 32
        
        self.kernel(
            x_2d, weight, qkv_out_2d,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout,
            self.gmem_tiled_copy_x,
            self.gmem_tiled_copy_w,
            self.num_write_iters
        ).launch(
            grid=[cute.ceil_div(total_m, self.BLOCK_M), cute.ceil_div(qkv_out_2d.shape[1], self.BLOCK_N), 1],
            block=[num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x_2d: cute.Tensor,
        weight_2d: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        gmem_tiled_copy_x: cute.TiledCopy,
        gmem_tiled_copy_w: cute.TiledCopy,
        num_write_iters: int
    ):
        block_idx_x = cute.arch.block_idx()[0]
        block_idx_y = cute.arch.block_idx()[1]
        
        grid_dim_m = cute.arch.grid_dim()[0]
        grid_dim_n = cute.arch.grid_dim()[1]
        
        pid = block_idx_y * grid_dim_m + block_idx_x
        group_span = 8
        num_pid_in_group = group_span * grid_dim_m
        group_id = pid // num_pid_in_group
        first_pid_n = group_id * group_span
        rem_n = grid_dim_n - first_pid_n
        this_group_span = group_span if group_span < rem_n else rem_n
        
        cta_n = first_pid_n + (pid % this_group_span)
        cta_m = (pid % num_pid_in_group) // this_group_span
        tidx, _, _ = cute.arch.thread_idx()
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        sX_tensor = storage.sX.get_tensor(a_smem_layout)
        sW_tensor = storage.sW.get_tensor(b_smem_layout)
        sX_aligned = cute.make_tensor(sX_tensor.iterator.align(16), sX_tensor.layout)
        sW_aligned = cute.make_tensor(sW_tensor.iterator.align(16), sW_tensor.layout)
        
        sC_epi = cute.make_tensor(storage.sX.data_ptr().align(16), epi_smem_layout)
        
        total_m = mC.shape[0]
        N_out = mC.shape[1]
        hidden = x_2d.shape[1]
        k_tile_cnt = hidden // self.BLOCK_K
        
        # Globally define thread slices to preserve type consistency (all 128 threads load)
        tidx_copy = tidx if tidx < 128 else 0
        thr_copy_x = gmem_tiled_copy_x.get_slice(tidx_copy)
        thr_copy_w = gmem_tiled_copy_w.get_slice(tidx_copy)
        tSsX = thr_copy_x.partition_D(sX_aligned)
        tSsW = thr_copy_w.partition_D(sW_aligned)

        
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Partition SMEM stages outside the loop
        tCsX = thr_mma.partition_A(sX_aligned)
        tCsW = thr_mma.partition_B(sW_aligned)
        
        # Pre-allocate double-buffered register fragments
        rA_0 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_0 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        rA_1 = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
        rB_1 = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
        
        tCsC = thr_mma.partition_C(sC_epi)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        gX_layout = cute.make_layout((self.BLOCK_M, self.BLOCK_K), stride=(hidden, 1))
        gW_layout = cute.make_layout((self.BLOCK_N, self.BLOCK_K), stride=(hidden, 1))
        
        # Prefetch first stage (k_tile = 0)
        if 0 < k_tile_cnt:
            x_ptr = (x_2d[cta_m * self.BLOCK_M, None].iterator + 0).align(16)
            gX = cute.make_tensor(x_ptr, gX_layout)
            tSgX = thr_copy_x.partition_S(gX)
            
            weight_ptr = (weight_2d[cta_n * self.BLOCK_N, None].iterator + 0).align(16)
            gW = cute.make_tensor(weight_ptr, gW_layout)
            tSgW = thr_copy_w.partition_S(gW)
            if tidx < 128:
                cute.copy(gmem_tiled_copy_x, tSgX, tSsX[None, None, None, 0])
                cute.copy(gmem_tiled_copy_w, tSgW, tSsW[None, None, None, 0])
            
            cute.arch.cp_async_commit_group()
            
        # Wait for the first stage (stage 0) to be ready in shared memory and prefetch it to rA_0/rB_0 (vectorized)
        if k_tile_cnt > 0:
            cute.arch.cp_async_wait_group(self.ab_stage - 2)
            cute.arch.sync_threads()
            
            tCsA_init = tCsX[None, None, None, 0]
            tCsB_init = tCsW[None, None, None, 0]
            cute.autovec_copy(tCsA_init, rA_0)
            cute.autovec_copy(tCsB_init, rB_0)
                
        # Main execution loop
        for k_tile in cutlass.range_constexpr(k_tile_cnt):
            pipe_read = k_tile % self.ab_stage
            pipe_write = (k_tile + self.ab_stage - 1) % self.ab_stage
            next_pipe_read = (k_tile + 1) % self.ab_stage
            
            next_gmem_read = k_tile + (self.ab_stage - 1)
            if next_gmem_read < k_tile_cnt:
                k_offset = next_gmem_read * self.BLOCK_K
                
                x_ptr = (x_2d[cta_m * self.BLOCK_M, None].iterator + k_offset).align(16)
                gX = cute.make_tensor(x_ptr, gX_layout)
                tSgX = thr_copy_x.partition_S(gX)
                
                weight_ptr = (weight_2d[cta_n * self.BLOCK_N, None].iterator + k_offset).align(16)
                gW = cute.make_tensor(weight_ptr, gW_layout)
                tSgW = thr_copy_w.partition_S(gW)
                if tidx < 128:
                    cute.copy(gmem_tiled_copy_x, tSgX, tSsX[None, None, None, pipe_write])
                    cute.copy(gmem_tiled_copy_w, tSgW, tSsW[None, None, None, pipe_write])
                
                cute.arch.cp_async_commit_group()
                
            # Prefetch next stage's registers
            if k_tile + 1 < k_tile_cnt:
                cute.arch.cp_async_wait_group(self.ab_stage - 2)
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
                
        # Write back accumulators to SMEM
        tCsC = thr_mma.partition_C(sC_epi)
        for i in range(cute.size(accum)):
            tCsC[i] = BFloat16(accum[i])
                
        cute.arch.sync_threads()
        
        # Vectorized epilogue writeback to Global Memory (threads < BLOCK_M)
        if tidx < self.BLOCK_M:
            row_idx = cta_m * self.BLOCK_M + tidx
            col_start = cta_n * self.BLOCK_N
            
            if row_idx < total_m:
                s_ptr = sC_epi.iterator + tidx * self.BLOCK_N
                g_ptr = mC.iterator + row_idx * N_out + col_start
                
                for j in cutlass.range(num_write_iters, unroll_full=True):
                    if col_start + j * 8 < N_out:
                        reg = cute.make_rmem_tensor((8,), BFloat16)
                        cute.autovec_copy(cute.make_tensor(s_ptr + j * 8, cute.make_layout((8,))), reg)
                        cute.autovec_copy(reg, cute.make_tensor(g_ptr + j * 8, cute.make_layout((8,))))


_pipelined_gemm_compile_cache = {}

def get_pipelined_gemm_fwd_kernel(
    num_warps=4,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    hidden=64,
    out_features=128,
    ab_stage=2,
    atom_layout_mnk=(2, 2, 1)
):
    key = (num_warps, BLOCK_M, BLOCK_N, BLOCK_K, hidden, out_features, ab_stage, atom_layout_mnk)
    if key not in _pipelined_gemm_compile_cache:
        B_sym = cute.sym_int()
        N_sym = cute.sym_int()
        
        fwd_op = PipelinedGemmCute(
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            ab_stage=ab_stage,
            atom_layout_mnk=atom_layout_mnk
        )
        
        x_fake = make_fake_tensor(BFloat16, (B_sym, N_sym, hidden), 16)
        weight_fake = make_fake_tensor(BFloat16, (out_features, hidden), 16)
        qkv_out_fake = make_fake_tensor(BFloat16, (B_sym, N_sym, out_features), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            x_fake, weight_fake, qkv_out_fake,
            fake_stream,
            options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v --maxrregcount=255'",
        )
        _pipelined_gemm_compile_cache[key] = compiled_fwd
    return _pipelined_gemm_compile_cache[key]

@torch.library.custom_op("flash_prologue::pipelined_gemm_cute", mutates_args=())
def pipelined_gemm_cute_op(
    x: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    B, N, hidden = x.shape
    out_features = weight.shape[0]
    
    x_bf16 = x.to(torch.bfloat16).contiguous()
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    
    qkv_out_bf16 = torch.empty((B, N, out_features), dtype=torch.bfloat16, device=x.device)
    
    # Query Autotuner for optimal shape configs
    from flash_attn_prologue.autotune import get_autotuned_config
    config = get_autotuned_config(B * N, out_features, hidden, is_fused=False, device=str(x.device))
    
    fwd_kernel = get_pipelined_gemm_fwd_kernel(
        num_warps=4,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        hidden=hidden,
        out_features=out_features
    )
    
    fwd_kernel(x_bf16, weight_bf16, qkv_out_bf16)
    
    return qkv_out_bf16

@pipelined_gemm_cute_op.register_fake
def _pipelined_gemm_cute_fake(x, weight):
    B, N, hidden = x.shape
    out_features = weight.shape[0]
    return torch.empty((B, N, out_features), dtype=x.dtype, device=x.device)
