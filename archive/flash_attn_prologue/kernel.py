import os
import torch

# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32

class FusedQKVCuteFwd:
    def __init__(self, num_warps: int = 4, BLOCK_M: int = 64, BLOCK_K: int = 512, NUM_HEADS: int = 16, HEAD_DIM: int = 64):
        self.num_warps = num_warps
        self.BLOCK_M = BLOCK_M
        self.BLOCK_K = BLOCK_K
        self.NUM_HEADS = NUM_HEADS
        self.HEAD_DIM = HEAD_DIM
        self.BLOCK_K_TILE = 32
        
    def _setup(self):
        # 1. Setup shared memory layouts
        self.smem_x_layout = cute.make_layout((self.BLOCK_M, self.BLOCK_K_TILE), stride=(self.BLOCK_K_TILE, 1))
        self.smem_w_layout = cute.make_layout((3 * self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K_TILE, 1))
        self.smem_accum_layout = cute.make_layout((self.BLOCK_M, 3 * self.HEAD_DIM), stride=(3 * self.HEAD_DIM, 1))
        
        # 2. Shared storage definition (Fits in 64 KB, fully compatible with dynamic SMEM on Blackwell)
        @cute.struct
        class SharedStorageFwd:
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, self.BLOCK_M * self.BLOCK_K_TILE], 128]
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, 3 * self.HEAD_DIM * self.BLOCK_K_TILE], 128]
            sBuf: cute.struct.Align[cute.struct.MemRange[Float32, self.BLOCK_M * 3 * self.HEAD_DIM], 128]
        self.shared_storage = SharedStorageFwd
        
        # 3. TiledCopy configurations using cp.async
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
        
        # 4. TiledMMA configuration
        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        self.mma = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        self.tiled_mma = cute.make_tiled_mma(self.mma, atom_layout_mnk=(2, 2, 1))

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
            self.smem_x_layout, self.smem_w_layout, self.smem_accum_layout,
            self.tiled_mma, self.gmem_tiled_copy_x, self.gmem_tiled_copy_w
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
        smem_w_layout: cute.Layout,
        smem_accum_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_x: cute.TiledCopy,
        gmem_tiled_copy_w: cute.TiledCopy
    ):
        cta_m, cta_b, cta_h = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        smem_x = storage.sX.get_tensor(smem_x_layout)
        smem_w = storage.sW.get_tensor(smem_w_layout)
        smem_accum = storage.sBuf.get_tensor(smem_accum_layout)
        
        seq_start = cta_m * self.BLOCK_M
        N = x.shape[1]
        
        thr_copy_x = gmem_tiled_copy_x.get_slice(tid)
        thr_copy_w = gmem_tiled_copy_w.get_slice(tid)
        thr_mma = tiled_mma.get_slice(tid)
        
        tCsC = thr_mma.partition_C(smem_accum)
        accum = tiled_mma.make_fragment_C(tCsC)
        accum.fill(0.0)
        
        for k_tile in range(0, self.BLOCK_K, self.BLOCK_K_TILE):
            x_ptr = (x[cta_b, seq_start, None].iterator + k_tile).align(16)
            gX_layout = cute.make_layout((self.BLOCK_M, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1))
            gX = cute.make_tensor(x_ptr, gX_layout)
            
            sX_aligned = cute.make_tensor(smem_x.iterator.align(16), smem_x.layout)
            tSgX = thr_copy_x.partition_S(gX)
            tSsX = thr_copy_x.partition_D(sX_aligned)
            cute.copy(gmem_tiled_copy_x, tSgX, tSsX)
            
            sW_sub_layout = cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K_TILE, 1))

            # Q Weight
            weight_ptr_q = (weight[cta_h * self.HEAD_DIM, None].iterator + k_tile).align(16)
            gW_q_layout = cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1))
            gW_q = cute.make_tensor(weight_ptr_q, gW_q_layout)
            sW_q = cute.make_tensor(smem_w.iterator.align(16), sW_sub_layout)
            tSgW_q = thr_copy_w.partition_S(gW_q)
            tSsW_q = thr_copy_w.partition_D(sW_q)
            cute.copy(gmem_tiled_copy_w, tSgW_q, tSsW_q)

            # K Weight
            weight_ptr_k = (weight[(self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + k_tile).align(16)
            gW_k_layout = cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1))
            gW_k = cute.make_tensor(weight_ptr_k, gW_k_layout)
            sW_k = cute.make_tensor((smem_w.iterator + self.HEAD_DIM * self.BLOCK_K_TILE).align(16), sW_sub_layout)
            tSgW_k = thr_copy_w.partition_S(gW_k)
            tSsW_k = thr_copy_w.partition_D(sW_k)
            cute.copy(gmem_tiled_copy_w, tSgW_k, tSsW_k)

            # V Weight
            weight_ptr_v = (weight[(2 * self.NUM_HEADS + cta_h) * self.HEAD_DIM, None].iterator + k_tile).align(16)
            gW_v_layout = cute.make_layout((self.HEAD_DIM, self.BLOCK_K_TILE), stride=(self.BLOCK_K, 1))
            gW_v = cute.make_tensor(weight_ptr_v, gW_v_layout)
            sW_v = cute.make_tensor((smem_w.iterator + 2 * self.HEAD_DIM * self.BLOCK_K_TILE).align(16), sW_sub_layout)
            tSgW_v = thr_copy_w.partition_S(gW_v)
            tSsW_v = thr_copy_w.partition_D(sW_v)
            cute.copy(gmem_tiled_copy_w, tSgW_v, tSsW_v)
            
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            
            tAsX = thr_mma.partition_A(smem_x)
            tBsW = thr_mma.partition_B(smem_w)
            
            rA = tiled_mma.make_fragment_A(tAsX)
            rB = tiled_mma.make_fragment_B(tBsW)
            
            for i in range(cute.size(rA)):
                rA[i] = tAsX[i]
            for i in range(cute.size(rB)):
                rB[i] = tBsW[i]
            
            cute.gemm(tiled_mma, accum, rA, rB, accum)
            cute.arch.sync_threads()
            
        for i in range(cute.size(accum)):
            tCsC[i] = accum[i]
        cute.arch.sync_threads()

        # Vectorized epilogue
        if tid < self.BLOCK_M and seq_start + tid < N:
            half_dim = self.HEAD_DIM // 2
            
            # Slices of shared memory (using flat layout offsets)
            q_ptr = smem_accum.iterator + tid * (3 * self.HEAD_DIM)
            k_ptr = smem_accum.iterator + tid * (3 * self.HEAD_DIM) + self.HEAD_DIM
            v_ptr = smem_accum.iterator + tid * (3 * self.HEAD_DIM) + 2 * self.HEAD_DIM
            
            q_smem = cute.make_tensor(q_ptr, cute.make_layout((self.HEAD_DIM,)))
            k_smem = cute.make_tensor(k_ptr, cute.make_layout((self.HEAD_DIM,)))
            v_smem = cute.make_tensor(v_ptr, cute.make_layout((self.HEAD_DIM,)))
            
            # Register fragments
            q_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
            k_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
            v_reg = cute.make_rmem_tensor((self.HEAD_DIM,), Float32)
            
            # Vectorized Shared-to-Register bulk load
            cute.autovec_copy(q_smem, q_reg)
            cute.autovec_copy(k_smem, k_reg)
            cute.autovec_copy(v_smem, v_reg)
            
            # Store V directly as BFloat16
            for j in cutlass.range(self.HEAD_DIM, unroll_full=True):
                v_proj[cta_b, cta_h, seq_start + tid, j] = BFloat16(v_reg[j])
            
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
            
            # Scale and Store Q / K directly on-the-fly to minimize register pressure
            scale_val = Float32(scales[cta_h, 0, 0])
            q_scale_factor = cute.math.exp(scale_val)
            
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
                
                # Store K directly as BFloat16
                k_proj[cta_b, cta_h, seq_start + tid, j] = BFloat16(kr0)
                k_proj[cta_b, cta_h, seq_start + tid, j + half_dim] = BFloat16(kr1)
                
                # Store Q directly as BFloat16
                q_proj[cta_b, cta_h, seq_start + tid, j] = BFloat16(qr0 * q_scale_factor)
                q_proj[cta_b, cta_h, seq_start + tid, j + half_dim] = BFloat16(qr1 * q_scale_factor)

_fwd_compile_cache = {}

def get_fused_qkv_fwd_kernel(num_warps=4, BLOCK_M=64, BLOCK_K=512, NUM_HEADS=16, HEAD_DIM=64):
    key = (num_warps, BLOCK_M, BLOCK_K, NUM_HEADS, HEAD_DIM)
    if key not in _fwd_compile_cache:
        from quack.compile_utils import make_fake_tensor
        
        B_sym = cute.sym_int()
        N_sym = cute.sym_int()
        
        fwd_op = FusedQKVCuteFwd(num_warps=num_warps, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM)
        
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
            options="--enable-tvm-ffi",
        )
        _fwd_compile_cache[key] = compiled_fwd
    return _fwd_compile_cache[key]

# Dispatch wrapper for benchmark testing
def dispatch_fused_qkv_cute_tvm(x, weight, rmsnorm_weight, scales, sin, cos):
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    
    # Keep in native BFloat16 contig format
    x_bf16 = x.to(torch.bfloat16).contiguous()
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    
    rmsnorm_weight_32 = rmsnorm_weight.to(torch.float32).contiguous()
    scales_32 = scales.to(torch.float32).contiguous()
    sin_32 = sin.to(torch.float32).contiguous()
    cos_32 = cos.to(torch.float32).contiguous()
    
    q_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    k_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    v_proj_bf16 = torch.empty((B, num_heads, N, head_dim), dtype=torch.bfloat16, device=x.device)
    
    fwd_kernel = get_fused_qkv_fwd_kernel(
        num_warps=4,
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

@torch.library.custom_op("flash_prologue::fused_qkv_cute", mutates_args=())
def fused_qkv_cute_op(
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
    
    fwd_kernel = get_fused_qkv_fwd_kernel(
        num_warps=4,
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

@fused_qkv_cute_op.register_fake
def _fused_qkv_cute_fake(x, weight, rmsnorm_weight, scales, sin, cos):
    B, N, hidden = x.shape
    num_heads = scales.shape[0]
    head_dim = rmsnorm_weight.shape[2] if rmsnorm_weight.dim() == 3 else rmsnorm_weight.shape[1]
    q_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    k_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    v_proj = torch.empty((B, num_heads, N, head_dim), dtype=x.dtype, device=x.device)
    return q_proj, k_proj, v_proj

def setup_fused_qkv_cute_context(ctx, inputs, output):
    x, weight, rmsnorm_weight, scales, sin, cos = inputs
    q, k, v = output
    ctx.save_for_backward(x, weight, rmsnorm_weight, scales, sin, cos)

def fused_qkv_cute_backward_hook(ctx, grad_q, grad_k, grad_v):
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
    "flash_prologue::fused_qkv_cute",
    fused_qkv_cute_backward_hook,
    setup_context=setup_fused_qkv_cute_context
)
