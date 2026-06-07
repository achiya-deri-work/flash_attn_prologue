import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import torch
import cutlass.cute as cute
from cutlass import BFloat16
from quack.compile_utils import make_fake_tensor
from quack import copy_utils

class TestTmaOp:
    @cute.jit
    def __call__(self, x, stream):
        # Setup smem
        smem_x_layout = cute.make_layout((128, 512), stride=(512, 1))
        smem_x_ptr = cute.arch.alloc_smem(x.element_type, 128 * 512)
        smem_x = cute.make_tensor(smem_x_ptr, smem_x_layout)

        # Allocate mbarrier pointer
        mbar_ptr = cute.arch.alloc_smem(cute.Int64, 1)

        # TMA Atom
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_x_atom, tma_x_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            x,
            smem_x_layout,
            (128, 512)
        )

        tid, _, _ = cute.arch.thread_idx()

        if tid == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
            # 128 * 512 * 2 bytes for bfloat16
            cute.arch.mbarrier_expect_tx(mbar_ptr, 128 * 512 * 2)

        cute.arch.sync_threads()

        batch_idx = 0
        cta_m = 0

        x_sliced = tma_x_tensor[batch_idx, None, None]
        gX = cute.local_tile(x_sliced, (128, 512), (cta_m, 0), proj=(1, 1))
        
        # Use copy_utils.tma_get_copy_fn
        copy_fn, s, g = copy_utils.tma_get_copy_fn(
            tma_x_atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gX,
            dst_tensor=smem_x,
            single_stage=True
        )

        # Perform copy with tma_bar_ptr
        if tid == 0:
            copy_fn(tma_bar_ptr=mbar_ptr)
            
        # Wait for the copy to finish on the mbarrier
        cute.arch.mbarrier_wait(mbar_ptr, 0)
        print("TMA Load executed successfully inside JIT!")

# Run kernel compilation
fwd_op = TestTmaOp()
x_fake = make_fake_tensor(BFloat16, (32, 512, 512), 16)
fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

try:
    print("Compiling TMA test kernel with --enable-tvm-ffi...")
    compiled_kernel = cute.compile(
        fwd_op,
        x_fake,
        fake_stream,
        options="--enable-tvm-ffi"
    )
    print("Compilation succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()
