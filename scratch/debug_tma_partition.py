import os
# Set architecture target to Blackwell
os.environ["CUTE_DSL_ARCH"] = "sm_120a"
os.environ["QUACK_ARCH"] = "sm_120a"

import torch
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor
from quack import copy_utils

class InspectTmaPartitionDx:
    def __init__(self):
        self.M_dim = 3072
        self.K_dim = 16384
        self.BLOCK_M = 64
        self.BLOCK_K_STAGE = 32
        self.ab_stage = 3
        
    def _setup(self):
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(self.BLOCK_K_STAGE, 1, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
            
        @cute.struct
        class SharedStorage:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, 24576], 1024]
        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(self, gA_tensor, stream):
        self._setup()
        
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_tensor,
            cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(self.BLOCK_K_STAGE, 1)),
            (self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        
        self.kernel(
            gA_tensor,
            tma_atom_a,
            tma_tensor_a,
            self.a_smem_layout_staged
        ).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        gA_tensor: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_tensor_a: cute.Tensor,
        a_smem_layout: cute.Layout
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gA_tile = cute.local_tile(tma_tensor_a, (self.BLOCK_M, self.BLOCK_K_STAGE), (0, None))
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_tile,
            dst_tensor=sX_aligned,
            single_stage=False
        )
        
        mbar_ptr = cute.arch.alloc_smem(cute.Int64, 1)
        if tidx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.sync_threads()
        
        # Issue TMA load
        if tidx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, self.BLOCK_M * self.BLOCK_K_STAGE * 2)
                copy_A(0, 0, tma_bar_ptr=mbar_ptr)
            
        cute.arch.mbarrier_wait(mbar_ptr, 0)

class InspectTmaPartitionDw:
    def __init__(self):
        self.M_dim = 3072
        self.K_dim = 16384
        self.BLOCK_M = 64
        self.BLOCK_K_STAGE = 32
        self.ab_stage = 3
        
    def _setup(self):
        self.a_smem_layout_staged = cute.make_layout(
            (self.BLOCK_M, self.BLOCK_K_STAGE, self.ab_stage),
            stride=(1, self.BLOCK_M, self.BLOCK_M * self.BLOCK_K_STAGE)
        )
            
        @cute.struct
        class SharedStorage:
            sX_raw: cute.struct.Align[cute.struct.MemRange[Float32, 24576], 1024]
        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(self, gA_tensor, stream):
        self._setup()
        
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gA_tensor,
            cute.make_layout((self.BLOCK_M, self.BLOCK_K_STAGE), stride=(1, self.BLOCK_M)),
            (self.BLOCK_M, self.BLOCK_K_STAGE)
        )
        
        self.kernel(
            gA_tensor,
            tma_atom_a,
            tma_tensor_a,
            self.a_smem_layout_staged
        ).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        gA_tensor: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_tensor_a: cute.Tensor,
        a_smem_layout: cute.Layout
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gA_tile = cute.local_tile(tma_tensor_a, (self.BLOCK_M, self.BLOCK_K_STAGE), (0, None))
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sX_aligned = cute.make_tensor(cute.recast_ptr(storage.sX_raw.data_ptr().align(16), dtype=BFloat16), a_smem_layout)
        
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_tile,
            dst_tensor=sX_aligned,
            single_stage=False
        )
        
        mbar_ptr = cute.arch.alloc_smem(cute.Int64, 1)
        if tidx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.sync_threads()
        
        # Issue TMA load
        if tidx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, self.BLOCK_M * self.BLOCK_K_STAGE * 2)
                copy_A(0, 0, tma_bar_ptr=mbar_ptr)
            
        cute.arch.mbarrier_wait(mbar_ptr, 0)

def test():
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    
    # 1. Compile DX
    print("Compiling DX inspect...")
    gA_dx_fake = make_fake_tensor(BFloat16, (3072, 16384), 16) # row-major
    op_dx = InspectTmaPartitionDx()
    compiled_dx = cute.compile(op_dx, gA_dx_fake, fake_stream, options="--enable-tvm-ffi")
    
    # Print source
    try:
        print("\n--- Compiled GPU Source ---")
        src = compiled_dx.imported_modules[0].get_source()
        print(src)
    except Exception as e:
        print("Error getting source:", e)
        
    print("Compilation inspect done. Exiting.")

if __name__ == "__main__":
    test()
