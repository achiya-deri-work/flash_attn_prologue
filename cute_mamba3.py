import os
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Boolean, const_expr, BFloat16
from cutlass._mlir.ir import Context
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from typing import Tuple, Optional
import cuda.bindings.driver as cuda # pyrefly: ignore
from advantage_first.layers.cute.autotuner import get_autotuned_config

class CuteMamba3Fwd:
    def __init__(self, num_warps: int = 4, N: int = 16, R: int = 2, TILE_S: int = 8, buf_size: Optional[int] = None, layout_option: str = "coprime", copy_layout_shape = (8, 4), mma_shape = (16, 8, 16)):
        self.num_warps = num_warps
        self.P = 16
        self.N = N
        self.R = R
        self.TILE_S = TILE_S
        self.custom_buf_size = buf_size
        self.layout_option = layout_option
        self.copy_layout_shape = copy_layout_shape
        self.mma_shape = mma_shape

    def _setup(self):
        if self.custom_buf_size is not None:
            self.buf_size = self.custom_buf_size
        else:
            fwd_smem_one_buf = 2 * self.num_warps * self.N * 20 * 4
            if self.TILE_S > 1:
                self.buf_size = 2 if (2 * fwd_smem_one_buf <= 80 * 1024) else 1
            else:
                self.buf_size = 1

        self.smem_layout = cute.make_layout(
            (self.buf_size, self.num_warps, self.N, 16),
            stride=(self.num_warps * self.N * 20, self.N * 20, 20, 1)
        )

        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        from cutlass import BFloat16, Float32

        self.mma = MmaF16BF16Op(BFloat16, Float32, self.mma_shape)
        self.tiled_mma = cute.make_tiled_mma(self.mma)

        if self.layout_option == "multiple":
            self.layout_C_smem = cute.make_layout(
                (self.num_warps, 8, self.N),
                stride=(10 * self.N, 1, 10)
            )
            self.layout_C_out = cute.make_layout(
                (self.num_warps, 16, 8),
                stride=(160, 1, 20)
            )
        else:
            self.layout_C_smem = cute.make_layout(
                (self.num_warps, 8, self.N),
                stride=(9 * self.N, 1, 9)
            )
            self.layout_C_out = cute.make_layout(
                (self.num_warps, 16, 8),
                stride=(136, 1, 17)
            )

        @cute.struct
        class SharedStorageFwd:
            sState: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
            sTrap: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
            sC: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.layout_C_smem)], 128]
            sC_out: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.layout_C_out)], 128]
        self.shared_storage = SharedStorageFwd

        # TiledCopy for loading: 128-bit CopyG2SOp (using cp.async)
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom = cute.make_copy_atom(copy_op, Float32, num_bits_per_copy=128)
        self.gmem_tiled_copy_state = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout(self.copy_layout_shape, order=(1, 0)),
            cute.make_layout((1, 4))
        )

        # TiledCopy for storing: 128-bit CopyUniversalOp (using universal store)
        copy_op_store = cute.nvgpu.CopyUniversalOp()
        copy_atom_store = cute.make_copy_atom(copy_op_store, Float32, num_bits_per_copy=128)
        self.gmem_tiled_copy_store = cute.make_tiled_copy_tv(
            copy_atom_store,
            cute.make_ordered_layout(self.copy_layout_shape, order=(1, 0)),
            cute.make_layout((1, 4))
        )

    @cute.jit
    def __call__(
        self,
        x_raw: cute.Tensor,      # [B, S, H * P]
        z_raw: cute.Tensor,      # [B, S, H * P]
        B_rot: cute.Tensor,      # [B, S, H, N, R]
        C_rot: cute.Tensor,      # [B, S, H, N, R]
        dt: cute.Tensor,         # [B, S, H]
        A: cute.Tensor,          # [B, S, H]
        val_lam: cute.Tensor,    # [B, S, H]
        ssm_state: cute.Tensor,  # [B, S, H, N, P]
        trap_mem: cute.Tensor,   # [B, S, H, N, P]
        W_X: cute.Tensor,        # [H, P, R]
        W_Z: cute.Tensor,        # [H, P, R]
        W_O: cute.Tensor,        # [H, P, R]
        D_skip: cute.Tensor,     # [H, P]
        out: cute.Tensor,        # [B, S, H * P]
        curr_state: cute.Tensor, # [B, S, H, N, P]
        curr_V: cute.Tensor,     # [B, S, H, N, P]
        stream: cuda.CUstream,
    ):
        self._setup()
        shape_B = x_raw.shape[0]
        shape_S = x_raw.shape[1]
        shape_H = W_X.shape[0]
        shape_P = W_X.shape[1]
        num_threads = self.num_warps * cute.arch.WARP_SIZE

        if self.buf_size == 2 and self.TILE_S > 1:
            self.kernel_pipelined(
                x_raw, z_raw, B_rot, C_rot, dt, A, val_lam, ssm_state, trap_mem,
                W_X, W_Z, W_O, D_skip, out, curr_state, curr_V,
                self.smem_layout, self.layout_C_smem, self.layout_C_out,
                self.tiled_mma,
                self.gmem_tiled_copy_state, self.gmem_tiled_copy_store
            ).launch(
                grid=[cute.ceil_div(shape_S, self.num_warps * self.TILE_S), shape_H * (shape_P // 16), shape_B],
                block=[num_threads, 1, 1],
                stream=stream,
            )
        else:
            self.kernel(
                x_raw, z_raw, B_rot, C_rot, dt, A, val_lam, ssm_state, trap_mem,
                W_X, W_Z, W_O, D_skip, out, curr_state, curr_V,
                self.smem_layout, self.layout_C_smem, self.layout_C_out,
                self.tiled_mma,
                self.gmem_tiled_copy_state, self.gmem_tiled_copy_store
            ).launch(
                grid=[cute.ceil_div(shape_S, self.num_warps * self.TILE_S), shape_H * (shape_P // 16), shape_B],
                block=[num_threads, 1, 1],
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        x_raw: cute.Tensor,
        z_raw: cute.Tensor,
        B_rot: cute.Tensor,
        C_rot: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        val_lam: cute.Tensor,
        ssm_state: cute.Tensor,
        trap_mem: cute.Tensor,
        W_X: cute.Tensor,
        W_Z: cute.Tensor,
        W_O: cute.Tensor,
        D_skip: cute.Tensor,
        out: cute.Tensor,
        curr_state: cute.Tensor,
        curr_V: cute.Tensor,
        smem_layout: cute.Layout,
        layout_C_smem: cute.Layout,
        layout_C_out: cute.Layout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_state: cute.TiledCopy,
        gmem_tiled_copy_store: cute.TiledCopy,
    ):
        shape_B = x_raw.shape[0]
        shape_S = x_raw.shape[1]
        shape_H = W_X.shape[0]
        shape_P = W_X.shape[1]

        # Re-layout 3D inputs/outputs to 4D internally using CuTe layouts
        layout_x = cute.make_layout(
            (shape_B, shape_S, shape_H, shape_P),
            stride=(shape_S * shape_H * shape_P, shape_H * shape_P, shape_P, 1)
        )
        x_raw = cute.make_tensor(x_raw.iterator, layout_x)
        z_raw = cute.make_tensor(z_raw.iterator, layout_x)
        out = cute.make_tensor(out.iterator, layout_x)

        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, bidx_z = cute.arch.block_idx()
        
        b = bidx_z
        h = bidx_y % shape_H
        c_block = bidx_y // shape_H

        warp_idx = tidx // 32
        lane_idx = tidx % 32
        L_each = (self.N * self.R) // 32

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        smem_tile_state = storage.sState.get_tensor(smem_layout)
        smem_tile_trap = storage.sTrap.get_tensor(smem_layout)
        smem_tile_C = storage.sC.get_tensor(layout_C_smem)
        smem_tile_C_out = storage.sC_out.get_tensor(layout_C_out)

        w = lane_idx // 4
        v = lane_idx % 4
        row_idx0 = (self.N // 8) * w
        col_idx0 = 4 * v

        # Initialize the padded columns of smem_tile_C to 0 once at the start of the warp
        elements_per_thread = cute.ceil_div(self.N, 32)
        for i in range(elements_per_thread):
            n_idx = i * 32 + lane_idx
            if n_idx < self.N:
                for r_pad in range(8 - self.R):
                    smem_tile_C[warp_idx, self.R + r_pad, n_idx] = Float32(0.0)

        # Temporal weight reuse: load weights into registers once per warp
        W_each = (16 * self.R) // 32
        W_X_val = cute.make_rmem_tensor((W_each,), Float32)
        W_Z_val = cute.make_rmem_tensor((W_each,), Float32)
        W_O_val = cute.make_rmem_tensor((W_each,), Float32)
        for idx in range(W_each):
            global_idx = idx * 32 + lane_idx
            p_idx = global_idx // self.R
            r_idx = global_idx % self.R
            W_X_val[idx] = Float32(W_X[h, c_block * 16 + p_idx, r_idx])
            W_Z_val[idx] = Float32(W_Z[h, c_block * 16 + p_idx, r_idx])
            W_O_val[idx] = Float32(W_O[h, c_block * 16 + p_idx, r_idx])

        D_skip_val = Float32(0.0)
        if lane_idx < 16:
            D_skip_val = Float32(D_skip[h, c_block * 16 + lane_idx])

        # Sequence tiling loop
        s_start_warp = (bidx_x * self.num_warps + warp_idx) * self.TILE_S

        # Prefetch sequence-dependent inputs for all TILE_S steps if TILE_S > 1
        if const_expr(self.TILE_S > 1):
            x_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            z_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            
            L_each = (self.N * self.R) // 32
            B_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            C_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            
            dt_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            A_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            lam_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)

            for s_offset in range(self.TILE_S):
                s_pref = s_start_warp + s_offset
                if s_pref < shape_S:
                    if lane_idx < 16:
                        x_tile[s_offset] = Float32(x_raw[b, s_pref, h, c_block * 16 + lane_idx])
                        z_tile[s_offset] = Float32(z_raw[b, s_pref, h, c_block * 16 + lane_idx])
                    else:
                        x_tile[s_offset] = Float32(0.0)
                        z_tile[s_offset] = Float32(0.0)
                    
                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        B_tile[s_offset, idx] = Float32(B_rot[b, s_pref, h, n_idx, r_idx])
                        C_tile[s_offset, idx] = Float32(C_rot[b, s_pref, h, n_idx, r_idx])
                        
                    dt_tile[s_offset] = Float32(dt[b, s_pref, h])
                    A_tile[s_offset] = Float32(A[b, s_pref, h])
                    lam_tile[s_offset] = Float32(val_lam[b, s_pref, h])
                else:
                    x_tile[s_offset] = Float32(0.0)
                    z_tile[s_offset] = Float32(0.0)
                    for idx in range(L_each):
                        B_tile[s_offset, idx] = Float32(0.0)
                        C_tile[s_offset, idx] = Float32(0.0)
                    dt_tile[s_offset] = Float32(0.0)
                    A_tile[s_offset] = Float32(0.0)
                    lam_tile[s_offset] = Float32(0.0)

        # If using double buffering, prefetch step 0
        if self.buf_size == 2:
            s_step0 = s_start_warp + 0
            if s_step0 < shape_S:
                gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                gState = cute.make_tensor((ssm_state[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gTrap = cute.make_tensor((trap_mem[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                
                sState_buf0 = smem_tile_state[0, warp_idx, None, None]
                sTrap_buf0 = smem_tile_trap[0, warp_idx, None, None]
                sState_buf0 = cute.make_tensor(sState_buf0.iterator.align(16), sState_buf0.layout)
                sTrap_buf0 = cute.make_tensor(sTrap_buf0.iterator.align(16), sTrap_buf0.layout)

                gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                tSgState = gmem_thr_copy_state.partition_S(gState)
                tSsState = gmem_thr_copy_state.partition_D(sState_buf0)
                cute.copy(gmem_tiled_copy_state, tSgState, tSsState)

                tSgTrap = gmem_thr_copy_state.partition_S(gTrap)
                tSsTrap = gmem_thr_copy_state.partition_D(sTrap_buf0)
                cute.copy(gmem_tiled_copy_state, tSgTrap, tSsTrap)
                cute.arch.cp_async_commit_group()

        for s_offset in range(self.TILE_S):
            s = s_start_warp + s_offset
            if s < shape_S:
                buf_idx = s_offset % 2 if self.buf_size == 2 else 0

                # 1. Prefetch next step (step + 1)
                if self.buf_size == 2:
                    s_next = s_start_warp + s_offset + 1
                    if s_next < shape_S:
                        gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                        gState_next = cute.make_tensor((ssm_state[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        gTrap_next = cute.make_tensor((trap_mem[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        
                        sState_next = smem_tile_state[(s_offset + 1) % 2, warp_idx, None, None]
                        sTrap_next = smem_tile_trap[(s_offset + 1) % 2, warp_idx, None, None]
                        sState_next = cute.make_tensor(sState_next.iterator.align(16), sState_next.layout)
                        sTrap_next = cute.make_tensor(sTrap_next.iterator.align(16), sTrap_next.layout)

                        gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                        tSgState_next = gmem_thr_copy_state.partition_S(gState_next)
                        tSsState_next = gmem_thr_copy_state.partition_D(sState_next)
                        cute.copy(gmem_tiled_copy_state, tSgState_next, tSsState_next)

                        tSgTrap_next = gmem_thr_copy_state.partition_S(gTrap_next)
                        tSsTrap_next = gmem_thr_copy_state.partition_D(sTrap_next)
                        cute.copy(gmem_tiled_copy_state, tSgTrap_next, tSsTrap_next)
                        cute.arch.cp_async_commit_group()

                # If no double-buffering, do blocking load
                if self.buf_size == 1:
                    gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                    gState = cute.make_tensor((ssm_state[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    gTrap = cute.make_tensor((trap_mem[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    
                    sState_buf0 = smem_tile_state[0, warp_idx, None, None]
                    sTrap_buf0 = smem_tile_trap[0, warp_idx, None, None]
                    sState_buf0 = cute.make_tensor(sState_buf0.iterator.align(16), sState_buf0.layout)
                    sTrap_buf0 = cute.make_tensor(sTrap_buf0.iterator.align(16), sTrap_buf0.layout)

                    gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                    tSgState = gmem_thr_copy_state.partition_S(gState)
                    tSsState = gmem_thr_copy_state.partition_D(sState_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgState, tSsState)

                    tSgTrap = gmem_thr_copy_state.partition_S(gTrap)
                    tSsTrap = gmem_thr_copy_state.partition_D(sTrap_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgTrap, tSsTrap)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                else:
                    # Double buffered: wait for current buffer's cp.async
                    s_next = s_start_warp + s_offset + 1
                    if s_next < shape_S:
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)

                cute.arch.sync_warp()

                # Read from SMEM to registers linearly (padded layout resolves bank conflicts)
                h_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                v_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        h_state[r, c] = smem_tile_state[buf_idx, warp_idx, row, col]
                        v_state[r, c] = smem_tile_trap[buf_idx, warp_idx, row, col]

                L_each = (self.N * self.R) // 32
                B_val = cute.make_rmem_tensor((L_each,), Float32)
                C_val = cute.make_rmem_tensor((L_each,), Float32)

                # Load inputs from prefetch registers or directly from GMEM
                if const_expr(self.TILE_S > 1):
                    x_val = x_tile[s_offset]
                    z_val = z_tile[s_offset]

                    for idx in range(L_each):
                        B_val[idx] = B_tile[s_offset, idx]
                        C_val[idx] = C_tile[s_offset, idx]

                    dt_val = dt_tile[s_offset]
                    A_val = A_tile[s_offset]
                    lam_val = lam_tile[s_offset]
                else:
                    x_val = Float32(0.0)
                    z_val = Float32(0.0)
                    if lane_idx < 16:
                        x_val = Float32(x_raw[b, s, h, c_block * 16 + lane_idx])
                        z_val = Float32(z_raw[b, s, h, c_block * 16 + lane_idx])

                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        B_val[idx] = Float32(B_rot[b, s, h, n_idx, r_idx])
                        C_val[idx] = Float32(C_rot[b, s, h, n_idx, r_idx])

                    dt_val = Float32(dt[b, s, h])
                    A_val = Float32(A[b, s, h])
                    lam_val = Float32(val_lam[b, s, h])

                # Write C_val to shared memory
                for idx in range(L_each):
                    global_idx = idx * 32 + lane_idx
                    n_idx = global_idx // self.R
                    r_idx = global_idx % self.R
                    smem_tile_C[warp_idx, r_idx, n_idx] = C_val[idx]
                cute.arch.sync_warp()

                # Extract linear column values for inputs/weights
                x_local = cute.make_rmem_tensor((4,), Float32)
                z_local = cute.make_rmem_tensor((4,), Float32)
                D_skip_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    col = col_idx0 + c
                    src_lane = col
                    x_local[c] = cute.arch.shuffle_sync(x_val, src_lane)
                    z_local[c] = cute.arch.shuffle_sync(z_val, src_lane)
                    D_skip_local[c] = cute.arch.shuffle_sync(D_skip_val, src_lane)

                B_local = cute.make_rmem_tensor((self.N // 8, self.R), Float32)
                for r in range(self.N // 8):
                    for rank in range(self.R):
                        global_idx = (row_idx0 + r) * self.R + rank
                        src_lane = global_idx % 32
                        reg_idx = global_idx // 32
                        val_B = Float32(0.0)
                        for l in range(L_each):
                            shfl_B = cute.arch.shuffle_sync(B_val[l], src_lane)
                            if reg_idx == l:
                                val_B = shfl_B
                        B_local[r, rank] = val_B

                W_X_local = cute.make_rmem_tensor((4, self.R), Float32)
                W_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
                W_O_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    col = col_idx0 + c
                    for rank in range(self.R):
                        global_idx = col * self.R + rank
                        src_lane = global_idx % 32
                        reg_idx = global_idx // 32
                        val_WX = Float32(0.0)
                        val_WZ = Float32(0.0)
                        val_WO = Float32(0.0)
                        for l in range(W_each):
                            shfl_WX = cute.arch.shuffle_sync(W_X_val[l], src_lane)
                            shfl_WZ = cute.arch.shuffle_sync(W_Z_val[l], src_lane)
                            shfl_WO = cute.arch.shuffle_sync(W_O_val[l], src_lane)
                            if reg_idx == l:
                                val_WX = shfl_WX
                                val_WZ = shfl_WZ
                                val_WO = shfl_WO
                        W_X_local[c, rank] = val_WX
                        W_Z_local[c, rank] = val_WZ
                        W_O_local[c, rank] = val_WO

                # Core forward math (now completely linear!)
                X_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        X_t_local[c, rank] = x_local[c] * W_X_local[c, rank]

                curr_V_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        val = Float32(0.0)
                        for rank in range(self.R):
                            val += B_local[r, rank] * X_t_local[c, rank]
                        curr_V_local[r, c] = val

                alpha_val = cute.math.exp(dt_val * A_val, fastmath=True)
                beta_val = (1.0 - lam_val) * dt_val * alpha_val
                gamma_val = lam_val * dt_val

                curr_state_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        curr_state_local[r, c] = alpha_val * h_state[r, c] + beta_val * v_state[r, c] + gamma_val * curr_V_local[r, c]

                # Store curr_state back to SMEM so GEMM can read it
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        smem_tile_state[buf_idx, warp_idx, row, col] = curr_state_local[r, c]
                cute.arch.sync_warp()

                # Get warp slices of shared memory
                sState_slice = smem_tile_state[buf_idx, warp_idx, None, None]
                sState_trans = cute.make_tensor(sState_slice.iterator, cute.make_layout((16, self.N), stride=(1, 20)))
                sB_slice = smem_tile_C[warp_idx, None, None]
                sC_out_warp = smem_tile_C_out[warp_idx, None, None]

                # Partition
                thr_mma = tiled_mma.get_slice(lane_idx)
                tAsA = thr_mma.partition_A(sState_trans)
                tBsB = thr_mma.partition_B(sB_slice)
                tCsC = thr_mma.partition_C(sC_out_warp)

                # Allocate fragments
                rA = tiled_mma.make_fragment_A(tAsA)
                rB = tiled_mma.make_fragment_B(tBsB)
                rC = tiled_mma.make_fragment_C(tCsC)

                # Load and cast using vectorized copies
                rA_f32 = cute.make_fragment_like(tAsA)
                rB_f32 = cute.make_fragment_like(tBsB)
                cute.autovec_copy(tAsA, rA_f32)
                cute.autovec_copy(tBsB, rB_f32)
                for i in range(cute.size(rA)):
                    rA[i] = BFloat16(rA_f32[i])
                for i in range(cute.size(rB)):
                    rB[i] = BFloat16(rB_f32[i])

                # Execute GEMM
                rC.fill(0.0)
                cute.gemm(tiled_mma, rC, rA, rB, rC)

                # Copy back to shared memory
                row_base = lane_idx // 4
                col_base = (lane_idx % 4) * 2
                sC_out_warp[row_base, col_base] = rC[(0, 0), 0, 0]
                sC_out_warp[row_base, col_base + 1] = rC[(1, 0), 0, 0]
                sC_out_warp[row_base + 8, col_base] = rC[(0, 1), 0, 0]
                sC_out_warp[row_base + 8, col_base + 1] = rC[(1, 1), 0, 0]

                cute.arch.sync_warp()

                # Read aligned outputs linearly
                Y_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        Y_t_local[c, rank] = sC_out_warp[col_idx0 + c, rank]

                Y_skip_local = cute.make_rmem_tensor((4, self.R), Float32)
                Z_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                out_gated_local = cute.make_rmem_tensor((4, self.R), Float32)

                for c in range(4):
                    for rank in range(self.R):
                        Y_skip_local[c, rank] = Y_t_local[c, rank] + D_skip_local[c] * X_t_local[c, rank]
                        Z_t_local[c, rank] = z_local[c] * W_Z_local[c, rank]
                        silu_Z = Z_t_local[c, rank] / (1.0 + cute.math.exp(-Z_t_local[c, rank], fastmath=True))
                        out_gated_local[c, rank] = Y_skip_local[c, rank] * silu_Z

                O_t_prime_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    val = Float32(0.0)
                    for rank in range(self.R):
                        val += out_gated_local[c, rank] * W_O_local[c, rank]
                    O_t_prime_local[c] = val

                # Store out directly to GMEM linearly using vectorized 128-bit stores
                if lane_idx < 4:
                    O_t_prime_bf16 = cute.make_rmem_tensor((4,), BFloat16)
                    for c in range(4):
                        O_t_prime_bf16[c] = BFloat16(O_t_prime_local[c])
                    out_ptr = (out[b, s, h, None].iterator + c_block * 16 + 4 * lane_idx).align(8)
                    gOut_slice = cute.make_tensor(out_ptr, cute.make_layout((4,), stride=(1,)))
                    cute.autovec_copy(O_t_prime_bf16, gOut_slice)

                # Store curr_V back to SMEM linearly (curr_state was already stored before GEMM)
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        smem_tile_trap[buf_idx, warp_idx, row, col] = curr_V_local[r, c]

                cute.arch.sync_warp()

                # Vectorized store of curr_state and curr_V from SMEM to GMEM (perfectly linear!)
                gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                gCurrState = cute.make_tensor((curr_state[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gCurrV = cute.make_tensor((curr_V[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)

                sState_buf = smem_tile_state[buf_idx, warp_idx, None, None]
                sState_buf = cute.make_tensor(sState_buf.iterator.align(16), sState_buf.layout)
                sTrap_buf = smem_tile_trap[buf_idx, warp_idx, None, None]
                sTrap_buf = cute.make_tensor(sTrap_buf.iterator.align(16), sTrap_buf.layout)

                gmem_thr_copy_store = gmem_tiled_copy_store.get_slice(lane_idx)
                tSgCurrState = gmem_thr_copy_store.partition_D(gCurrState)
                tSsState_s2g = gmem_thr_copy_store.partition_S(sState_buf)
                cute.copy(gmem_tiled_copy_store, tSsState_s2g, tSgCurrState)

                tSgCurrV = gmem_thr_copy_store.partition_D(gCurrV)
                tSsTrap_s2g = gmem_thr_copy_store.partition_S(sTrap_buf)
                cute.copy(gmem_tiled_copy_store, tSsTrap_s2g, tSgCurrV)



    @cute.kernel
    def kernel_pipelined(
        self,
        x_raw: cute.Tensor,
        z_raw: cute.Tensor,
        B_rot: cute.Tensor,
        C_rot: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        val_lam: cute.Tensor,
        ssm_state: cute.Tensor,
        trap_mem: cute.Tensor,
        W_X: cute.Tensor,
        W_Z: cute.Tensor,
        W_O: cute.Tensor,
        D_skip: cute.Tensor,
        out: cute.Tensor,
        curr_state: cute.Tensor,
        curr_V: cute.Tensor,
        smem_layout: cute.Layout,
        layout_C_smem: cute.Layout,
        layout_C_out: cute.Layout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_state: cute.TiledCopy,
        gmem_tiled_copy_store: cute.TiledCopy,
    ):
        shape_B = x_raw.shape[0]
        shape_S = x_raw.shape[1]
        shape_H = W_X.shape[0]
        shape_P = W_X.shape[1]

        # Re-layout 3D inputs/outputs to 4D internally using CuTe layouts
        layout_x = cute.make_layout(
            (shape_B, shape_S, shape_H, shape_P),
            stride=(shape_S * shape_H * shape_P, shape_H * shape_P, shape_P, 1)
        )
        x_raw = cute.make_tensor(x_raw.iterator, layout_x)
        z_raw = cute.make_tensor(z_raw.iterator, layout_x)
        out = cute.make_tensor(out.iterator, layout_x)

        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, bidx_z = cute.arch.block_idx()
        
        b = bidx_z
        h = bidx_y % shape_H
        c_block = bidx_y // shape_H

        warp_idx = tidx // 32
        lane_idx = tidx % 32
        L_each = (self.N * self.R) // 32

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        smem_tile_state = storage.sState.get_tensor(smem_layout)
        smem_tile_trap = storage.sTrap.get_tensor(smem_layout)
        smem_tile_C = storage.sC.get_tensor(layout_C_smem)
        smem_tile_C_out = storage.sC_out.get_tensor(layout_C_out)

        w = lane_idx // 4
        v = lane_idx % 4
        row_idx0 = (self.N // 8) * w
        col_idx0 = 4 * v

        # Initialize the padded columns of smem_tile_C to 0 once at the start of the warp
        elements_per_thread = cute.ceil_div(self.N, 32)
        for i in range(elements_per_thread):
            n_idx = i * 32 + lane_idx
            if n_idx < self.N:
                for r_pad in range(8 - self.R):
                    smem_tile_C[warp_idx, self.R + r_pad, n_idx] = Float32(0.0)

        # Temporal weight reuse: load weights into registers once per warp
        W_each = (16 * self.R) // 32
        W_X_val = cute.make_rmem_tensor((W_each,), Float32)
        W_Z_val = cute.make_rmem_tensor((W_each,), Float32)
        W_O_val = cute.make_rmem_tensor((W_each,), Float32)
        for idx in range(W_each):
            global_idx = idx * 32 + lane_idx
            p_idx = global_idx // self.R
            r_idx = global_idx % self.R
            W_X_val[idx] = Float32(W_X[h, c_block * 16 + p_idx, r_idx])
            W_Z_val[idx] = Float32(W_Z[h, c_block * 16 + p_idx, r_idx])
            W_O_val[idx] = Float32(W_O[h, c_block * 16 + p_idx, r_idx])

        D_skip_val = Float32(0.0)
        if lane_idx < 16:
            D_skip_val = Float32(D_skip[h, c_block * 16 + lane_idx])

        s_start_warp = (bidx_x * self.num_warps + warp_idx) * self.TILE_S

        if s_start_warp < shape_S:
            # Allocate double-buffered GEMM registers
            sState_slice0 = smem_tile_state[0, warp_idx, None, None]
            sState_trans0 = cute.make_tensor(sState_slice0.iterator, cute.make_layout((16, self.N), stride=(1, 20)))
            sB_slice = smem_tile_C[warp_idx, None, None]

            thr_mma = tiled_mma.get_slice(lane_idx)
            tAsA0 = thr_mma.partition_A(sState_trans0)
            tBsB = thr_mma.partition_B(sB_slice)
            tCsC = thr_mma.partition_C(smem_tile_C_out[warp_idx, None, None])

            rA0 = tiled_mma.make_fragment_A(tAsA0)
            rA1 = tiled_mma.make_fragment_A(tAsA0)
            rB0 = tiled_mma.make_fragment_B(tBsB)
            rB1 = tiled_mma.make_fragment_B(tBsB)
            rC0 = tiled_mma.make_fragment_C(tCsC)
            rC1 = tiled_mma.make_fragment_C(tCsC)

            # Double-buffered post-processing registers
            z_local0 = cute.make_rmem_tensor((4,), Float32)
            z_local1 = cute.make_rmem_tensor((4,), Float32)
            X_t_local0 = cute.make_rmem_tensor((4, self.R), Float32)
            X_t_local1 = cute.make_rmem_tensor((4, self.R), Float32)

            # Prefetch sequence-dependent inputs for all TILE_S steps
            x_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            z_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            
            L_each = (self.N * self.R) // 32
            B_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            C_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            
            dt_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            A_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            lam_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)

            for s_offset in range(self.TILE_S):
                s_pref = s_start_warp + s_offset
                if s_pref < shape_S:
                    if lane_idx < 16:
                        x_tile[s_offset] = Float32(x_raw[b, s_pref, h, c_block * 16 + lane_idx])
                        z_tile[s_offset] = Float32(z_raw[b, s_pref, h, c_block * 16 + lane_idx])
                    else:
                        x_tile[s_offset] = Float32(0.0)
                        z_tile[s_offset] = Float32(0.0)
                    
                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        B_tile[s_offset, idx] = Float32(B_rot[b, s_pref, h, n_idx, r_idx])
                        C_tile[s_offset, idx] = Float32(C_rot[b, s_pref, h, n_idx, r_idx])
                        
                    dt_tile[s_offset] = Float32(dt[b, s_pref, h])
                    A_tile[s_offset] = Float32(A[b, s_pref, h])
                    lam_tile[s_offset] = Float32(val_lam[b, s_pref, h])
                else:
                    x_tile[s_offset] = Float32(0.0)
                    z_tile[s_offset] = Float32(0.0)
                    for idx in range(L_each):
                        B_tile[s_offset, idx] = Float32(0.0)
                        C_tile[s_offset, idx] = Float32(0.0)
                    dt_tile[s_offset] = Float32(0.0)
                    A_tile[s_offset] = Float32(0.0)
                    lam_tile[s_offset] = Float32(0.0)

            row_base = lane_idx // 4
            col_base = (lane_idx % 4) * 2

            # Extract weights and skip projections once before the loop
            D_skip_local = cute.make_rmem_tensor((4,), Float32)
            for c in range(4):
                col = col_idx0 + c
                src_lane = col
                D_skip_local[c] = cute.arch.shuffle_sync(D_skip_val, src_lane)

            W_X_local = cute.make_rmem_tensor((4, self.R), Float32)
            W_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
            W_O_local = cute.make_rmem_tensor((4, self.R), Float32)
            for c in range(4):
                col = col_idx0 + c
                for rank in range(self.R):
                    global_idx = col * self.R + rank
                    src_lane = global_idx % 32
                    reg_idx = global_idx // 32
                    val_WX = Float32(0.0)
                    val_WZ = Float32(0.0)
                    val_WO = Float32(0.0)
                    for l in range(W_each):
                        shfl_WX = cute.arch.shuffle_sync(W_X_val[l], src_lane)
                        shfl_WZ = cute.arch.shuffle_sync(W_Z_val[l], src_lane)
                        shfl_WO = cute.arch.shuffle_sync(W_O_val[l], src_lane)
                        if reg_idx == l:
                            val_WX = shfl_WX
                            val_WZ = shfl_WZ
                            val_WO = shfl_WO
                    W_X_local[c, rank] = val_WX
                    W_Z_local[c, rank] = val_WZ
                    W_O_local[c, rank] = val_WO

            # Prefetch step 0 (s_step0) using cp.async
            s_step0 = s_start_warp + 0
            if s_step0 < shape_S:
                gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                gState = cute.make_tensor((ssm_state[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gTrap = cute.make_tensor((trap_mem[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                
                sState_buf0 = smem_tile_state[0, warp_idx, None, None]
                sTrap_buf0 = smem_tile_trap[0, warp_idx, None, None]
                sState_buf0 = cute.make_tensor(sState_buf0.iterator.align(16), sState_buf0.layout)
                sTrap_buf0 = cute.make_tensor(sTrap_buf0.iterator.align(16), sTrap_buf0.layout)

                gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                tSgState = gmem_thr_copy_state.partition_S(gState)
                tSsState = gmem_thr_copy_state.partition_D(sState_buf0)
                cute.copy(gmem_tiled_copy_state, tSgState, tSsState)

                tSgTrap = gmem_thr_copy_state.partition_S(gTrap)
                tSsTrap = gmem_thr_copy_state.partition_D(sTrap_buf0)
                cute.copy(gmem_tiled_copy_state, tSgTrap, tSsTrap)
                cute.arch.cp_async_commit_group()

            # Sequential tile loop covering (TILE_S + 1) steps for software pipelining
            for s_offset in range(self.TILE_S + 1):
                s_curr = s_start_warp + s_offset - 1
                s_next = s_start_warp + s_offset

                # --- PART A: Synchronize & Write-Back for s_curr ---
                if s_curr >= s_start_warp and s_curr < shape_S:
                    curr_buf_idx = (s_offset - 1) % 2
                    sC_out_warp = smem_tile_C_out[warp_idx, None, None]

                    # Write GEMM accumulator registers back to shared memory
                    if (s_offset - 1) % 2 == 0:
                        sC_out_warp[row_base, col_base] = rC0[(0, 0), 0, 0]
                        sC_out_warp[row_base, col_base + 1] = rC0[(1, 0), 0, 0]
                        sC_out_warp[row_base + 8, col_base] = rC0[(0, 1), 0, 0]
                        sC_out_warp[row_base + 8, col_base + 1] = rC0[(1, 1), 0, 0]
                    else:
                        sC_out_warp[row_base, col_base] = rC1[(0, 0), 0, 0]
                        sC_out_warp[row_base, col_base + 1] = rC1[(1, 0), 0, 0]
                        sC_out_warp[row_base + 8, col_base] = rC1[(0, 1), 0, 0]
                        sC_out_warp[row_base + 8, col_base + 1] = rC1[(1, 1), 0, 0]

                    cute.arch.sync_warp()

                    # Read aligned outputs linearly
                    Y_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                    for c in range(4):
                        for rank in range(self.R):
                            Y_t_local[c, rank] = sC_out_warp[col_idx0 + c, rank]

                    Y_skip_local = cute.make_rmem_tensor((4, self.R), Float32)
                    Z_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                    out_gated_local = cute.make_rmem_tensor((4, self.R), Float32)

                    # Retrieve step-specific registers
                    if (s_offset - 1) % 2 == 0:
                        for c in range(4):
                            for rank in range(self.R):
                                Y_skip_local[c, rank] = Y_t_local[c, rank] + D_skip_local[c] * X_t_local0[c, rank]
                                Z_t_local[c, rank] = z_local0[c] * W_Z_local[c, rank]
                                silu_Z = Z_t_local[c, rank] / (1.0 + cute.math.exp(-Z_t_local[c, rank], fastmath=True))
                                out_gated_local[c, rank] = Y_skip_local[c, rank] * silu_Z
                    else:
                        for c in range(4):
                            for rank in range(self.R):
                                Y_skip_local[c, rank] = Y_t_local[c, rank] + D_skip_local[c] * X_t_local1[c, rank]
                                Z_t_local[c, rank] = z_local1[c] * W_Z_local[c, rank]
                                silu_Z = Z_t_local[c, rank] / (1.0 + cute.math.exp(-Z_t_local[c, rank], fastmath=True))
                                out_gated_local[c, rank] = Y_skip_local[c, rank] * silu_Z

                    O_t_prime_local = cute.make_rmem_tensor((4,), Float32)
                    for c in range(4):
                        val = Float32(0.0)
                        for rank in range(self.R):
                            val += out_gated_local[c, rank] * W_O_local[c, rank]
                        O_t_prime_local[c] = val

                    # Store out directly to GMEM linearly using vectorized 128-bit stores
                    if lane_idx < 4:
                        O_t_prime_bf16 = cute.make_rmem_tensor((4,), BFloat16)
                        for c in range(4):
                            O_t_prime_bf16[c] = BFloat16(O_t_prime_local[c])
                        out_ptr = (out[b, s_curr, h, None].iterator + c_block * 16 + 4 * lane_idx).align(8)
                        gOut_slice = cute.make_tensor(out_ptr, cute.make_layout((4,), stride=(1,)))
                        cute.autovec_copy(O_t_prime_bf16, gOut_slice)

                    # Vectorized store of curr_state and curr_V from SMEM to GMEM
                    gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                    gCurrState = cute.make_tensor((curr_state[b, s_curr, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    gCurrV = cute.make_tensor((curr_V[b, s_curr, h, None, None].iterator + c_block * 16).align(16), gState_layout)

                    sState_buf = smem_tile_state[curr_buf_idx, warp_idx, None, None]
                    sState_buf = cute.make_tensor(sState_buf.iterator.align(16), sState_buf.layout)
                    sTrap_buf = smem_tile_trap[curr_buf_idx, warp_idx, None, None]
                    sTrap_buf = cute.make_tensor(sTrap_buf.iterator.align(16), sTrap_buf.layout)

                    gmem_thr_copy_store = gmem_tiled_copy_store.get_slice(lane_idx)
                    tSgCurrState = gmem_thr_copy_store.partition_D(gCurrState)
                    tSsState_s2g = gmem_thr_copy_store.partition_S(sState_buf)
                    cute.copy(gmem_tiled_copy_store, tSsState_s2g, tSgCurrState)

                    tSgCurrV = gmem_thr_copy_store.partition_D(gCurrV)
                    tSsTrap_s2g = gmem_thr_copy_store.partition_S(sTrap_buf)
                    cute.copy(gmem_tiled_copy_store, tSsTrap_s2g, tSgCurrV)

                # --- PART B: Pre-work for s_next (load, ODE compute, and setup GEMM) ---
                if s_next < shape_S and s_next < s_start_warp + self.TILE_S:
                    buf_idx = s_offset % 2

                    # Prefetch next step (s_next + 1)
                    s_next_next = s_next + 1
                    if s_next_next < shape_S:
                        gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                        gState_next = cute.make_tensor((ssm_state[b, s_next_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        gTrap_next = cute.make_tensor((trap_mem[b, s_next_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        
                        sState_next = smem_tile_state[s_next_next % 2, warp_idx, None, None]
                        sTrap_next = smem_tile_trap[s_next_next % 2, warp_idx, None, None]
                        sState_next = cute.make_tensor(sState_next.iterator.align(16), sState_next.layout)
                        sTrap_next = cute.make_tensor(sTrap_next.iterator.align(16), sTrap_next.layout)

                        gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                        tSgState_next = gmem_thr_copy_state.partition_S(gState_next)
                        tSsState_next = gmem_thr_copy_state.partition_D(sState_next)
                        cute.copy(gmem_tiled_copy_state, tSgState_next, tSsState_next)

                        tSgTrap_next = gmem_thr_copy_state.partition_S(gTrap_next)
                        tSsTrap_next = gmem_thr_copy_state.partition_D(sTrap_next)
                        cute.copy(gmem_tiled_copy_state, tSgTrap_next, tSsTrap_next)
                        cute.arch.cp_async_commit_group()

                    # Wait for cp.async of s_next state/trap
                    s_next_next = s_next + 1
                    if s_next_next < shape_S:
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_warp()

                    # Read from SMEM to registers linearly
                    h_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                    v_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                    for r in range(self.N // 8):
                        row = row_idx0 + r
                        for c in range(4):
                            col = col_idx0 + c
                            h_state[r, c] = smem_tile_state[buf_idx, warp_idx, row, col]
                            v_state[r, c] = smem_tile_trap[buf_idx, warp_idx, row, col]

                    # Load inputs from prefetch registers instead of GMEM
                    x_val = x_tile[s_offset]
                    z_val = z_tile[s_offset]

                    L_each = (self.N * self.R) // 32
                    B_val = cute.make_rmem_tensor((L_each,), Float32)
                    C_val = cute.make_rmem_tensor((L_each,), Float32)
                    for idx in range(L_each):
                        B_val[idx] = B_tile[s_offset, idx]
                        C_val[idx] = C_tile[s_offset, idx]

                    # Write C_val to shared memory
                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        smem_tile_C[warp_idx, r_idx, n_idx] = C_val[idx]
                    cute.arch.sync_warp()

                    dt_val = dt_tile[s_offset]
                    A_val = A_tile[s_offset]
                    lam_val = lam_tile[s_offset]

                    # Extract linear column values for inputs/weights
                    x_local = cute.make_rmem_tensor((4,), Float32)
                    z_local_tmp = cute.make_rmem_tensor((4,), Float32)
                    for c in range(4):
                        col = col_idx0 + c
                        src_lane = col
                        x_local[c] = cute.arch.shuffle_sync(x_val, src_lane)
                        z_local_tmp[c] = cute.arch.shuffle_sync(z_val, src_lane)

                    # Store z_local to ping-pong register buffer
                    if s_offset % 2 == 0:
                        for c in range(4):
                            z_local0[c] = z_local_tmp[c]
                    else:
                        for c in range(4):
                            z_local1[c] = z_local_tmp[c]

                    B_local = cute.make_rmem_tensor((self.N // 8, self.R), Float32)
                    for r in range(self.N // 8):
                        for rank in range(self.R):
                            global_idx = (row_idx0 + r) * self.R + rank
                            src_lane = global_idx % 32
                            reg_idx = global_idx // 32
                            val_B = Float32(0.0)
                            for l in range(L_each):
                                shfl_B = cute.arch.shuffle_sync(B_val[l], src_lane)
                                if reg_idx == l:
                                    val_B = shfl_B
                            B_local[r, rank] = val_B

                    # Core forward math
                    X_t_local_tmp = cute.make_rmem_tensor((4, self.R), Float32)
                    for c in range(4):
                        for rank in range(self.R):
                            X_t_local_tmp[c, rank] = x_local[c] * W_X_local[c, rank]

                    # Store X_t_local to ping-pong register buffer
                    if s_offset % 2 == 0:
                        for c in range(4):
                            for rank in range(self.R):
                                X_t_local0[c, rank] = X_t_local_tmp[c, rank]
                    else:
                        for c in range(4):
                            for rank in range(self.R):
                                X_t_local1[c, rank] = X_t_local_tmp[c, rank]

                    curr_V_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                    for r in range(self.N // 8):
                        for c in range(4):
                            val = Float32(0.0)
                            for rank in range(self.R):
                                val += B_local[r, rank] * X_t_local_tmp[c, rank]
                            curr_V_local[r, c] = val

                    alpha_val = cute.math.exp(dt_val * A_val, fastmath=True)
                    beta_val = (1.0 - lam_val) * dt_val * alpha_val
                    gamma_val = lam_val * dt_val

                    curr_state_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                    for r in range(self.N // 8):
                        for c in range(4):
                            curr_state_local[r, c] = alpha_val * h_state[r, c] + beta_val * v_state[r, c] + gamma_val * curr_V_local[r, c]

                    # Store curr_state back to SMEM
                    for r in range(self.N // 8):
                        row = row_idx0 + r
                        for c in range(4):
                            col = col_idx0 + c
                            smem_tile_state[buf_idx, warp_idx, row, col] = curr_state_local[r, c]

                    # Store curr_V back to SMEM
                    for r in range(self.N // 8):
                        row = row_idx0 + r
                        for c in range(4):
                            col = col_idx0 + c
                            smem_tile_trap[buf_idx, warp_idx, row, col] = curr_V_local[r, c]
                    cute.arch.sync_warp()

                    # Load GEMM operands into partition views and copy vectorially
                    sState_slice = smem_tile_state[buf_idx, warp_idx, None, None]
                    sState_trans = cute.make_tensor(sState_slice.iterator, cute.make_layout((16, self.N), stride=(1, 20)))
                    sB_slice = smem_tile_C[warp_idx, None, None]

                    tAsA = thr_mma.partition_A(sState_trans)
                    tBsB = thr_mma.partition_B(sB_slice)

                    rA_f32 = cute.make_fragment_like(tAsA)
                    rB_f32 = cute.make_fragment_like(tBsB)
                    cute.autovec_copy(tAsA, rA_f32)
                    cute.autovec_copy(tBsB, rB_f32)

                    if s_offset % 2 == 0:
                        for i in range(cute.size(rA0)):
                            rA0[i] = BFloat16(rA_f32[i])
                        for i in range(cute.size(rB0)):
                            rB0[i] = BFloat16(rB_f32[i])
                    else:
                        for i in range(cute.size(rA1)):
                            rA1[i] = BFloat16(rA_f32[i])
                        for i in range(cute.size(rB1)):
                            rB1[i] = BFloat16(rB_f32[i])

                # --- PART C: Launch GEMM for s_next ---
                if s_next < shape_S and s_next < s_start_warp + self.TILE_S:
                    if s_offset % 2 == 0:
                        rC0.fill(0.0)
                        cute.gemm(tiled_mma, rC0, rA0, rB0, rC0)
                    else:
                        rC1.fill(0.0)
                        cute.gemm(tiled_mma, rC1, rA1, rB1, rC1)


class CuteMamba3Bwd:
    def __init__(self, num_warps: int = 4, N: int = 16, R: int = 2, TILE_S: int = 8, buf_size: Optional[int] = None, layout_option: str = "coprime", copy_layout_shape = (8, 4), mma_shape = (16, 8, 16)):
        self.num_warps = num_warps
        self.P = 16
        self.N = N
        self.R = R
        self.TILE_S = TILE_S
        self.custom_buf_size = buf_size
        self.layout_option = layout_option
        self.copy_layout_shape = copy_layout_shape
        self.mma_shape = mma_shape

    def _setup(self):
        if self.custom_buf_size is not None:
            self.buf_size = self.custom_buf_size
        else:
            bwd_smem_one_buf = 4 * self.num_warps * self.N * 20 * 4
            if self.TILE_S > 1:
                self.buf_size = 2 if (2 * bwd_smem_one_buf <= 80 * 1024) else 1
            else:
                self.buf_size = 1

        self.smem_layout = cute.make_layout(
            (self.buf_size, self.num_warps, self.N, 16),
            stride=(self.num_warps * self.N * 20, self.N * 20, 20, 1)
        )

        from cutlass.cute.nvgpu.warp import MmaF16BF16Op
        from cutlass import BFloat16, Float32

        self.mma = MmaF16BF16Op(BFloat16, Float32, self.mma_shape)
        self.tiled_mma = cute.make_tiled_mma(self.mma)

        if self.layout_option == "multiple":
            self.layout_GradOut = cute.make_layout(
                (self.num_warps, 16, 8),
                stride=(160, 1, 20)
            )
            self.layout_GradC = cute.make_layout(
                (self.num_warps, self.N, 8),
                stride=(self.N * 10, 10, 1)
            )
            self.layout_B_smem = cute.make_layout(
                (self.num_warps, self.N, 8),
                stride=(self.N * 10, 10, 1)
            )
        else:
            self.layout_GradOut = cute.make_layout(
                (self.num_warps, 16, 8),
                stride=(136, 1, 17)
            )
            self.layout_GradC = cute.make_layout(
                (self.num_warps, self.N, 8),
                stride=(self.N * 9, 9, 1)
            )
            self.layout_B_smem = cute.make_layout(
                (self.num_warps, self.N, 8),
                stride=(self.N * 9, 9, 1)
            )

        @cute.struct
        class SharedStorageBwd:
            sState: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
            sTrap: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
            sGradState: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
            sGradV: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout)], 128]
        self.shared_storage = SharedStorageBwd

        # TiledCopy for loading: 128-bit CopyG2SOp (using cp.async)
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom = cute.make_copy_atom(copy_op, Float32, num_bits_per_copy=128)
        self.gmem_tiled_copy_state = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout(self.copy_layout_shape, order=(1, 0)),
            cute.make_layout((1, 4))
        )

        # TiledCopy for storing: 128-bit CopyUniversalOp (using universal store)
        copy_op_store = cute.nvgpu.CopyUniversalOp()
        copy_atom_store = cute.make_copy_atom(copy_op_store, Float32, num_bits_per_copy=128)
        self.gmem_tiled_copy_store = cute.make_tiled_copy_tv(
            copy_atom_store,
            cute.make_ordered_layout(self.copy_layout_shape, order=(1, 0)),
            cute.make_layout((1, 4))
        )

    @cute.jit
    def __call__(
        self,
        grad_out: cute.Tensor,   # [B, S, H * P]
        x_raw: cute.Tensor,      # [B, S, H * P]
        z_raw: cute.Tensor,      # [B, S, H * P]
        B_rot: cute.Tensor,      # [B, S, H, N, R]
        C_rot: cute.Tensor,      # [B, S, H, N, R]
        dt: cute.Tensor,         # [B, S, H]
        A: cute.Tensor,          # [B, S, H]
        val_lam: cute.Tensor,    # [B, S, H]
        ssm_state: cute.Tensor,  # [B, S, H, N, P]
        trap_mem: cute.Tensor,   # [B, S, H, N, P]
        W_X: cute.Tensor,        # [H, P, R]
        W_Z: cute.Tensor,        # [H, P, R]
        W_O: cute.Tensor,        # [H, P, R]
        D_skip: cute.Tensor,     # [H, P]
        grad_curr_state: cute.Tensor,
        grad_curr_V: cute.Tensor,
        dx_raw: cute.Tensor,     # [B, S, H * P]
        dz_raw: cute.Tensor,     # [B, S, H * P]
        dB_rot: cute.Tensor,
        dC_rot: cute.Tensor,
        dt_grad: cute.Tensor,
        A_grad: cute.Tensor,
        lam_grad: cute.Tensor,
        dssm_state: cute.Tensor,
        dtrap_mem: cute.Tensor,
        dW_X: cute.Tensor,
        dW_Z: cute.Tensor,
        dW_O: cute.Tensor,
        dD_skip: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._setup()
        shape_B = x_raw.shape[0]
        shape_S = x_raw.shape[1]
        shape_H = W_X.shape[0]
        shape_P = W_X.shape[1]
        num_threads = self.num_warps * cute.arch.WARP_SIZE

        self.kernel(
            grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, val_lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip,
            grad_curr_state, grad_curr_V,
            dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
            dW_X, dW_Z, dW_O, dD_skip,
            self.smem_layout, self.layout_GradOut, self.layout_GradC, self.layout_B_smem,
            self.tiled_mma, self.gmem_tiled_copy_state, self.gmem_tiled_copy_store
        ).launch(
            grid=[cute.ceil_div(shape_S, self.num_warps * self.TILE_S), shape_H * (shape_P // 16), shape_B],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        grad_out: cute.Tensor,
        x_raw: cute.Tensor,
        z_raw: cute.Tensor,
        B_rot: cute.Tensor,
        C_rot: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        val_lam: cute.Tensor,
        ssm_state: cute.Tensor,
        trap_mem: cute.Tensor,
        W_X: cute.Tensor,
        W_Z: cute.Tensor,
        W_O: cute.Tensor,
        D_skip: cute.Tensor,
        grad_curr_state: cute.Tensor,
        grad_curr_V: cute.Tensor,
        dx_raw: cute.Tensor,
        dz_raw: cute.Tensor,
        dB_rot: cute.Tensor,
        dC_rot: cute.Tensor,
        dt_grad: cute.Tensor,
        A_grad: cute.Tensor,
        lam_grad: cute.Tensor,
        dssm_state: cute.Tensor,
        dtrap_mem: cute.Tensor,
        dW_X: cute.Tensor,
        dW_Z: cute.Tensor,
        dW_O: cute.Tensor,
        dD_skip: cute.Tensor,
        smem_layout: cute.Layout,
        layout_GradOut: cute.Layout,
        layout_GradC: cute.Layout,
        layout_B_smem: cute.Layout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_state: cute.TiledCopy,
        gmem_tiled_copy_store: cute.TiledCopy,
    ):
        shape_B = x_raw.shape[0]
        shape_S = x_raw.shape[1]
        shape_H = W_X.shape[0]
        shape_P = W_X.shape[1]

        # Re-layout 3D inputs/outputs to 4D internally using CuTe layouts
        layout_x = cute.make_layout(
            (shape_B, shape_S, shape_H, shape_P),
            stride=(shape_S * shape_H * shape_P, shape_H * shape_P, shape_P, 1)
        )
        grad_out = cute.make_tensor(grad_out.iterator, layout_x)
        x_raw = cute.make_tensor(x_raw.iterator, layout_x)
        z_raw = cute.make_tensor(z_raw.iterator, layout_x)
        dx_raw = cute.make_tensor(dx_raw.iterator, layout_x)
        dz_raw = cute.make_tensor(dz_raw.iterator, layout_x)

        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, bidx_z = cute.arch.block_idx()
        
        b = bidx_z
        h = bidx_y % shape_H
        c_block = bidx_y // shape_H

        warp_idx = tidx // 32
        lane_idx = tidx % 32
        L_each = (self.N * self.R) // 32

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        smem_tile_state = storage.sState.get_tensor(smem_layout)
        smem_tile_trap = storage.sTrap.get_tensor(smem_layout)
        smem_tile_grad_state = storage.sGradState.get_tensor(smem_layout)
        smem_tile_grad_V = storage.sGradV.get_tensor(smem_layout)

        smem_tile_GradOut = storage.sGradState.get_tensor(layout_GradOut)
        smem_tile_GradC = storage.sState.get_tensor(layout_GradC)
        smem_tile_GradB = storage.sTrap.get_tensor(layout_GradC)
        smem_tile_B_smem = storage.sGradV.get_tensor(layout_B_smem)

        w = lane_idx // 4
        v = lane_idx % 4
        row_idx0 = (self.N // 8) * w
        col_idx0 = 4 * v

        # Temporal weight reuse: load weights into registers once per warp
        W_each = (16 * self.R) // 32
        W_X_val = cute.make_rmem_tensor((W_each,), Float32)
        W_Z_val = cute.make_rmem_tensor((W_each,), Float32)
        W_O_val = cute.make_rmem_tensor((W_each,), Float32)
        for idx in range(W_each):
            global_idx = idx * 32 + lane_idx
            p_idx = global_idx // self.R
            r_idx = global_idx % self.R
            W_X_val[idx] = Float32(W_X[h, c_block * 16 + p_idx, r_idx])
            W_Z_val[idx] = Float32(W_Z[h, c_block * 16 + p_idx, r_idx])
            W_O_val[idx] = Float32(W_O[h, c_block * 16 + p_idx, r_idx])

        D_skip_val = Float32(0.0)
        if lane_idx < 16:
            D_skip_val = Float32(D_skip[h, c_block * 16 + lane_idx])

        # Allocate gradient registers for local parameter accumulation
        dW_X_accum = cute.make_rmem_tensor((4, self.R), Float32)
        dW_Z_accum = cute.make_rmem_tensor((4, self.R), Float32)
        dW_O_accum = cute.make_rmem_tensor((4, self.R), Float32)
        dD_skip_accum = cute.make_rmem_tensor((4,), Float32)

        for c in range(4):
            dD_skip_accum[c] = Float32(0.0)
            for rank in range(self.R):
                dW_X_accum[c, rank] = Float32(0.0)
                dW_Z_accum[c, rank] = Float32(0.0)
                dW_O_accum[c, rank] = Float32(0.0)

        # Sequence tiling loop
        s_start_warp = (bidx_x * self.num_warps + warp_idx) * self.TILE_S

        # Prefetch sequence-dependent inputs for all TILE_S steps if TILE_S > 1
        if const_expr(self.TILE_S > 1):
            x_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            z_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            grad_out_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            
            L_each = (self.N * self.R) // 32
            B_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            C_tile = cute.make_rmem_tensor((self.TILE_S, L_each), Float32)
            
            dt_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            A_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)
            lam_tile = cute.make_rmem_tensor((self.TILE_S,), Float32)

            for s_offset in range(self.TILE_S):
                s_pref = s_start_warp + s_offset
                if s_pref < shape_S:
                    if lane_idx < 16:
                        x_tile[s_offset] = Float32(x_raw[b, s_pref, h, c_block * 16 + lane_idx])
                        z_tile[s_offset] = Float32(z_raw[b, s_pref, h, c_block * 16 + lane_idx])
                        grad_out_tile[s_offset] = Float32(grad_out[b, s_pref, h, c_block * 16 + lane_idx])
                    else:
                        x_tile[s_offset] = Float32(0.0)
                        z_tile[s_offset] = Float32(0.0)
                        grad_out_tile[s_offset] = Float32(0.0)
                    
                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        B_tile[s_offset, idx] = Float32(B_rot[b, s_pref, h, n_idx, r_idx])
                        C_tile[s_offset, idx] = Float32(C_rot[b, s_pref, h, n_idx, r_idx])
                    
                    dt_tile[s_offset] = Float32(dt[b, s_pref, h])
                    A_tile[s_offset] = Float32(A[b, s_pref, h])
                    lam_tile[s_offset] = Float32(val_lam[b, s_pref, h])
                else:
                    x_tile[s_offset] = Float32(0.0)
                    z_tile[s_offset] = Float32(0.0)
                    grad_out_tile[s_offset] = Float32(0.0)
                    for idx in range(L_each):
                        B_tile[s_offset, idx] = Float32(0.0)
                        C_tile[s_offset, idx] = Float32(0.0)
                    dt_tile[s_offset] = Float32(0.0)
                    A_tile[s_offset] = Float32(0.0)
                    lam_tile[s_offset] = Float32(0.0)

        # If using double buffering, prefetch step 0
        if self.buf_size == 2:
            s_step0 = s_start_warp + 0
            if s_step0 < shape_S:
                gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                
                gState = cute.make_tensor((ssm_state[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gTrap = cute.make_tensor((trap_mem[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gGradState = cute.make_tensor((grad_curr_state[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gGradV = cute.make_tensor((grad_curr_V[b, s_step0, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                
                sState_buf0 = smem_tile_state[0, warp_idx, None, None]
                sTrap_buf0 = smem_tile_trap[0, warp_idx, None, None]
                sGradState_buf0 = smem_tile_grad_state[0, warp_idx, None, None]
                sGradV_buf0 = smem_tile_grad_V[0, warp_idx, None, None]
                
                sState_buf0 = cute.make_tensor(sState_buf0.iterator.align(16), sState_buf0.layout)
                sTrap_buf0 = cute.make_tensor(sTrap_buf0.iterator.align(16), sTrap_buf0.layout)
                sGradState_buf0 = cute.make_tensor(sGradState_buf0.iterator.align(16), sGradState_buf0.layout)
                sGradV_buf0 = cute.make_tensor(sGradV_buf0.iterator.align(16), sGradV_buf0.layout)

                gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                
                tSgState = gmem_thr_copy_state.partition_S(gState)
                tSsState = gmem_thr_copy_state.partition_D(sState_buf0)
                cute.copy(gmem_tiled_copy_state, tSgState, tSsState)

                tSgTrap = gmem_thr_copy_state.partition_S(gTrap)
                tSsTrap = gmem_thr_copy_state.partition_D(sTrap_buf0)
                cute.copy(gmem_tiled_copy_state, tSgTrap, tSsTrap)

                tSgGradState = gmem_thr_copy_state.partition_S(gGradState)
                tSsGradState = gmem_thr_copy_state.partition_D(sGradState_buf0)
                cute.copy(gmem_tiled_copy_state, tSgGradState, tSsGradState)

                tSgGradV = gmem_thr_copy_state.partition_S(gGradV)
                tSsGradV = gmem_thr_copy_state.partition_D(sGradV_buf0)
                cute.copy(gmem_tiled_copy_state, tSgGradV, tSsGradV)

                cute.arch.cp_async_commit_group()

        for s_offset in range(self.TILE_S):
            s = s_start_warp + s_offset
            if s < shape_S:
                buf_idx = s_offset % 2 if self.buf_size == 2 else 0

                # 1. Prefetch next step (step + 1)
                if self.buf_size == 2:
                    s_next = s_start_warp + s_offset + 1
                    if s_next < shape_S:
                        gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                        
                        gState_next = cute.make_tensor((ssm_state[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        gTrap_next = cute.make_tensor((trap_mem[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        gGradState_next = cute.make_tensor((grad_curr_state[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        gGradV_next = cute.make_tensor((grad_curr_V[b, s_next, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                        
                        sState_next = smem_tile_state[(s_offset + 1) % 2, warp_idx, None, None]
                        sTrap_next = smem_tile_trap[(s_offset + 1) % 2, warp_idx, None, None]
                        sGradState_next = smem_tile_grad_state[(s_offset + 1) % 2, warp_idx, None, None]
                        sGradV_next = smem_tile_grad_V[(s_offset + 1) % 2, warp_idx, None, None]
                        
                        sState_next = cute.make_tensor(sState_next.iterator.align(16), sState_next.layout)
                        sTrap_next = cute.make_tensor(sTrap_next.iterator.align(16), sTrap_next.layout)
                        sGradState_next = cute.make_tensor(sGradState_next.iterator.align(16), sGradState_next.layout)
                        sGradV_next = cute.make_tensor(sGradV_next.iterator.align(16), sGradV_next.layout)

                        gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                        
                        tSgState_next = gmem_thr_copy_state.partition_S(gState_next)
                        tSsState_next = gmem_thr_copy_state.partition_D(sState_next)
                        cute.copy(gmem_tiled_copy_state, tSgState_next, tSsState_next)

                        tSgTrap_next = gmem_thr_copy_state.partition_S(gTrap_next)
                        tSsTrap_next = gmem_thr_copy_state.partition_D(sTrap_next)
                        cute.copy(gmem_tiled_copy_state, tSgTrap_next, tSsTrap_next)

                        tSgGradState_next = gmem_thr_copy_state.partition_S(gGradState_next)
                        tSsGradState_next = gmem_thr_copy_state.partition_D(sGradState_next)
                        cute.copy(gmem_tiled_copy_state, tSgGradState_next, tSsGradState_next)

                        tSgGradV_next = gmem_thr_copy_state.partition_S(gGradV_next)
                        tSsGradV_next = gmem_thr_copy_state.partition_D(sGradV_next)
                        cute.copy(gmem_tiled_copy_state, tSgGradV_next, tSsGradV_next)

                        cute.arch.cp_async_commit_group()

                # If no double-buffering, do blocking load
                if self.buf_size == 1:
                    gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                    
                    gState = cute.make_tensor((ssm_state[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    gTrap = cute.make_tensor((trap_mem[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    gGradState = cute.make_tensor((grad_curr_state[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    gGradV = cute.make_tensor((grad_curr_V[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                    
                    sState_buf0 = smem_tile_state[0, warp_idx, None, None]
                    sTrap_buf0 = smem_tile_trap[0, warp_idx, None, None]
                    sGradState_buf0 = smem_tile_grad_state[0, warp_idx, None, None]
                    sGradV_buf0 = smem_tile_grad_V[0, warp_idx, None, None]
                    
                    sState_buf0 = cute.make_tensor(sState_buf0.iterator.align(16), sState_buf0.layout)
                    sTrap_buf0 = cute.make_tensor(sTrap_buf0.iterator.align(16), sTrap_buf0.layout)
                    sGradState_buf0 = cute.make_tensor(sGradState_buf0.iterator.align(16), sGradState_buf0.layout)
                    sGradV_buf0 = cute.make_tensor(sGradV_buf0.iterator.align(16), sGradV_buf0.layout)

                    gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(lane_idx)
                    
                    tSgState = gmem_thr_copy_state.partition_S(gState)
                    tSsState = gmem_thr_copy_state.partition_D(sState_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgState, tSsState)

                    tSgTrap = gmem_thr_copy_state.partition_S(gTrap)
                    tSsTrap = gmem_thr_copy_state.partition_D(sTrap_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgTrap, tSsTrap)

                    tSgGradState = gmem_thr_copy_state.partition_S(gGradState)
                    tSsGradState = gmem_thr_copy_state.partition_D(sGradState_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgGradState, tSsGradState)

                    tSgGradV = gmem_thr_copy_state.partition_S(gGradV)
                    tSsGradV = gmem_thr_copy_state.partition_D(sGradV_buf0)
                    cute.copy(gmem_tiled_copy_state, tSgGradV, tSsGradV)

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                else:
                    # Double buffered: wait for current buffer's cp.async
                    s_next = s_start_warp + s_offset + 1
                    if s_next < shape_S:
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)

                cute.arch.sync_warp()

                # Read state, trap, gradState, and gradV from SMEM linearly
                h_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                v_state = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                grad_curr_state_val = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                grad_curr_V_val = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        h_state[r, c] = smem_tile_state[buf_idx, warp_idx, row, col]
                        v_state[r, c] = smem_tile_trap[buf_idx, warp_idx, row, col]
                        grad_curr_state_val[r, c] = smem_tile_grad_state[buf_idx, warp_idx, row, col]
                        grad_curr_V_val[r, c] = smem_tile_grad_V[buf_idx, warp_idx, row, col]

                L_each = (self.N * self.R) // 32
                B_val = cute.make_rmem_tensor((L_each,), Float32)
                C_val = cute.make_rmem_tensor((L_each,), Float32)

                # Load inputs from prefetch registers or directly from GMEM
                if const_expr(self.TILE_S > 1):
                    x_val = x_tile[s_offset]
                    z_val = z_tile[s_offset]
                    grad_out_val = grad_out_tile[s_offset]

                    for idx in range(L_each):
                        B_val[idx] = B_tile[s_offset, idx]
                        C_val[idx] = C_tile[s_offset, idx]

                    dt_val = dt_tile[s_offset]
                    A_val = A_tile[s_offset]
                    lam_val = lam_tile[s_offset]
                else:
                    x_val = Float32(0.0)
                    z_val = Float32(0.0)
                    grad_out_val = Float32(0.0)
                    if lane_idx < 16:
                        x_val = Float32(x_raw[b, s, h, c_block * 16 + lane_idx])
                        z_val = Float32(z_raw[b, s, h, c_block * 16 + lane_idx])
                        grad_out_val = Float32(grad_out[b, s, h, c_block * 16 + lane_idx])

                    for idx in range(L_each):
                        global_idx = idx * 32 + lane_idx
                        n_idx = global_idx // self.R
                        r_idx = global_idx % self.R
                        B_val[idx] = Float32(B_rot[b, s, h, n_idx, r_idx])
                        C_val[idx] = Float32(C_rot[b, s, h, n_idx, r_idx])

                    dt_val = Float32(dt[b, s, h])
                    A_val = Float32(A[b, s, h])
                    lam_val = Float32(val_lam[b, s, h])

                # Extract linear column values for inputs/weights
                x_local = cute.make_rmem_tensor((4,), Float32)
                z_local = cute.make_rmem_tensor((4,), Float32)
                D_skip_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    col = col_idx0 + c
                    src_lane = col
                    x_local[c] = cute.arch.shuffle_sync(x_val, src_lane)
                    z_local[c] = cute.arch.shuffle_sync(z_val, src_lane)
                    D_skip_local[c] = cute.arch.shuffle_sync(D_skip_val, src_lane)

                B_local = cute.make_rmem_tensor((self.N // 8, self.R), Float32)
                C_local = cute.make_rmem_tensor((self.N // 8, self.R), Float32)
                for r in range(self.N // 8):
                    for rank in range(self.R):
                        global_idx = (row_idx0 + r) * self.R + rank
                        src_lane = global_idx % 32
                        reg_idx = global_idx // 32
                        val_B = Float32(0.0)
                        val_C = Float32(0.0)
                        for l in range(L_each):
                            shfl_B = cute.arch.shuffle_sync(B_val[l], src_lane)
                            shfl_C = cute.arch.shuffle_sync(C_val[l], src_lane)
                            if reg_idx == l:
                                val_B = shfl_B
                                val_C = shfl_C
                        B_local[r, rank] = val_B
                        C_local[r, rank] = val_C

                W_X_local = cute.make_rmem_tensor((4, self.R), Float32)
                W_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
                W_O_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    col = col_idx0 + c
                    for rank in range(self.R):
                        global_idx = col * self.R + rank
                        src_lane = global_idx % 32
                        reg_idx = global_idx // 32
                        val_WX = Float32(0.0)
                        val_WZ = Float32(0.0)
                        val_WO = Float32(0.0)
                        for l in range(W_each):
                            shfl_WX = cute.arch.shuffle_sync(W_X_val[l], src_lane)
                            shfl_WZ = cute.arch.shuffle_sync(W_Z_val[l], src_lane)
                            shfl_WO = cute.arch.shuffle_sync(W_O_val[l], src_lane)
                            if reg_idx == l:
                                val_WX = shfl_WX
                                val_WZ = shfl_WZ
                                val_WO = shfl_WO
                        W_X_local[c, rank] = val_WX
                        W_Z_local[c, rank] = val_WZ
                        W_O_local[c, rank] = val_WO

                # Core backward math (completely linear!)
                X_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        X_t_local[c, rank] = x_local[c] * W_X_local[c, rank]

                curr_V_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        val = Float32(0.0)
                        for rank in range(self.R):
                            val += B_local[r, rank] * X_t_local[c, rank]
                        curr_V_local[r, c] = val

                alpha_val = cute.math.exp(dt_val * A_val, fastmath=True)
                beta_val = (1.0 - lam_val) * dt_val * alpha_val
                gamma_val = lam_val * dt_val

                curr_state_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        curr_state_local[r, c] = alpha_val * h_state[r, c] + beta_val * v_state[r, c] + gamma_val * curr_V_local[r, c]

                Y_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        val = Float32(0.0)
                        for r in range(self.N // 8):
                            val += curr_state_local[r, c] * C_local[r, rank]
                        val = val + cute.arch.shuffle_sync(val, lane_idx ^ 4)
                        val = val + cute.arch.shuffle_sync(val, lane_idx ^ 8)
                        val = val + cute.arch.shuffle_sync(val, lane_idx ^ 16)
                        Y_t_local[c, rank] = val

                Y_skip_local = cute.make_rmem_tensor((4, self.R), Float32)
                Z_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                out_gated_local = cute.make_rmem_tensor((4, self.R), Float32)
                silu_Z_local = cute.make_rmem_tensor((4, self.R), Float32)

                for c in range(4):
                    for rank in range(self.R):
                        Y_skip_local[c, rank] = Y_t_local[c, rank] + D_skip_local[c] * X_t_local[c, rank]
                        Z_t_local[c, rank] = z_local[c] * W_Z_local[c, rank]
                        silu_Z = Z_t_local[c, rank] / (1.0 + cute.math.exp(-Z_t_local[c, rank], fastmath=True))
                        silu_Z_local[c, rank] = silu_Z
                        out_gated_local[c, rank] = Y_skip_local[c, rank] * silu_Z

                grad_out_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    col = col_idx0 + c
                    src_lane = col
                    grad_out_local[c] = cute.arch.shuffle_sync(grad_out_val, src_lane)

                dout_gated_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dout_gated_local[c, rank] = grad_out_local[c] * W_O_local[c, rank]

                dW_O_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dW_O_local[c, rank] = out_gated_local[c, rank] * grad_out_local[c]

                sig_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
                dsilu_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
                dZ_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                dY_skip_local = cute.make_rmem_tensor((4, self.R), Float32)

                for c in range(4):
                    for rank in range(self.R):
                        sig_Z = 1.0 / (1.0 + cute.math.exp(-Z_t_local[c, rank], fastmath=True))
                        sig_Z_local[c, rank] = sig_Z
                        dsilu_Z = sig_Z * (1.0 + Z_t_local[c, rank] * (1.0 - sig_Z))
                        dsilu_Z_local[c, rank] = dsilu_Z
                        dY_skip_local[c, rank] = dout_gated_local[c, rank] * silu_Z_local[c, rank]
                        dZ_t_local[c, rank] = dout_gated_local[c, rank] * Y_skip_local[c, rank] * dsilu_Z

                dz_raw_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    val = Float32(0.0)
                    for rank in range(self.R):
                        val += dZ_t_local[c, rank] * W_Z_local[c, rank]
                    dz_raw_local[c] = val

                dD_skip_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    val = Float32(0.0)
                    for rank in range(self.R):
                        val += dY_skip_local[c, rank] * X_t_local[c, rank]
                    dD_skip_local[c] = val

                dX_t_skip_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dX_t_skip_local[c, rank] = dY_skip_local[c, rank] * D_skip_local[c]

                dcurr_state_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        val = Float32(0.0)
                        for rank in range(self.R):
                            val += C_local[r, rank] * dY_skip_local[c, rank]
                        dcurr_state_local[r, c] = val + grad_curr_state_val[r, c]

                # Initialize padded columns of current grad_state buffer to 0
                for r_pad in range(8 - self.R):
                    for c in range(4):
                        row = col_idx0 + c
                        smem_tile_grad_state[buf_idx, warp_idx, row, self.R + r_pad] = Float32(0.0)

                # Write dY_skip_local to current grad_state buffer
                for c in range(4):
                    row = col_idx0 + c
                    for rank in range(self.R):
                        smem_tile_grad_state[buf_idx, warp_idx, row, rank] = dY_skip_local[c, rank]

                # Store reconstructed curr_state back to SMEM so GEMM can read it
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        smem_tile_state[buf_idx, warp_idx, row, col] = curr_state_local[r, c]
                cute.arch.sync_warp()

                sState_slice = smem_tile_state[buf_idx, warp_idx, None, None]
                sState_gemm = cute.make_tensor(sState_slice.iterator, cute.make_layout((self.N, 16), stride=(20, 1)))
                sGradOut_slice = smem_tile_grad_state[buf_idx, warp_idx, None, None]
                sGradOut_trans = cute.make_tensor(sGradOut_slice.iterator, cute.make_layout((8, 16), stride=(1, 20)))
                sGradC_slice = smem_tile_GradC[warp_idx, None, None]

                # Partition
                tAsA_dC = tiled_mma.get_slice(lane_idx).partition_A(sState_gemm)
                tBsB_dC = tiled_mma.get_slice(lane_idx).partition_B(sGradOut_trans)
                tCsC_dC = tiled_mma.get_slice(lane_idx).partition_C(sGradC_slice)

                # Fragment allocations
                rA_dC = tiled_mma.make_fragment_A(tAsA_dC)
                rB_dC = tiled_mma.make_fragment_B(tBsB_dC)
                rC_dC = tiled_mma.make_fragment_C(tCsC_dC)

                # Load and cast using vectorized copies
                rA_dC_f32 = cute.make_fragment_like(tAsA_dC)
                rB_dC_f32 = cute.make_fragment_like(tBsB_dC)
                cute.autovec_copy(tAsA_dC, rA_dC_f32)
                cute.autovec_copy(tBsB_dC, rB_dC_f32)
                for i in range(cute.size(rA_dC)):
                    rA_dC[i] = BFloat16(rA_dC_f32[i])
                for i in range(cute.size(rB_dC)):
                    rB_dC[i] = BFloat16(rB_dC_f32[i])

                # GEMM
                rC_dC.fill(0.0)
                cute.gemm(tiled_mma, rC_dC, rA_dC, rB_dC, rC_dC)

                # Write directly to GMEM: use direct writes if shape_P == 16 (no c_block overlap), otherwise atomic_add
                row_base_bwd = lane_idx // 4
                col_base_bwd = (lane_idx % 4) * 2
                offset_base = (b * shape_S * shape_H + s * shape_H + h) * (self.N * self.R)
                for i in range(self.N // 16):
                    row1 = row_base_bwd + i * 16
                    row2 = row_base_bwd + i * 16 + 8
                    col1 = col_base_bwd
                    col2 = col_base_bwd + 1
                    
                    if shape_P == 16:
                        if row1 < self.N:
                            if col1 < self.R:
                                dC_rot[b, s, h, row1, col1] = rC_dC[(0, 0), i, 0]
                            if col2 < self.R:
                                dC_rot[b, s, h, row1, col2] = rC_dC[(1, 0), i, 0]
                        if row2 < self.N:
                            if col1 < self.R:
                                dC_rot[b, s, h, row2, col1] = rC_dC[(0, 1), i, 0]
                            if col2 < self.R:
                                dC_rot[b, s, h, row2, col2] = rC_dC[(1, 1), i, 0]
                    else:
                        if row1 < self.N:
                            if col1 < self.R:
                                cute.arch.atomic_add(dC_rot.iterator + (offset_base + row1 * self.R + col1), rC_dC[(0, 0), i, 0])
                            if col2 < self.R:
                                cute.arch.atomic_add(dC_rot.iterator + (offset_base + row1 * self.R + col2), rC_dC[(1, 0), i, 0])
                        if row2 < self.N:
                            if col1 < self.R:
                                cute.arch.atomic_add(dC_rot.iterator + (offset_base + row2 * self.R + col1), rC_dC[(0, 1), i, 0])
                            if col2 < self.R:
                                cute.arch.atomic_add(dC_rot.iterator + (offset_base + row2 * self.R + col2), rC_dC[(1, 1), i, 0])

                dssm_state_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                dtrap_mem_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                dcurr_V_local = cute.make_rmem_tensor((self.N // 8, 4), Float32)
                for r in range(self.N // 8):
                    for c in range(4):
                        dssm_state_local[r, c] = alpha_val * dcurr_state_local[r, c]
                        dtrap_mem_local[r, c] = beta_val * dcurr_state_local[r, c]
                        dcurr_V_local[r, c] = gamma_val * dcurr_state_local[r, c] + grad_curr_V_val[r, c]

                dalpha_local = Float32(0.0)
                dbeta_local = Float32(0.0)
                dgamma_local = Float32(0.0)
                for r in range(self.N // 8):
                    for c in range(4):
                        dalpha_local += dcurr_state_local[r, c] * h_state[r, c]
                        dbeta_local += dcurr_state_local[r, c] * v_state[r, c]
                        dgamma_local += dcurr_state_local[r, c] * curr_V_local[r, c]

                for i in range(5):
                    step = 1 << i
                    dalpha_local = dalpha_local + cute.arch.shuffle_sync(dalpha_local, lane_idx ^ step)
                    dbeta_local = dbeta_local + cute.arch.shuffle_sync(dbeta_local, lane_idx ^ step)
                    dgamma_local = dgamma_local + cute.arch.shuffle_sync(dgamma_local, lane_idx ^ step)


                # Write dcurr_V_local to SMEM
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        smem_tile_grad_V[buf_idx, warp_idx, row, col] = dcurr_V_local[r, c]
                cute.arch.sync_warp()

                # 1. Compute dB_rot_local using GEMM
                # Initialize padded columns of current grad_state buffer to 0
                for r_pad in range(8 - self.R):
                    for c in range(4):
                        row = col_idx0 + c
                        smem_tile_grad_state[buf_idx, warp_idx, row, self.R + r_pad] = Float32(0.0)

                # Write X_t_local to current grad_state buffer
                for c in range(4):
                    row = col_idx0 + c
                    for rank in range(self.R):
                        smem_tile_grad_state[buf_idx, warp_idx, row, rank] = X_t_local[c, rank]
                cute.arch.sync_warp()

                sdcurrV_slice = smem_tile_grad_V[buf_idx, warp_idx, None, None]
                sdcurrV_gemm = cute.make_tensor(sdcurrV_slice.iterator, cute.make_layout((self.N, 16), stride=(20, 1)))
                sGradB_slice = smem_tile_GradB[warp_idx, None, None]
                sGradOut_slice = smem_tile_grad_state[buf_idx, warp_idx, None, None]
                sGradOut_trans = cute.make_tensor(sGradOut_slice.iterator, cute.make_layout((8, 16), stride=(1, 20)))

                # Partition
                tAsA_dB = tiled_mma.get_slice(lane_idx).partition_A(sdcurrV_gemm)
                tBsB_dB = tiled_mma.get_slice(lane_idx).partition_B(sGradOut_trans)
                tCsC_dB = tiled_mma.get_slice(lane_idx).partition_C(sGradB_slice)

                # Fragment allocations
                rA_dB = tiled_mma.make_fragment_A(tAsA_dB)
                rB_dB = tiled_mma.make_fragment_B(tBsB_dB)
                rC_dB = tiled_mma.make_fragment_C(tCsC_dB)

                # Load and cast using vectorized copies
                rA_dB_f32 = cute.make_fragment_like(tAsA_dB)
                rB_dB_f32 = cute.make_fragment_like(tBsB_dB)
                cute.autovec_copy(tAsA_dB, rA_dB_f32)
                cute.autovec_copy(tBsB_dB, rB_dB_f32)
                for i in range(cute.size(rA_dB)):
                    rA_dB[i] = BFloat16(rA_dB_f32[i])
                for i in range(cute.size(rB_dB)):
                    rB_dB[i] = BFloat16(rB_dB_f32[i])

                # GEMM
                rC_dB.fill(0.0)
                cute.gemm(tiled_mma, rC_dB, rA_dB, rB_dB, rC_dB)

                # Write directly to GMEM: use direct writes if shape_P == 16 (no c_block overlap), otherwise atomic_add
                for i in range(self.N // 16):
                    row1 = row_base_bwd + i * 16
                    row2 = row_base_bwd + i * 16 + 8
                    col1 = col_base_bwd
                    col2 = col_base_bwd + 1
                    
                    if shape_P == 16:
                        if row1 < self.N:
                            if col1 < self.R:
                                dB_rot[b, s, h, row1, col1] = rC_dB[(0, 0), i, 0]
                            if col2 < self.R:
                                dB_rot[b, s, h, row1, col2] = rC_dB[(1, 0), i, 0]
                        if row2 < self.N:
                            if col1 < self.R:
                                dB_rot[b, s, h, row2, col1] = rC_dB[(0, 1), i, 0]
                            if col2 < self.R:
                                dB_rot[b, s, h, row2, col2] = rC_dB[(1, 1), i, 0]
                    else:
                        if row1 < self.N:
                            if col1 < self.R:
                                cute.arch.atomic_add(dB_rot.iterator + (offset_base + row1 * self.R + col1), rC_dB[(0, 0), i, 0])
                            if col2 < self.R:
                                cute.arch.atomic_add(dB_rot.iterator + (offset_base + row1 * self.R + col2), rC_dB[(1, 0), i, 0])
                        if row2 < self.N:
                            if col1 < self.R:
                                cute.arch.atomic_add(dB_rot.iterator + (offset_base + row2 * self.R + col1), rC_dB[(0, 1), i, 0])
                            if col2 < self.R:
                                cute.arch.atomic_add(dB_rot.iterator + (offset_base + row2 * self.R + col2), rC_dB[(1, 1), i, 0])

                # 2. Compute dX_t_V_local using GEMM
                # Initialize padded columns to 0 in current trap buffer (safely reuse unused sTrap)
                for r_pad in range(8 - self.R):
                    for r in range(self.N // 8):
                        row = row_idx0 + r
                        smem_tile_trap[buf_idx, warp_idx, row, self.R + r_pad] = Float32(0.0)

                # Write B_local to current trap buffer (safely reuse unused sTrap)
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for rank in range(self.R):
                        smem_tile_trap[buf_idx, warp_idx, row, rank] = B_local[r, rank]
                cute.arch.sync_warp()

                sdcurrV_trans = cute.make_tensor(sdcurrV_slice.iterator, cute.make_layout((16, self.N), stride=(1, 20)))
                sB_slice = smem_tile_trap[buf_idx, warp_idx, None, None]
                sB_smem_trans = cute.make_tensor(sB_slice.iterator, cute.make_layout((8, self.N), stride=(1, 20)))
                sGradOut_slice_gemm = cute.make_tensor(sGradOut_slice.iterator, cute.make_layout((16, 8), stride=(20, 1)))

                # Partition
                tAsA_dX = tiled_mma.get_slice(lane_idx).partition_A(sdcurrV_trans)
                tBsB_dX = tiled_mma.get_slice(lane_idx).partition_B(sB_smem_trans)
                tCsC_dX = tiled_mma.get_slice(lane_idx).partition_C(sGradOut_slice_gemm)

                # Fragment allocations
                rA_dX = tiled_mma.make_fragment_A(tAsA_dX)
                rB_dX = tiled_mma.make_fragment_B(tBsB_dX)
                rC_dX = tiled_mma.make_fragment_C(tCsC_dX)

                # Load and cast using vectorized copies
                rA_dX_f32 = cute.make_fragment_like(tAsA_dX)
                rB_dX_f32 = cute.make_fragment_like(tBsB_dX)
                cute.autovec_copy(tAsA_dX, rA_dX_f32)
                cute.autovec_copy(tBsB_dX, rB_dX_f32)
                for i in range(cute.size(rA_dX)):
                    rA_dX[i] = BFloat16(rA_dX_f32[i])
                for i in range(cute.size(rB_dX)):
                    rB_dX[i] = BFloat16(rB_dX_f32[i])

                # GEMM
                rC_dX.fill(0.0)
                cute.gemm(tiled_mma, rC_dX, rA_dX, rB_dX, rC_dX)

                # Copy back to shared memory (C)
                sGradOut_slice_gemm[row_base_bwd, col_base_bwd] = rC_dX[(0, 0), 0, 0]
                sGradOut_slice_gemm[row_base_bwd, col_base_bwd + 1] = rC_dX[(1, 0), 0, 0]
                sGradOut_slice_gemm[row_base_bwd + 8, col_base_bwd] = rC_dX[(0, 1), 0, 0]
                sGradOut_slice_gemm[row_base_bwd + 8, col_base_bwd + 1] = rC_dX[(1, 1), 0, 0]
                cute.arch.sync_warp()

                # Read back to registers
                dX_t_V_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dX_t_V_local[c, rank] = sGradOut_slice_gemm[col_idx0 + c, rank]

                dX_t_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dX_t_local[c, rank] = dX_t_skip_local[c, rank] + dX_t_V_local[c, rank]

                dx_raw_local = cute.make_rmem_tensor((4,), Float32)
                for c in range(4):
                    val = Float32(0.0)
                    for rank in range(self.R):
                        val += dX_t_local[c, rank] * W_X_local[c, rank]
                    dx_raw_local[c] = val

                dW_X_local = cute.make_rmem_tensor((4, self.R), Float32)
                dW_Z_local = cute.make_rmem_tensor((4, self.R), Float32)
                for c in range(4):
                    for rank in range(self.R):
                        dW_X_local[c, rank] = x_local[c] * dX_t_local[c, rank]
                        dW_Z_local[c, rank] = z_local[c] * dZ_t_local[c, rank]

                # Accumulate local gradients into warp-level registers (perfectly linear!)
                for c in range(4):
                    dD_skip_accum[c] += dD_skip_local[c]
                    for rank in range(self.R):
                        dW_X_accum[c, rank] += dW_X_local[c, rank]
                        dW_Z_accum[c, rank] += dW_Z_local[c, rank]
                        dW_O_accum[c, rank] += dW_O_local[c, rank]

                # Store sequence-token dependent gradients to GMEM linearly
                if lane_idx < 4:
                    for c in range(4):
                        p = 4 * lane_idx + c
                        global_p = c_block * 16 + p
                        dx_raw[b, s, h, global_p] = dx_raw_local[c]
                        dz_raw[b, s, h, global_p] = dz_raw_local[c]

                # dB_rot and dC_rot are written directly from register fragments

                if lane_idx == 0:
                    ddt_val = dalpha_local * A_val * alpha_val + dbeta_local * (1.0 - lam_val) * alpha_val * (1.0 + dt_val * A_val) + dgamma_local * lam_val
                    dA_val = dalpha_local * dt_val * alpha_val + dbeta_local * (1.0 - lam_val) * dt_val * dt_val * alpha_val
                    dlam_val = -dbeta_local * dt_val * alpha_val + dgamma_local * dt_val

                    offset_scalar = b * shape_S * shape_H + s * shape_H + h
                    if shape_P == 16:
                        dt_grad[b, s, h] = ddt_val
                        A_grad[b, s, h] = dA_val
                        lam_grad[b, s, h] = dlam_val
                    else:
                        cute.arch.atomic_add(dt_grad.iterator + offset_scalar, ddt_val)
                        cute.arch.atomic_add(A_grad.iterator + offset_scalar, dA_val)
                        cute.arch.atomic_add(lam_grad.iterator + offset_scalar, dlam_val)

                # Write dssm_state and dtrap_mem linearly to SMEM
                for r in range(self.N // 8):
                    row = row_idx0 + r
                    for c in range(4):
                        col = col_idx0 + c
                        smem_tile_state[buf_idx, warp_idx, row, col] = dssm_state_local[r, c]
                        smem_tile_trap[buf_idx, warp_idx, row, col] = dtrap_mem_local[r, c]

                cute.arch.sync_warp()

                gState_layout = cute.make_layout((self.N, 16), stride=(shape_P, 1))
                gDssm = cute.make_tensor((dssm_state[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)
                gDtrap = cute.make_tensor((dtrap_mem[b, s, h, None, None].iterator + c_block * 16).align(16), gState_layout)

                sState_buf = smem_tile_state[buf_idx, warp_idx, None, None]
                sState_buf = cute.make_tensor(sState_buf.iterator.align(16), sState_buf.layout)
                sTrap_buf = smem_tile_trap[buf_idx, warp_idx, None, None]
                sTrap_buf = cute.make_tensor(sTrap_buf.iterator.align(16), sTrap_buf.layout)

                gmem_thr_copy_store = gmem_tiled_copy_store.get_slice(lane_idx)
                tSgDssm = gmem_thr_copy_store.partition_D(gDssm)
                tSsState_s2g = gmem_thr_copy_store.partition_S(sState_buf)
                cute.copy(gmem_tiled_copy_store, tSsState_s2g, tSgDssm)

                tSgDtrap = gmem_thr_copy_store.partition_D(gDtrap)
                tSsTrap_s2g = gmem_thr_copy_store.partition_S(sTrap_buf)
                cute.copy(gmem_tiled_copy_store, tSsTrap_s2g, tSgDtrap)

        # Write accumulated parameter gradients to GMEM linearly once at the end of the warp
        if s_start_warp < shape_S:
            if lane_idx < 4:
                for c in range(4):
                    p = 4 * lane_idx + c
                    global_p = c_block * 16 + p
                    
                    offset_dD = h * shape_P + global_p
                    cute.arch.atomic_add(dD_skip.iterator + offset_dD, dD_skip_accum[c])
                    
                    offset_param = h * shape_P * self.R + global_p * self.R
                    for rank in range(self.R):
                        cute.arch.atomic_add(dW_X.iterator + (offset_param + rank), dW_X_accum[c, rank])
                        cute.arch.atomic_add(dW_Z.iterator + (offset_param + rank), dW_Z_accum[c, rank])
                        cute.arch.atomic_add(dW_O.iterator + (offset_param + rank), dW_O_accum[c, rank])




_fwd_compile_cache = {}
_bwd_compile_cache = {}

def get_cute_mamba3_fwd_kernel(num_warps=4, P_val=16, N_val=16, R_val=2, TILE_S=None, buf_size=None, layout_option="coprime", maxrregcount=255, copy_layout_shape=(8, 4), mma_shape=(16, 8, 16)):
    if TILE_S is None:
        TILE_S = 8 if (N_val <= 16 or (N_val == 32 and R_val == 2)) else 1
    key = (num_warps, P_val, N_val, R_val, TILE_S, buf_size, layout_option, maxrregcount, copy_layout_shape, mma_shape)
    if key not in _fwd_compile_cache:
        os.environ["CUTE_DSL_ARCH"] = "sm_120a"
        os.environ["QUACK_ARCH"] = "sm_120a"
        
        B_sym = cute.sym_int()
        S_sym = cute.sym_int()
        H_sym = cute.sym_int()
        
        fwd_op = CuteMamba3Fwd(
            num_warps=num_warps, N=N_val, R=R_val, TILE_S=TILE_S, buf_size=buf_size, layout_option=layout_option,
            copy_layout_shape=copy_layout_shape, mma_shape=mma_shape
        )
        x_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        z_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        B_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        C_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        dt_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        A_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        lam_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        ssm_state_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        trap_mem_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        W_X_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        W_Z_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        W_O_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        D_skip_fake = make_fake_tensor(Float32, (H_sym, P_val), 16)
        
        out_fake = make_fake_tensor(BFloat16, (B_sym, S_sym, H_sym * P_val), 8)
        curr_state_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        curr_V_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_fwd = cute.compile(
            fwd_op,
            x_raw_fake, z_raw_fake, B_rot_fake, C_rot_fake, dt_fake, A_fake, lam_fake, ssm_state_fake, trap_mem_fake,
            W_X_fake, W_Z_fake, W_O_fake, D_skip_fake, out_fake, curr_state_fake, curr_V_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --ptxas-options=-maxrregcount={maxrregcount}",
        )
        _fwd_compile_cache[key] = compiled_fwd
    return _fwd_compile_cache[key]
 
def get_cute_mamba3_bwd_kernel(num_warps=4, P_val=16, N_val=16, R_val=2, TILE_S=None, buf_size=None, layout_option="coprime", maxrregcount=255, copy_layout_shape=(8, 4), mma_shape=(16, 8, 16)):
    if TILE_S is None:
        TILE_S = 8 if (N_val <= 16 or (N_val == 32 and R_val == 2)) else 1
    key = (num_warps, P_val, N_val, R_val, TILE_S, buf_size, layout_option, maxrregcount, copy_layout_shape, mma_shape)
    if key not in _bwd_compile_cache:
        os.environ["CUTE_DSL_ARCH"] = "sm_120a"
        os.environ["QUACK_ARCH"] = "sm_120a"
        
        B_sym = cute.sym_int()
        S_sym = cute.sym_int()
        H_sym = cute.sym_int()
        
        bwd_op = CuteMamba3Bwd(
            num_warps=num_warps, N=N_val, R=R_val, TILE_S=TILE_S, buf_size=buf_size, layout_option=layout_option,
            copy_layout_shape=copy_layout_shape, mma_shape=mma_shape
        )
        grad_out_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        x_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        z_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        B_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        C_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        dt_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        A_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        lam_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        ssm_state_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        trap_mem_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        W_X_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        W_Z_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        W_O_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        D_skip_fake = make_fake_tensor(Float32, (H_sym, P_val), 16)
        
        grad_curr_state_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        grad_curr_V_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        
        dx_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        dz_raw_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym * P_val), 16)
        dB_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        dC_rot_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, R_val), 2)
        dt_grad_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        A_grad_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        lam_grad_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym), 1)
        dssm_state_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        dtrap_mem_fake = make_fake_tensor(Float32, (B_sym, S_sym, H_sym, N_val, P_val), 16)
        dW_X_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        dW_Z_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        dW_O_fake = make_fake_tensor(Float32, (H_sym, P_val, R_val), 2)
        dD_skip_fake = make_fake_tensor(Float32, (H_sym, P_val), 16)
        
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        
        compiled_bwd = cute.compile(
            bwd_op,
            grad_out_fake, x_raw_fake, z_raw_fake, B_rot_fake, C_rot_fake, dt_fake, A_fake, lam_fake, ssm_state_fake, trap_mem_fake,
            W_X_fake, W_Z_fake, W_O_fake, D_skip_fake,
            grad_curr_state_fake, grad_curr_V_fake,
            dx_raw_fake, dz_raw_fake, dB_rot_fake, dC_rot_fake, dt_grad_fake, A_grad_fake, lam_grad_fake,
            dssm_state_fake, dtrap_mem_fake, dW_X_fake, dW_Z_fake, dW_O_fake, dD_skip_fake,
            fake_stream,
            options=f"--enable-tvm-ffi --ptxas-options=-maxrregcount={maxrregcount}",
        )
        _bwd_compile_cache[key] = compiled_bwd
    return _bwd_compile_cache[key]


def cute_mamba3_fallback(
    x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
    W_X, W_Z, W_O, D_skip
):
    orig_dtype = x_raw.dtype
    
    H_val = W_X.shape[0]
    P_val = W_X.shape[1]
    x_raw = x_raw.view(x_raw.shape[0], x_raw.shape[1], H_val, P_val)
    z_raw = z_raw.view(z_raw.shape[0], z_raw.shape[1], H_val, P_val)

    x_32 = x_raw.to(torch.float32)
    z_32 = z_raw.to(torch.float32)
    B_32 = B_rot.to(torch.float32)
    C_32 = C_rot.to(torch.float32)
    dt_32 = dt.to(torch.float32)
    A_32 = A.to(torch.float32)
    lam_32 = lam.to(torch.float32)
    ssm_state_32 = ssm_state.to(torch.float32)
    trap_mem_32 = trap_mem.to(torch.float32)
    W_X_32 = W_X.to(torch.float32)
    W_Z_32 = W_Z.to(torch.float32)
    W_O_32 = W_O.to(torch.float32)
    D_skip_32 = D_skip.to(torch.float32)
    
    B, S, H, P = x_32.shape
    N = B_32.shape[3]
    R = B_32.shape[4]
    
    # Factored curr_V: [B, S, H, N, R] @ [1, 1, H, R, P] -> [B, S, H, N, P]
    W_X_T = W_X_32.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    curr_V = torch.matmul(B_32, W_X_T) * x_32.unsqueeze(-2)
    
    alpha_val = torch.exp(dt_32 * A_32)
    beta_val = (1.0 - lam_32) * dt_32 * alpha_val
    gamma_val = lam_32 * dt_32
    
    alpha = alpha_val.reshape(B, S, H, 1, 1)
    beta = beta_val.reshape(B, S, H, 1, 1)
    gamma = gamma_val.reshape(B, S, H, 1, 1)
    
    curr_state = (alpha * ssm_state_32) + (beta * trap_mem_32) + (gamma * curr_V)
    Y_t = torch.matmul(curr_state.transpose(-1, -2), C_32)
    
    # Avoid materializing X_t by scaling x_32 by D_skip_32 in 4D first
    DX_W = (x_32 * D_skip_32.unsqueeze(0).unsqueeze(0)).unsqueeze(-1) * W_X_32.unsqueeze(0).unsqueeze(0)
    Y_skip = Y_t + DX_W
    
    Z_t = z_32.unsqueeze(-1) * W_Z_32.unsqueeze(0).unsqueeze(0)
    out_gated = Y_skip * torch.nn.functional.silu(Z_t)
    
    W_O_5d = W_O_32.unsqueeze(0).unsqueeze(0)
    O_t_prime = (out_gated * W_O_5d).sum(dim=-1)
    
    O_t_prime = O_t_prime.view(B, S, H * P)
        
    return O_t_prime.to(torch.bfloat16), curr_state, curr_V


def cute_mamba3_backward_fallback(
    grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
    W_X, W_Z, W_O, D_skip, grad_curr_state, grad_curr_V
):
    with torch.enable_grad():
        x_raw_c = x_raw.detach().requires_grad_(True)
        z_raw_c = z_raw.detach().requires_grad_(True)
        B_rot_c = B_rot.detach().requires_grad_(True)
        C_rot_c = C_rot.detach().requires_grad_(True)
        dt_c = dt.detach().requires_grad_(True)
        A_c = A.detach().requires_grad_(True)
        lam_c = lam.detach().requires_grad_(True)
        ssm_state_c = ssm_state.detach().requires_grad_(True)
        trap_mem_c = trap_mem.detach().requires_grad_(True)
        W_X_c = W_X.detach().requires_grad_(True)
        W_Z_c = W_Z.detach().requires_grad_(True)
        W_O_c = W_O.detach().requires_grad_(True)
        D_skip_c = D_skip.detach().requires_grad_(True)
        
        O_t_prime, curr_state, curr_V = cute_mamba3_fallback(
            x_raw_c, z_raw_c, B_rot_c, C_rot_c, dt_c, A_c, lam_c, ssm_state_c, trap_mem_c,
            W_X_c, W_Z_c, W_O_c, D_skip_c
        )
        
        outputs = []
        grad_tensors = []
        
        if grad_out is not None:
            outputs.append(O_t_prime)
            grad_tensors.append(grad_out)
        else:
            outputs.append(O_t_prime)
            grad_tensors.append(torch.zeros_like(O_t_prime))
            
        if grad_curr_state is not None:
            outputs.append(curr_state)
            grad_tensors.append(grad_curr_state)
        else:
            outputs.append(curr_state)
            grad_tensors.append(torch.zeros_like(curr_state))
            
        if grad_curr_V is not None:
            outputs.append(curr_V)
            grad_tensors.append(grad_curr_V)
        else:
            outputs.append(curr_V)
            grad_tensors.append(torch.zeros_like(curr_V))
            
        inputs = [
            x_raw_c, z_raw_c, B_rot_c, C_rot_c, dt_c, A_c, lam_c, ssm_state_c, trap_mem_c,
            W_X_c, W_Z_c, W_O_c, D_skip_c
        ]
        
        grads = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_tensors,
            retain_graph=False,
            allow_unused=True
        )
        
        res = []
        for g, inp in zip(grads, inputs):
            if g is None:
                res.append(torch.zeros_like(inp))
            else:
                res.append(g)
                
        return tuple(res)


def ensure_fp32_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t

# Custom operator registration
@torch.library.custom_op("advantage_first::cute_mamba3_custom_op", mutates_args=())
def _cute_mamba3_custom_op(
    x_raw: torch.Tensor,
    z_raw: torch.Tensor,
    B_rot: torch.Tensor,
    C_rot: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    ssm_state: torch.Tensor,
    trap_mem: torch.Tensor,
    W_X: torch.Tensor,
    W_Z: torch.Tensor,
    W_O: torch.Tensor,
    D_skip: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_raw.device
    B_val, S_val, _ = x_raw.shape
    H_val = W_X.shape[0]
    P_val = W_X.shape[1]
    N_val = B_rot.shape[3]
    R_val = B_rot.shape[4]
    
    # Cast tensors to float32 if needed
    x_raw_32 = ensure_fp32_contiguous(x_raw)
    z_raw_32 = ensure_fp32_contiguous(z_raw)
    B_rot_32 = ensure_fp32_contiguous(B_rot)
    C_rot_32 = ensure_fp32_contiguous(C_rot)
    dt_32 = ensure_fp32_contiguous(dt)
    A_32 = ensure_fp32_contiguous(A)
    lam_32 = ensure_fp32_contiguous(lam)
    ssm_state_32 = ensure_fp32_contiguous(ssm_state)
    trap_mem_32 = ensure_fp32_contiguous(trap_mem)
    W_X_32 = ensure_fp32_contiguous(W_X)
    W_Z_32 = ensure_fp32_contiguous(W_Z)
    W_O_32 = ensure_fp32_contiguous(W_O)
    D_skip_32 = ensure_fp32_contiguous(D_skip)

    out = torch.empty((B_val, S_val, H_val * P_val), dtype=torch.bfloat16, device=device)
    curr_state = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)
    curr_V = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)
    
    config = get_autotuned_config(P_val, N_val, R_val, "fwd", str(device))
    copy_layout_shape = config.get("copy_layout_shape", (8, 4))
    if isinstance(copy_layout_shape, list):
        copy_layout_shape = tuple(copy_layout_shape)
    mma_shape = config.get("mma_shape", (16, 8, 16))
    if isinstance(mma_shape, list):
        mma_shape = tuple(mma_shape)
    fwd_kernel = get_cute_mamba3_fwd_kernel(
        num_warps=config["num_warps"],
        P_val=P_val,
        N_val=N_val,
        R_val=R_val,
        TILE_S=config["TILE_S"],
        buf_size=config["buf_size"],
        layout_option=config.get("layout_option", "coprime"),
        maxrregcount=config.get("maxrregcount", 255),
        copy_layout_shape=copy_layout_shape,
        mma_shape=mma_shape
    )
    fwd_kernel(
        x_raw_32, z_raw_32, B_rot_32, C_rot_32, dt_32, A_32, lam_32, ssm_state_32, trap_mem_32,
        W_X_32, W_Z_32, W_O_32, D_skip_32, out, curr_state, curr_V
    )
    return out, curr_state, curr_V

@_cute_mamba3_custom_op.register_fake
def _cute_mamba3_custom_op_fake(
    x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
    W_X, W_Z, W_O, D_skip
):
    B_val, S_val, _ = x_raw.shape
    H_val = W_X.shape[0]
    P_val = W_X.shape[1]
    N_val = B_rot.shape[3]
    out = torch.empty((B_val, S_val, H_val * P_val), dtype=torch.bfloat16, device=x_raw.device)
    curr_state = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=x_raw.device)
    curr_V = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=x_raw.device)
    return out, curr_state, curr_V

def is_fake_tensor(*tensors):
    for t in tensors:
        if t is not None:
            if t.device.type == "meta" or type(t).__name__ == "FakeTensor":
                return True
    return False

def _cute_mamba3_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


@torch.library.custom_op("advantage_first::cute_mamba3_custom_op_backward", mutates_args=())
def _cute_mamba3_custom_op_backward(
    grad_out: Optional[torch.Tensor],
    x_raw: torch.Tensor,
    z_raw: torch.Tensor,
    B_rot: torch.Tensor,
    C_rot: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    ssm_state: torch.Tensor,
    trap_mem: torch.Tensor,
    W_X: torch.Tensor,
    W_Z: torch.Tensor,
    W_O: torch.Tensor,
    D_skip: torch.Tensor,
    grad_curr_state: Optional[torch.Tensor],
    grad_curr_V: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_raw.device
    B_val, S_val, _ = x_raw.shape
    H_val = W_X.shape[0]
    P_val = W_X.shape[1]
    N_val = B_rot.shape[3]
    R_val = B_rot.shape[4]
    
    # Cast tensors to float32
    grad_out_32 = ensure_fp32_contiguous(grad_out) if grad_out is not None else torch.zeros((B_val, S_val, H_val * P_val), dtype=torch.float32, device=device)
    x_raw_32 = ensure_fp32_contiguous(x_raw)
    z_raw_32 = ensure_fp32_contiguous(z_raw)
    B_rot_32 = ensure_fp32_contiguous(B_rot)
    C_rot_32 = ensure_fp32_contiguous(C_rot)
    dt_32 = ensure_fp32_contiguous(dt)
    A_32 = ensure_fp32_contiguous(A)
    lam_32 = ensure_fp32_contiguous(lam)
    ssm_state_32 = ensure_fp32_contiguous(ssm_state)
    trap_mem_32 = ensure_fp32_contiguous(trap_mem)
    W_X_32 = ensure_fp32_contiguous(W_X)
    W_Z_32 = ensure_fp32_contiguous(W_Z)
    W_O_32 = ensure_fp32_contiguous(W_O)
    D_skip_32 = ensure_fp32_contiguous(D_skip)
    
    grad_curr_state_32 = ensure_fp32_contiguous(grad_curr_state) if grad_curr_state is not None else torch.zeros((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)
    grad_curr_V_32 = ensure_fp32_contiguous(grad_curr_V) if grad_curr_V is not None else torch.zeros((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)

    dx_raw = torch.empty((B_val, S_val, H_val * P_val), dtype=torch.float32, device=device)
    dz_raw = torch.empty((B_val, S_val, H_val * P_val), dtype=torch.float32, device=device)
    dB_rot = torch.zeros((B_val, S_val, H_val, N_val, R_val), dtype=torch.float32, device=device)
    dC_rot = torch.zeros((B_val, S_val, H_val, N_val, R_val), dtype=torch.float32, device=device)
    dt_grad = torch.zeros((B_val, S_val, H_val), dtype=torch.float32, device=device)
    A_grad = torch.zeros((B_val, S_val, H_val), dtype=torch.float32, device=device)
    lam_grad = torch.zeros((B_val, S_val, H_val), dtype=torch.float32, device=device)
    dssm_state = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)
    dtrap_mem = torch.empty((B_val, S_val, H_val, N_val, P_val), dtype=torch.float32, device=device)
    
    dW_X = torch.zeros((H_val, P_val, R_val), dtype=torch.float32, device=device)
    dW_Z = torch.zeros((H_val, P_val, R_val), dtype=torch.float32, device=device)
    dW_O = torch.zeros((H_val, P_val, R_val), dtype=torch.float32, device=device)
    dD_skip = torch.zeros((H_val, P_val), dtype=torch.float32, device=device)
    config = get_autotuned_config(P_val, N_val, R_val, "bwd", str(device))
    copy_layout_shape = config.get("copy_layout_shape", (8, 4))
    if isinstance(copy_layout_shape, list):
        copy_layout_shape = tuple(copy_layout_shape)
    mma_shape = config.get("mma_shape", (16, 8, 16))
    if isinstance(mma_shape, list):
        mma_shape = tuple(mma_shape)
    bwd_kernel = get_cute_mamba3_bwd_kernel(
        num_warps=config["num_warps"],
        P_val=P_val,
        N_val=N_val,
        R_val=R_val,
        TILE_S=config["TILE_S"],
        buf_size=config["buf_size"],
        layout_option=config.get("layout_option", "coprime"),
        maxrregcount=config.get("maxrregcount", 255),
        copy_layout_shape=copy_layout_shape,
        mma_shape=mma_shape
    )
    bwd_kernel(
        grad_out_32, x_raw_32, z_raw_32, B_rot_32, C_rot_32, dt_32, A_32, lam_32, ssm_state_32, trap_mem_32,
        W_X_32, W_Z_32, W_O_32, D_skip_32,
        grad_curr_state_32, grad_curr_V_32,
        dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem,
        dW_X, dW_Z, dW_O, dD_skip
    )
    
    return (
        dx_raw,
        dz_raw,
        dB_rot,
        dC_rot,
        dt_grad,
        A_grad,
        lam_grad,
        dssm_state,
        dtrap_mem,
        dW_X,
        dW_Z,
        dW_O,
        dD_skip
    )

@_cute_mamba3_custom_op_backward.register_fake
def _cute_mamba3_custom_op_backward_fake(
    grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
    W_X, W_Z, W_O, D_skip, grad_curr_state, grad_curr_V
):
    dx_raw = torch.empty_like(x_raw)
    dz_raw = torch.empty_like(z_raw)
    dB_rot = torch.empty_like(B_rot)
    dC_rot = torch.empty_like(C_rot)
    dt_grad = torch.empty_like(dt)
    A_grad = torch.empty_like(A)
    lam_grad = torch.empty_like(lam)
    dssm_state = torch.empty_like(ssm_state)
    dtrap_mem = torch.empty_like(trap_mem)
    dW_X = torch.empty_like(W_X)
    dW_Z = torch.empty_like(W_Z)
    dW_O = torch.empty_like(W_O)
    dD_skip = torch.empty_like(D_skip)
    return dx_raw, dz_raw, dB_rot, dC_rot, dt_grad, A_grad, lam_grad, dssm_state, dtrap_mem, dW_X, dW_Z, dW_O, dD_skip

def _cute_mamba3_custom_op_autograd_backward(ctx, grad_out, grad_curr_state, grad_curr_V):
    saved = ctx.saved_tensors
    x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem, W_X, W_Z, W_O, D_skip = saved
    return _cute_mamba3_custom_op_backward(
        grad_out, x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
        W_X, W_Z, W_O, D_skip, grad_curr_state, grad_curr_V
    )

torch.library.register_autograd("advantage_first::cute_mamba3_custom_op", _cute_mamba3_custom_op_autograd_backward, setup_context=_cute_mamba3_setup_context)

def cute_mamba3(
    x_raw: torch.Tensor,
    z_raw: torch.Tensor,
    B_rot: torch.Tensor,
    C_rot: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    ssm_state: torch.Tensor,
    trap_mem: torch.Tensor,
    W_X: torch.Tensor,
    W_Z: torch.Tensor,
    W_O: torch.Tensor,
    D_skip: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Save original dtypes and shape for exact reconstruction
    orig_x_dtype = x_raw.dtype
    orig_state_dtype = ssm_state.dtype
    orig_ndim = x_raw.ndim

    # 1. Cast and make contiguous (exposed to torch.compile)
    x_raw = ensure_fp32_contiguous(x_raw)
    z_raw = ensure_fp32_contiguous(z_raw)
    B_rot = ensure_fp32_contiguous(B_rot)
    C_rot = ensure_fp32_contiguous(C_rot)
    dt = ensure_fp32_contiguous(dt)
    A = ensure_fp32_contiguous(A)
    lam = ensure_fp32_contiguous(lam)
    ssm_state = ensure_fp32_contiguous(ssm_state)
    trap_mem = ensure_fp32_contiguous(trap_mem)
    W_X = ensure_fp32_contiguous(W_X)
    W_Z = ensure_fp32_contiguous(W_Z)
    W_O = ensure_fp32_contiguous(W_O)
    D_skip = ensure_fp32_contiguous(D_skip)

    # 2. Extract dimensions
    H_val = W_X.shape[0]
    P_val = W_X.shape[1]
    device = x_raw.device
    N_val = B_rot.shape[3]
    R_val = B_rot.shape[4]
    
    is_optimized = (P_val % 16 == 0) and (N_val % 8 == 0) and ((N_val * R_val) % 32 == 0)
    if is_optimized:
        is_tracing = is_fake_tensor(x_raw) or torch.compiler.is_dynamo_compiling()
        if is_tracing:
            out, curr_state, curr_V = _cute_mamba3_custom_op(
                x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                W_X, W_Z, W_O, D_skip
            )
        else:
            config = get_autotuned_config(P_val, N_val, R_val, "fwd", str(device))
            if config.get("fallback"):
                is_optimized = False
            else:
                out, curr_state, curr_V = _cute_mamba3_custom_op(
                    x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
                    W_X, W_Z, W_O, D_skip
                )

    if not is_optimized:
        out, curr_state, curr_V = cute_mamba3_fallback(
            x_raw, z_raw, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
            W_X, W_Z, W_O, D_skip
        )
        
    return out, curr_state, curr_V
