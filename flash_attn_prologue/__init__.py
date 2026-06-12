import os
import sys
from pathlib import Path

# Automatically discover and set CUTE_DSL_LIBS environment variable if not already set.
if "CUTE_DSL_LIBS" not in os.environ:
    # Search all sys.path directories for nvidia_cutlass_dsl runtime library
    for p in sys.path:
        candidate = Path(p) / "nvidia_cutlass_dsl" / "lib" / "libcute_dsl_runtime.so"
        if candidate.exists():
            os.environ["CUTE_DSL_LIBS"] = str(candidate.resolve())
            break
    # Fallback to sys.prefix site-packages
    if "CUTE_DSL_LIBS" not in os.environ:
        prefix_path = Path(sys.prefix)
        candidate = prefix_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia_cutlass_dsl" / "lib" / "libcute_dsl_runtime.so"
        if candidate.exists():
            os.environ["CUTE_DSL_LIBS"] = str(candidate.resolve())

from .epilogue_kernel import qknorm_rope_scale_op
from .fused_epilogue import qknorm_rope_scale_cute_op, qknorm_rope_scale_cute_backward_op, qknorm_rope_scale_cute_inverse_op
from .fused_qkv import (
    fused_qkv_op as fused_qkv,
    fused_qkv_epilogue_op as fused_qkv_epilogue,
    fused_qkv_epilogue_ckp_op as fused_qkv_epilogue_ckp
)
from .gemm_bwd import matmul_bwd_dx, matmul_bwd_dw, fused_qkv_backward
from .layers.RMSNorm import RMSNormRoPEScale
from .coordinate_descent import CoordinateDescentAutotuner

__all__ = [
    "qknorm_rope_scale_op",
    "qknorm_rope_scale_cute_op",
    "qknorm_rope_scale_cute_backward_op",
    "qknorm_rope_scale_cute_inverse_op",
    "fused_qkv",
    "fused_qkv_epilogue",
    "fused_qkv_epilogue_ckp",
    "matmul_bwd_dx",
    "matmul_bwd_dw",
    "fused_qkv_backward",
    "RMSNormRoPEScale",
    "CoordinateDescentAutotuner",
]



