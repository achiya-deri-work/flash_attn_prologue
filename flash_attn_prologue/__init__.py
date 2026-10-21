"""Q/K prologue APIs; optional legacy CUDA kernels are imported on demand."""
from importlib import import_module
from .functional import qknorm_rope_scale, prepare_backend
from .layers.RMSNorm import RMSNormRoPEScale

_LEGACY = {
    "qknorm_rope_scale_op": ("epilogue_kernel", "qknorm_rope_scale_op"),
    "qknorm_rope_scale_cute_op": ("fused_epilogue", "qknorm_rope_scale_cute_op"),
    "qknorm_rope_scale_cute_backward_op": ("fused_epilogue", "qknorm_rope_scale_cute_backward_op"),
    "qknorm_rope_scale_cute_inverse_op": ("fused_epilogue", "qknorm_rope_scale_cute_inverse_op"),
    "fused_qkv": ("fused_qkv", "fused_qkv_op"),
    "fused_qkv_epilogue": ("fused_qkv", "fused_qkv_epilogue_op"),
    "fused_qkv_epilogue_ckp": ("fused_qkv", "fused_qkv_epilogue_ckp_op"),
    "matmul_bwd_dx": ("gemm_bwd", "matmul_bwd_dx"),
    "matmul_bwd_dw": ("gemm_bwd", "matmul_bwd_dw"),
    "fused_qkv_backward": ("gemm_bwd", "fused_qkv_backward"),
    "CoordinateDescentAutotuner": ("coordinate_descent", "CoordinateDescentAutotuner"),
}
__all__ = ["qknorm_rope_scale", "prepare_backend", "RMSNormRoPEScale", *_LEGACY]


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(name)
    module, symbol = _LEGACY[name]
    if module == "epilogue_kernel":
        import_module(".fused_epilogue", __name__)
    if module == "fused_qkv":
        import_module(".gemm_bwd", __name__)
    loaded = import_module(f".{module}", __name__)
    # Importing the submodule installs a module-valued package attribute.
    # Restore every public alias, especially the colliding fused_qkv name.
    for alias, (source, exported) in _LEGACY.items():
        if source == module:
            globals()[alias] = getattr(loaded, exported)
    return globals()[name]
