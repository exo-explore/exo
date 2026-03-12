import triton
import triton.language as tl
from ..numerics import (
    MAX_FINITE_FLOAT8E4B8 as MAX_FINITE_FLOAT8E4B8,
    MAX_FINITE_FLOAT8E4NV as MAX_FINITE_FLOAT8E4NV,
    MAX_FINITE_FLOAT8E5 as MAX_FINITE_FLOAT8E5,
)
from _typeshed import Incomplete

TL_MAX_FINITE_FLOAT8E5: Incomplete
TL_MAX_FINITE_FLOAT8E4NV: Incomplete
TL_MAX_FINITE_FLOAT8E4B8: Incomplete
TL_MAX_FINITE_FLOAT8E4B15: Incomplete
TL_MAX_FINITE_FLOAT16: Incomplete
TL_RCP_MAX_FINITE_FLOAT8E5: Incomplete
TL_RCP_MAX_FINITE_FLOAT8E4NV: Incomplete
TL_RCP_MAX_FINITE_FLOAT8E4B8: Incomplete
TL_RCP_MAX_FINITE_FLOAT8E4B15: Incomplete
TL_RCP_MAX_FINITE_FLOAT16: Incomplete

@triton.jit
def max_finite(dtype): ...
@triton.jit
def rcp_max_finite(dtype): ...
@triton.jit
def sm86_min_nan_xorsign_abs_f32(a, b): ...
@triton.jit
def sm86_max_nan_xorsign_abs_f32(a, b): ...
@triton.jit
def load_scale(scale_ptr): ...
@triton.jit
def flex_to_float(x, scale_ptr): ...
@triton.jit
def clip(x, limit): ...
@triton.jit
def nan_propagating_absmax_reduce(x, axis=None): ...
@triton.jit
def compute_scale(x, Out): ...
@triton.jit
def update_scale(x, scale_ptr, Out) -> None: ...
@triton.jit
def float_to_flex(
    x,
    expected_scale_ptr_or_val,
    actual_scale_ptr,
    checksum_scale_ptr,
    mask,
    Out,
    saturate_infs: tl.constexpr,
): ...
