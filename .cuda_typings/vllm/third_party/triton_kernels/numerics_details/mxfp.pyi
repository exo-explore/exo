import torch
from .mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE as MXFP_BLOCK_SIZE
from _typeshed import Incomplete
from enum import Enum

class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1

def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = ...,
): ...
def upcast_from_mxfp(
    tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int
): ...
def right_shift_unsigned(x, shift): ...
def get_max_quant_val(dtype: torch.dtype): ...
def downcast_to_mxfp_torch(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = ...,
): ...
def cvt_e2m1_to_fp32(input_tensor): ...
def upcast_from_mxfp_torch(
    tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int
): ...

quantize_mxfp8_fn: Incomplete
