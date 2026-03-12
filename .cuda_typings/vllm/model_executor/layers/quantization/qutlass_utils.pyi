import torch
from typing import Literal
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv

@triton.jit
def triton_scale_swizzle(
    scale_ptr: torch.Tensor,
    scale_rows: int,
    scale_cols: int,
    output_ptr: torch.Tensor,
    input_row_stride: int,
    output_block_stride: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
): ...
def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor: ...
def to_blocked(
    input_matrix: torch.Tensor, backend: Literal["torch", "triton"] = "triton"
) -> torch.Tensor: ...
