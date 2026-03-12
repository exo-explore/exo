import torch
from .index import prepare_chunk_indices as prepare_chunk_indices
from .op import make_tensor_descriptor as make_tensor_descriptor
from .utils import (
    input_guard as input_guard,
    is_amd as is_amd,
    is_tma_supported as is_tma_supported,
)
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

FLA_TRIL_PRECISION: Incomplete
ALLOWED_TRIL_PRECISIONS: Incomplete

def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
): ...
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
): ...
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
): ...
@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
