import functools
import torch
from _typeshed import Incomplete
from typing import Any
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton

logger: Incomplete

def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def input_to_int8(
    x: torch.Tensor, dtype: torch.dtype = ...
) -> tuple[torch.Tensor, torch.Tensor]: ...
def block_dequant(
    x_q_block: torch.Tensor, x_s: torch.Tensor, block_size: list[int]
) -> torch.Tensor: ...
@triton.jit
def round_int8(x): ...
def per_token_quant_int8(x): ...
def per_token_group_quant_int8(
    x: torch.Tensor, group_size: int, eps: float = 1e-10, dtype: torch.dtype = ...
) -> tuple[torch.Tensor, torch.Tensor]: ...
@functools.lru_cache
def get_w8a8_block_int8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None: ...
def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
