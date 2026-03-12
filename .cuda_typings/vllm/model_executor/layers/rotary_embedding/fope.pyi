import torch
from .base import RotaryEmbedding as RotaryEmbedding
from .common import rotate_neox as rotate_neox
from _typeshed import Incomplete
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)

class FourierRotaryEmbedding(RotaryEmbedding):
    num_key_value_heads: Incomplete
    num_inv_freq: Incomplete
    fope_sep_head: Incomplete
    fope_init_factor: Incomplete
    inv_freq: torch.Tensor
    input_dim: Incomplete
    output_dim: Incomplete
    cos_coef: Incomplete
    sin_coef: Incomplete
    cos_sin_cache: torch.Tensor
    update_cache: bool
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool,
        num_key_value_heads: int,
        num_inv_freq: int,
        fope_sep_head: bool,
        fope_init_factor: float,
    ) -> None: ...
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor): ...
