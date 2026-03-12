import torch
import torch.nn as nn
from .common import rotate_neox as rotate_neox
from _typeshed import Incomplete
from vllm.config import get_current_vllm_config as get_current_vllm_config
from vllm.logger import init_logger as init_logger

logger: Incomplete

class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    rotary_dim: Incomplete
    head_size: Incomplete
    max_position_embeddings: Incomplete
    original_max_position_embeddings: Incomplete
    base: Incomplete
    short_factor: Incomplete
    long_factor: Incomplete
    use_long_rope: Incomplete
    short_mscale: Incomplete
    long_mscale: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        short_factor: list[float],
        long_factor: list[float],
        short_mscale: float | None = None,
        long_mscale: float | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
