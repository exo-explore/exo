import torch
from .mrope import MRotaryEmbedding as MRotaryEmbedding
from _typeshed import Incomplete

class MRotaryEmbeddingInterleaved(MRotaryEmbedding):
    cache_max_position_num: Incomplete
    mrope_section: Incomplete
    mrope_interleaved: Incomplete
    mrope_dim: Incomplete
    layer_cache: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int],
        mrope_interleaved: bool = True,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    @staticmethod
    def get_mrope_interleaved_id_list(
        a: int, b: int, c: int, force_last: bool = False
    ) -> list[int]: ...
