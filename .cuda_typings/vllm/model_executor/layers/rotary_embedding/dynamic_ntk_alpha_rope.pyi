import torch
from .base import RotaryEmbedding as RotaryEmbedding
from _typeshed import Incomplete

class DynamicNTKAlphaRotaryEmbedding(RotaryEmbedding):
    scaling_alpha: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_alpha: float,
        dtype: torch.dtype,
    ) -> None: ...
