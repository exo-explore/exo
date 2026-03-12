import torch
from .base import RotaryEmbedding as RotaryEmbedding
from _typeshed import Incomplete

class Llama3RotaryEmbedding(RotaryEmbedding):
    scaling_factor: Incomplete
    low_freq_factor: Incomplete
    high_freq_factor: Incomplete
    orig_max_position: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None: ...
