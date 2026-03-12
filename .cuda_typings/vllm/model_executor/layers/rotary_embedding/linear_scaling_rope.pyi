import torch
from .base import RotaryEmbedding as RotaryEmbedding

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    scaling_factors: list[float]
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factors: list[float] | float,
        dtype: torch.dtype,
    ) -> None: ...
    @property
    def scaling_factor_to_offset(self) -> dict[float, int]: ...
