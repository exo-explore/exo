import torch
from .base import RotaryEmbedding as RotaryEmbedding
from .common import (
    yarn_find_correction_range as yarn_find_correction_range,
    yarn_get_mscale as yarn_get_mscale,
    yarn_linear_ramp_mask as yarn_linear_ramp_mask,
)
from _typeshed import Incomplete

class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    scaling_factor: Incomplete
    extrapolation_factor: Incomplete
    attn_factor: Incomplete
    beta_fast: Incomplete
    beta_slow: Incomplete
    truncate: Incomplete
    mscale: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_yarn_scaling: bool = True,
        truncate: bool = True,
    ) -> None: ...
