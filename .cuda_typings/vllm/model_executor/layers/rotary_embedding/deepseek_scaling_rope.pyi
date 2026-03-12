import torch
from .base import RotaryEmbeddingBase as RotaryEmbeddingBase
from .common import (
    rotate_gptj as rotate_gptj,
    rotate_neox as rotate_neox,
    yarn_find_correction_range as yarn_find_correction_range,
    yarn_linear_ramp_mask as yarn_linear_ramp_mask,
)
from _typeshed import Incomplete
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import has_flashinfer as has_flashinfer

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float: ...

class DeepseekScalingRotaryEmbedding(RotaryEmbeddingBase):
    scaling_factor: Incomplete
    extrapolation_factor: Incomplete
    attn_factor: Incomplete
    beta_fast: Incomplete
    beta_slow: Incomplete
    mscale: Incomplete
    use_flashinfer: Incomplete
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
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None: ...
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
