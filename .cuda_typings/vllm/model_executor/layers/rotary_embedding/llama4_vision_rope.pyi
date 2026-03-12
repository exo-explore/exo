import torch
from .base import RotaryEmbeddingBase as RotaryEmbeddingBase

class Llama4VisionRotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None: ...
    def forward_native(
        self, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cuda(
        self, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
