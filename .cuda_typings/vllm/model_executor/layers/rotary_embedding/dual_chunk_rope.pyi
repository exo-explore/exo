import torch
from .common import rotate_gptj as rotate_gptj, rotate_neox as rotate_neox
from _typeshed import Incomplete
from vllm.model_executor.custom_op import CustomOp as CustomOp

class DualChunkRotaryEmbedding(CustomOp):
    head_size: Incomplete
    rotary_dim: Incomplete
    max_position_embeddings: Incomplete
    base: Incomplete
    is_neox_style: Incomplete
    chunk_size: Incomplete
    local_size: Incomplete
    dtype: Incomplete
    device: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        chunk_size: int,
        local_size: int,
    ) -> None: ...
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def extra_repr(self) -> str: ...
