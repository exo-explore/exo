import torch
from .common import ApplyRotaryEmb as ApplyRotaryEmb
from _typeshed import Incomplete
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp as CustomOp

class RotaryEmbeddingBase(CustomOp):
    head_size: Incomplete
    rotary_dim: Incomplete
    max_position_embeddings: Incomplete
    base: Incomplete
    is_neox_style: Incomplete
    dtype: Incomplete
    use_flashinfer: bool
    use_aiter: Incomplete
    rocm_aiter_triton_rotary_embedding: Incomplete
    cos_sin_cache: torch.Tensor
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None: ...
    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]: ...

class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None: ...
    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def extra_repr(self) -> str: ...
