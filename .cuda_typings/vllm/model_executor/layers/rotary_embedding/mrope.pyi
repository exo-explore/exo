import numpy as np
import torch
from .base import RotaryEmbeddingBase as RotaryEmbeddingBase
from .yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding as YaRNScalingRotaryEmbedding,
    yarn_get_mscale as yarn_get_mscale,
)
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def apply_interleaved_rope(
    x: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor: ...

class MRotaryEmbedding(RotaryEmbeddingBase):
    scaling_factor: Incomplete
    extrapolation_factor: Incomplete
    attn_factor: Incomplete
    beta_fast: Incomplete
    beta_slow: Incomplete
    truncate: Incomplete
    mscale: Incomplete
    cache_max_position_num: Incomplete
    mrope_section: Incomplete
    mrope_interleaved: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        *,
        scaling_factor: float | None = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        truncate: bool = True,
    ) -> None: ...
    def forward_native(
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
    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int, context_len: int, seq_len: int
    ) -> list[list[int]]: ...
    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ): ...
