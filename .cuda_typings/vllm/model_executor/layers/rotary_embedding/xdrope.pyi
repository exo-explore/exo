import numpy as np
import torch
from .dynamic_ntk_alpha_rope import (
    DynamicNTKAlphaRotaryEmbedding as DynamicNTKAlphaRotaryEmbedding,
)
from _typeshed import Incomplete

class XDRotaryEmbedding(DynamicNTKAlphaRotaryEmbedding):
    xdrope_section: Incomplete
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_alpha: float,
        dtype: torch.dtype,
        xdrope_section: list[int],
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
    @staticmethod
    def get_next_input_positions(
        context_len: int, seq_len: int, xd_sections: int = 4
    ) -> list[list[int]]: ...
    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray, out_offset: int, context_len: int, num_new_tokens: int
    ): ...
