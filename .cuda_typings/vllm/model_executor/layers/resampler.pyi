import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)

DEFAULT_LN: Incomplete

def get_abs_pos(
    abs_pos: torch.Tensor, tgt_size: torch.Tensor | int
) -> torch.Tensor: ...
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray, version: tuple[int, int] = (2, 0)
) -> torch.Tensor: ...
def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray, version: tuple[int, int] = (2, 0)
) -> torch.Tensor: ...
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
    version: tuple[int, int] = (2, 0),
) -> torch.Tensor: ...

class BaseResampler(nn.Module):
    num_queries: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    query: Incomplete
    kv_proj: Incomplete
    attn: Incomplete
    ln_q: Incomplete
    ln_kv: Incomplete
    do_post_projection: Incomplete
    ln_post: Incomplete
    proj: Incomplete
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = ...,
        do_post_projection: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...

class Resampler2(BaseResampler):
    adaptive: Incomplete
    pos_embed: Incomplete
    def __init__(
        self,
        grid_size: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = ...,
        adaptive: bool = False,
        do_post_projection: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        tgt_sizes: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
