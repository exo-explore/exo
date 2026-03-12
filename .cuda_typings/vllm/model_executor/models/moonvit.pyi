import torch
import torch.nn as nn
from _typeshed import Incomplete
from functools import cached_property as cached_property
from transformers.modeling_utils import PreTrainedModel
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.model_executor.models.vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
)
from vllm.platforms import current_platform as current_platform
from vllm.transformers_utils.configs.moonvit import MoonViTConfig as MoonViTConfig

def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Learnable2DInterpPosEmb(nn.Module):
    height: Incomplete
    width: Incomplete
    interpolation_mode: Incomplete
    weight: Incomplete
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor: ...

class MoonVisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    proj: Incomplete
    pos_emb: Incomplete
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ) -> None: ...
    def forward(self, x: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor: ...

class Rope2DPosEmb(nn.Module):
    dim: Incomplete
    max_height: Incomplete
    max_width: Incomplete
    theta_base: Incomplete
    device: Incomplete
    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: int = 10000,
        device=...,
    ) -> None: ...
    def extra_repr(self): ...
    @cached_property
    def precomputed_freqs_cis(self) -> torch.Tensor: ...
    def get_freqs_cis_by_seqlens(self, grid_hws: torch.Tensor) -> torch.Tensor: ...
    def get_freqs_cis_by_idx(
        self, pos_idx: torch.Tensor, pos_idx_mask: torch.Tensor
    ) -> torch.Tensor: ...

class MLP2(nn.Module):
    use_data_parallel: Incomplete
    fc0: Incomplete
    fc1: Incomplete
    activation: Incomplete
    def __init__(
        self, dims: list[int], activation, bias: bool = True, prefix: str = ""
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MoonVitEncoderLayer(nn.Module):
    use_data_parallel: Incomplete
    num_heads: Incomplete
    hidden_dim: Incomplete
    hidden_size_per_attention_head: Incomplete
    tp_size: Incomplete
    num_attention_heads_per_partition: Incomplete
    norm0: Incomplete
    norm1: Incomplete
    mlp: Incomplete
    wqkv: Incomplete
    wo: Incomplete
    attn: Incomplete
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        prefix: str = "",
        *,
        activation=...,
        attn_bias: bool = False,
    ) -> None: ...
    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class MoonVitEncoder(nn.Module):
    rope_2d: Incomplete
    blocks: Incomplete
    final_layernorm: Incomplete
    def __init__(
        self, hidden_dim: int, num_layers: int, block_cfg: dict, prefix: str = ""
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor: ...

def patch_merger(
    x: torch.Tensor, grid_hw: torch.Tensor, merge_kernel_size: list[int, int] = (2, 2)
) -> list[torch.Tensor]: ...

class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type: str
    merge_kernel_size: Incomplete
    hidden_size: Incomplete
    patch_size: Incomplete
    vit_processing_type: str
    patch_embed: Incomplete
    encoder: Incomplete
    def __init__(
        self, config: MoonViTConfig, prefix: str = "", *inputs, **kwargs
    ) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor: ...
