import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)

NORM2FN: Incomplete

class InternS1VisionPatchEmbeddings(nn.Module):
    image_size: Incomplete
    patch_size: Incomplete
    num_channels: Incomplete
    num_patches: Incomplete
    patch_shape: Incomplete
    projection: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class InternS1VisionEmbeddings(nn.Module):
    config: Incomplete
    cls_token: Incomplete
    mask_token: Incomplete
    patch_embeddings: Incomplete
    patch_size: Incomplete
    image_size: Incomplete
    position_embeddings: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
    ) -> torch.Tensor: ...

class InternSdpaAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    dummy_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    qk_normalization: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    projection_layer: Incomplete
    attn: Incomplete
    def __init__(
        self, config: PretrainedConfig, *, num_dummy_heads: int = 0, prefix: str = ""
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class InternS1VisionMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class InternS1VisionLayer(nn.Module):
    attention: Incomplete
    mlp: Incomplete
    layernorm_before: Incomplete
    layernorm_after: Incomplete
    lambda_1: Incomplete
    lambda_2: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class InternS1VisionEncoder(nn.Module):
    config: Incomplete
    layer: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def forward(self, inputs_embeds: torch.Tensor): ...

class InternS1VisionModel(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def get_input_embeddings(self): ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_embeds: torch.Tensor | None = None,
    ) -> torch.FloatTensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
