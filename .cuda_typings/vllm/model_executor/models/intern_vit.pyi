import torch
import torch.nn as nn
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_vision_model as run_dp_sharded_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim as split_tensor_along_last_dim,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)

NORM2FN: Incomplete

class InternVisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    class_embedding: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: ...

class InternVisionPatchModel(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_input_embeddings(self): ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_embeds: torch.Tensor | None = None,
    ) -> torch.FloatTensor: ...

class InternParallelAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    dummy_dim: Incomplete
    num_heads_per_partition: Incomplete
    scale: Incomplete
    qkv: Incomplete
    qk_normalization: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class InternMLP(nn.Module):
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

class InternVisionEncoderLayer(nn.Module):
    embed_dim: Incomplete
    intermediate_size: Incomplete
    norm_type: Incomplete
    attn_cls: Incomplete
    attn: Incomplete
    mlp: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    ls1: Incomplete
    ls2: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
        attn_cls: type[InternParallelAttention] = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class InternVisionEncoder(nn.Module):
    config: Incomplete
    layer_cls: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
        layer_cls: type[InternVisionEncoderLayer] = ...,
    ) -> None: ...
    def forward(self, inputs_embeds: torch.Tensor): ...

class InternVisionModel(nn.Module):
    packed_modules_mapping: Incomplete
    config: Incomplete
    use_data_parallel: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
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
