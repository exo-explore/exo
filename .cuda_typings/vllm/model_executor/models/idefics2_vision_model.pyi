import torch
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_vision_model as run_dp_sharded_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2Config as Idefics2Config,
    Idefics2VisionConfig as Idefics2VisionConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
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

class Idefics2VisionEmbeddings(nn.Module):
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    patch_embedding: Incomplete
    num_patches_per_side: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: Idefics2VisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> torch.Tensor: ...

class Idefics2VisionAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    dropout: Incomplete
    num_heads_per_partition: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class Idefics2VisionMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Idefics2EncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    layer_norm1: Incomplete
    mlp: Incomplete
    layer_norm2: Incomplete
    def __init__(
        self,
        config: Idefics2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class Idefics2Encoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: Idefics2Config,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class Idefics2VisionTransformer(nn.Module):
    config: Incomplete
    use_data_parallel: Incomplete
    apply_encoder_attention_mask: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    require_post_norm: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool = True,
        apply_encoder_attention_mask: bool = False,
        prefix: str = "",
    ) -> None: ...
    def get_input_embeddings(self): ...
    def forward(
        self,
        pixel_values,
        patch_attention_mask: torch.BoolTensor | None = None,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
