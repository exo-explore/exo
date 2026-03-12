import torch
from .vision import is_vit_use_data_parallel as is_vit_use_data_parallel
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Siglip2VisionConfig as Siglip2VisionConfig
from transformers.configuration_utils import PretrainedConfig as PretrainedConfig
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    LinearBase as LinearBase,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.platforms import current_platform as current_platform

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Siglip2VisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    patch_size: Incomplete
    image_size: Incomplete
    num_patches: Incomplete
    preserve_original_pe: Incomplete
    hidden_stride: Incomplete
    patch_embedding: Incomplete
    position_embedding_size: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, grid_thws: torch.LongTensor | None = None
    ) -> torch.Tensor: ...

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_flash_attn_backend: bool,
    apply_rotary_emb: ApplyRotaryEmb,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Siglip2Attention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    dropout: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    tp_size: Incomplete
    num_heads_per_partition: Incomplete
    use_rope: Incomplete
    attn: Incomplete
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Siglip2MLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Siglip2EncoderLayer(nn.Module):
    embed_dim: Incomplete
    layer_norm1: Incomplete
    self_attn: Incomplete
    layer_norm2: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> tuple[torch.FloatTensor]: ...

class Siglip2Encoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    rotary_pos_emb: Incomplete
    patch_size: Incomplete
    hidden_stride: Incomplete
    window_size: Incomplete
    spatial_merge_unit: Incomplete
    fullatt_block_indexes: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def rot_pos_emb(self, grid_thw): ...
    def get_window_index(self, grid_thw): ...
    def forward(
        self, inputs_embeds: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor: ...

class Siglip2VisionTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, grid_thws: torch.LongTensor
    ) -> torch.Tensor: ...

class Siglip2NavitModel(torch.nn.Module):
    vision_model: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, grid_thws: torch.LongTensor
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
