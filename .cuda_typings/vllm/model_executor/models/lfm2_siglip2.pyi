import torch
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    resolve_visual_encoder_outputs as resolve_visual_encoder_outputs,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Siglip2VisionConfig as Siglip2VisionConfig
from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder as should_torch_compile_mm_encoder,
    support_torch_compile as support_torch_compile,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
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

class Siglip2VisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    patch_size: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    position_embedding_size: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    def forward(
        self, pixel_values_packed: torch.FloatTensor, spatial_shapes: torch.LongTensor
    ) -> torch.Tensor: ...
    @staticmethod
    def resize_positional_embeddings_packed(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        lengths_list: list[int],
    ) -> torch.Tensor: ...

class Siglip2Attention(nn.Module):
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
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor: ...

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
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor: ...

class Siglip2Encoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]: ...

class Siglip2VisionTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_input_embeddings(self): ...
    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: list[int] | None = None,
    ) -> torch.Tensor: ...

class Siglip2Model(torch.nn.Module):
    vision_model: Incomplete
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: list[int] | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
