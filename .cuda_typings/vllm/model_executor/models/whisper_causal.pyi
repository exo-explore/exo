import functools
import torch
from .utils import make_layers as make_layers
from _typeshed import Incomplete
from torch import nn
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.models.mistral import MistralMLP as MistralMLP
from vllm.model_executor.models.whisper import (
    WhisperPosEmbedType as WhisperPosEmbedType,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend_with_overrides as subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend as FlashAttentionBackend,
)
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend as AiterFlashAttentionBackend,
)
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend as RocmAttentionBackend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend as TritonAttentionBackend,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete
CausalRMSNorm: Incomplete

class WhisperCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

@functools.lru_cache
def create_whisper_attention_backend_with_block_pooling(
    underlying_attn_backend: AttentionBackend, block_pool_size: int
) -> type[AttentionBackend]: ...

class WhisperCausalAttentionWithBlockPooling(Attention):
    block_pool_size: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = ...,
        kv_sharing_target_layer_name: str | None = None,
        block_pool_size: int = 1,
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig): ...

class WhisperCausalAttention(nn.Module):
    embed_dim: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    attn_type: Incomplete
    scaling: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        bias: bool = True,
        attn_type: AttentionType = ...,
        per_layer_sliding_window: int | None = None,
        block_pool_size: int = 1,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor | None = None
    ): ...

class WhisperCausalEncoderLayer(nn.Module):
    embed_dim: Incomplete
    head_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    mlp: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor | None = None
    ): ...

class WhisperCausalEncoder(nn.Module):
    num_mel_bins: Incomplete
    max_source_positions: Incomplete
    embed_scale: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    total_stride: Incomplete
    layer_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward_conv(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...
