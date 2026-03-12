import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable
from typing import ClassVar
from vllm.config import (
    CacheConfig as CacheConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.cache import CacheDType as CacheDType
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales as set_default_quant_scales,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer as maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype as kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    is_quantized_kv_cache as is_quantized_kv_cache,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    KVCacheSpec as KVCacheSpec,
    MLAAttentionSpec as MLAAttentionSpec,
)

def unified_kv_cache_update(
    to_cache: torch.Tensor, layer_name: str
) -> torch.Tensor: ...
@maybe_transfer_kv_layer
def dummy_attention(layer_name, _placeholder): ...
def basic_cache(
    to_cache: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor
): ...

class CacheOnlyAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_name() -> str: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
    @classmethod
    def supports_mm_prefix(cls) -> bool: ...
    @staticmethod
    def get_impl_cls() -> type["CacheOnlyAttentionImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def get_builder_cls() -> type["CacheOnlyAttentionMetadataBuilder"]: ...
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...

class CacheOnlyAttentionMetadata:
    slot_mapping: Incomplete
    def __init__(self, slot_mapping: torch.Tensor) -> None: ...

class CacheOnlyAttentionMetadataBuilder(
    AttentionMetadataBuilder[CacheOnlyAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CacheOnlyAttentionMetadata: ...

class CacheOnlyAttentionImpl(AttentionImpl):
    num_heads: Incomplete
    head_size: Incomplete
    kv_cache_dtype: Incomplete
    kv_cache_torch_dtype: Incomplete
    num_queries_per_kv: int
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        kv_cache_dtype: str,
        kv_cache_torch_dtype: torch.dtype,
        attn_type: AttentionType = ...,
    ) -> None: ...
    def do_kv_cache_update(self, layer, to_cache, kv_cache, slot_mapping) -> None: ...
    def forward(self, *args, **kwargs) -> None: ...

class CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase):
    num_heads: Incomplete
    head_size: Incomplete
    layer_name: Incomplete
    block_size: Incomplete
    kv_cache_torch_dtype: Incomplete
    attn_backend: Incomplete
    impl: Incomplete
    kv_cache: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = ...,
    ) -> None: ...
    def forward(self, to_cache: torch.Tensor) -> torch.Tensor: ...
    def get_attn_backend(self) -> type[AttentionBackend]: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...

class ExtractHiddenStatesModel(nn.Module):
    vllm_config: Incomplete
    hf_config: Incomplete
    hidden_size: Incomplete
    target_num_hidden_layers: Incomplete
    num_hidden_states: Incomplete
    cache_only_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
