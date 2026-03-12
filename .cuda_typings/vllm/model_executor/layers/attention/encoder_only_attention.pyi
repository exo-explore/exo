import functools
from vllm.config import CacheConfig as CacheConfig
from vllm.config.vllm import VllmConfig as VllmConfig
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend as subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec as KVCacheSpec

@functools.lru_cache
def create_encoder_only_attention_backend(
    underlying_attn_backend: AttentionBackend,
) -> type[AttentionBackend]: ...

class EncoderOnlyAttention(Attention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        **kwargs,
    ) -> None: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...
