import functools
from _typeshed import Incomplete
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend_with_overrides as subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec as CrossAttentionSpec,
    KVCacheSpec as KVCacheSpec,
)

logger: Incomplete

@functools.lru_cache
def create_cross_attention_backend(
    underlying_attn_backend: AttentionBackend,
) -> type[AttentionBackend]: ...

class CrossAttention(Attention):
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
