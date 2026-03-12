import functools
from _typeshed import Incomplete
from vllm.config import CacheConfig as CacheConfig
from vllm.config.vllm import VllmConfig as VllmConfig
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend as subclass_attention_backend,
)
from vllm.v1.attention.backends.utils import (
    make_local_attention_virtual_batches as make_local_attention_virtual_batches,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    ChunkedLocalAttentionSpec as ChunkedLocalAttentionSpec,
    KVCacheSpec as KVCacheSpec,
)

@functools.lru_cache
def create_chunked_local_attention_backend(
    underlying_attn_backend: AttentionBackend, attention_chunk_size: int
) -> type[AttentionBackend]: ...

class ChunkedLocalAttention(Attention):
    attention_chunk_size: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        attention_chunk_size: int,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        kv_sharing_target_layer_name: str | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...
