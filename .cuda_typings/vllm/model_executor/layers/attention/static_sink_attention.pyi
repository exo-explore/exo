import functools
import torch
from _typeshed import Incomplete
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend as subclass_attention_backend,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv as triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    KVCacheSpec as KVCacheSpec,
    SinkFullAttentionSpec as SinkFullAttentionSpec,
)

logger: Incomplete

@functools.lru_cache
def create_static_sink_attention_backend(
    underlying_attn_backend: type[AttentionBackend], sink_len: int = 0
) -> type[AttentionBackend]: ...

class StaticSinkAttention(Attention, CustomOp):
    sink_len: Incomplete
    sink_populated: bool
    sink_key: Incomplete
    sink_value: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sink_len: int,
        attn_backend: type[AttentionBackend] | None = None,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ) -> None: ...
    def update_sink_kv(self, sink_key, sink_value) -> None: ...
    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...
    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...
    def forward(self, *args, **kwargs): ...
    def populate_sink_kv(self, self_kv_cache) -> None: ...
    block_size: Incomplete
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...

def maybe_populate_sink(self_kv_cache: torch.Tensor, layer_name: str) -> None: ...
def maybe_populate_sink_fake(self_kv_cache: torch.Tensor, layer_name: str) -> None: ...
