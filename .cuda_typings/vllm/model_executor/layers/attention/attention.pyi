import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any
from vllm.config import (
    CacheConfig as CacheConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.vllm import VllmConfig as VllmConfig
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import MLAAttention as MLAAttention
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer as maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod as BaseKVCacheMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
    kv_cache_dtype_str_to_dtype as kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionType as AttentionType,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec as FullAttentionSpec,
    KVCacheSpec as KVCacheSpec,
    SlidingWindowSpec as SlidingWindowSpec,
)

logger: Incomplete

def validate_kv_sharing_target(
    current_layer_name, target_layer_name, static_forward_context
) -> None: ...
def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool: ...
def set_default_quant_scales(
    layer: nn.Module, register_buffer: bool = False
) -> None: ...

class Attention(nn.Module, AttentionLayerBase):
    kv_cache_torch_dtype: Incomplete
    kv_cache_dtype: Incomplete
    calculate_kv_scales: Incomplete
    quant_config: Incomplete
    layer_name: Incomplete
    num_heads: Incomplete
    head_size: Incomplete
    head_size_v: Incomplete
    num_kv_heads: Incomplete
    sliding_window: Incomplete
    has_sink: Incomplete
    use_mm_prefix: Incomplete
    attn_backend: Incomplete
    use_alibi_sqrt: Incomplete
    impl: Incomplete
    backend: Incomplete
    dtype: Incomplete
    use_direct_call: Incomplete
    use_output: Incomplete
    attn_type: Incomplete
    kv_sharing_target_layer_name: Incomplete
    kv_cache: Incomplete
    query_quant: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = ...,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...
    def calc_kv_scales(self, query, key, value) -> None: ...
    def extra_repr(self) -> str: ...
    def process_weights_after_loading(self, act_dtype: torch.dtype): ...
    def get_attn_backend(self) -> type[AttentionBackend]: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...

def maybe_calc_kv_scales(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> None: ...
def maybe_calc_kv_scales_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> None: ...
def get_attention_context(
    layer_name: str,
) -> tuple[Any, Attention | MLAAttention, torch.Tensor, torch.Tensor]: ...
@maybe_transfer_kv_layer
def unified_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> torch.Tensor: ...
def unified_attention_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> torch.Tensor: ...
def unified_kv_cache_update(
    key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> torch.Tensor: ...
def unified_kv_cache_update_fake(
    key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> torch.Tensor: ...
@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None: ...
def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None: ...
