import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionImpl as AttentionImpl,
    AttentionLayer as AttentionLayer,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata as FlashAttentionMetadata,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention as PagedAttention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash as triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete

@dataclass
class RocmAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    scheduler_metadata: torch.Tensor | None = ...
    prefix_scheduler_metadata: torch.Tensor | None = ...

class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    block_size: Incomplete
    num_heads_q: Incomplete
    num_heads_kv: Incomplete
    headdim: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata: ...

class RocmAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def supports_mm_prefix(cls) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: ...
    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]: ...

class RocmAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey): ...
    attn_type: Incomplete
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    alibi_slopes: Incomplete
    sliding_window: Incomplete
    kv_cache_dtype: Incomplete
    logits_soft_cap: Incomplete
    kv_sharing_target_layer_name: Incomplete
    num_queries_per_kv: Incomplete
    fp8_dtype: Incomplete
    sinks: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = ...,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None: ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ): ...
    def fused_rope_kvcache_supported(self): ...
    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ): ...
