import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.platforms import (
    CpuArchEnum as CpuArchEnum,
    current_platform as current_platform,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
    AttentionLayer as AttentionLayer,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    is_quantized_kv_cache as is_quantized_kv_cache,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    CrossAttentionSpec as CrossAttentionSpec,
)

logger: Incomplete

class CPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @staticmethod
    def get_name() -> str: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
    @staticmethod
    def get_impl_cls() -> type["CPUAttentionBackendImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["CPUAttentionMetadataBuilder"]: ...
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

@dataclass
class CPUAttentionMetadata:
    isa: str
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    causal: bool = ...
    use_sdpa_prefill: bool = ...
    num_decode_tokens: int = ...
    sdpa_attn_masks: list[torch.Tensor | None] | None = ...
    sdpa_start_loc: torch.Tensor | None = ...

class CPUAttentionMetadataBuilder(AttentionMetadataBuilder[CPUAttentionMetadata]):
    use_sdpa_prefill: bool
    kv_cache_spec: Incomplete
    vllm_config: Incomplete
    num_kv_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    dtype: Incomplete
    window_size: Incomplete
    block_size: Incomplete
    isa: Incomplete
    is_cross_attention: Incomplete
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
    ) -> CPUAttentionMetadata: ...

class CPUAttentionBackendImpl(AttentionImpl):
    kv_sharing_target_layer_name: Incomplete
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    logits_soft_cap: Incomplete
    num_kv_heads: Incomplete
    alibi_slopes: Incomplete
    sliding_window: Incomplete
    kv_cache_dtype: Incomplete
    num_queries_per_kv: Incomplete
    attn_type: Incomplete
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
        attn_type: str = ...,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None: ...
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
