import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as unified_attention,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete

class TreeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]: ...
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: ...

@dataclass
class TreeAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_prefill_tokens: int = ...
    num_decode_tokens: int = ...
    num_prefills: int = ...
    num_decodes: int = ...
    tree_attn_bias: torch.Tensor | None = ...
    @property
    def prefill_metadata(self) -> TreeAttentionMetadata | None: ...
    @property
    def decode_metadata(self) -> TreeAttentionMetadata | None: ...

class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    block_size: Incomplete
    tree_attn_bias: Incomplete
    reorder_batch_threshold: Incomplete
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
    ) -> TreeAttentionMetadata: ...
    def build_for_drafting(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int
    ) -> TreeAttentionMetadata: ...

class TreeAttentionImpl(AttentionImpl):
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    num_queries_per_kv: Incomplete
    kv_cache_dtype: Incomplete
    kv_sharing_target_layer_name: Incomplete
    alibi_slopes: Incomplete
    logits_soft_cap: Incomplete
    sliding_window: Incomplete
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
        kv_sharing_target_layer_name: str | None = None,
    ) -> None: ...
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None: ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
