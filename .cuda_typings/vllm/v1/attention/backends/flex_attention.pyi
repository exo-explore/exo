import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from functools import cached_property as cached_property
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    _score_mod_signature,
)
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    is_quantized_kv_cache as is_quantized_kv_cache,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete
create_block_mask_compiled: Incomplete
flex_attention_compiled: Incomplete

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int): ...

class FlexAttentionBackend(AttentionBackend):
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
    def get_impl_cls() -> type["FlexAttentionImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]: ...
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...

def physical_to_logical_mapping(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> torch.Tensor: ...
def unique_static_unsorted(
    x: torch.Tensor, *, M: int, dim: int = -1, ignored_val: int = 0, pad_val: int = -1
) -> torch.Tensor: ...
def causal_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
): ...
@dataclass
class FlexAttentionMetadata:
    causal: bool
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
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor
    num_blocks_per_seq: torch.Tensor
    num_input_tokens: int = ...
    num_blocks = ...
    block_mask: BlockMask | None = ...
    score_mod: _score_mod_signature | None = ...
    logical_mask_mod: _mask_mod_signature = ...
    doc_ids: torch.Tensor | None = ...
    direct_build: bool = ...
    q_block_size: int = ...
    kv_block_size: int = ...
    transformed_score_mod: _score_mod_signature | None = ...
    sliding_window: int | None = ...
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = ...
    @cached_property
    def logical_block_ids(self): ...
    def get_causal_mask_mod(self) -> _mask_mod_signature: ...
    def get_bidirectional_mask_mod(self) -> _mask_mod_signature: ...
    def get_sliding_window_mask_mod(self) -> _mask_mod_signature: ...
    def get_prefix_lm_mask_mod(self) -> _mask_mod_signature: ...
    def get_mask_mod(self): ...
    def get_transformed_score_mod(self) -> _score_mod_signature | None: ...
    def build_block_mask(self) -> BlockMask: ...
    mask_mod = ...
    def __post_init__(self) -> None: ...

class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    model_config: Incomplete
    parallel_config: Incomplete
    cache_config: Incomplete
    num_heads_q: Incomplete
    num_heads_kv: Incomplete
    headdim: Incomplete
    block_size: Incomplete
    kv_cache_spec: Incomplete
    direct_build: bool
    q_block_size: int
    kv_block_size: int
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
    ) -> FlexAttentionMetadata: ...
    def use_cascade_attention(self, *args, **kwargs) -> bool: ...

class FlexAttentionImpl(AttentionImpl):
    sliding_window: int | None
    alibi_slopes: torch.Tensor | None
    logits_soft_cap: float | None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    attn_type: Incomplete
    kv_cache_dtype: Incomplete
    num_queries_per_kv: Incomplete
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
        **kwargs,
    ) -> None: ...
    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor: ...
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
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

def get_kernel_options(
    query, block_m, block_n, use_direct_build: bool
) -> dict[str, int | bool]: ...
