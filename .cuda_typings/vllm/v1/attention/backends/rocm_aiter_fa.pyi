import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_prefills_and_extends as split_decodes_prefills_and_extends,
)
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states as merge_attn_states,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

def block_size(x, head_dim): ...
def num_programs(total_tokens): ...
@triton.jit
def cp_mha_gather_cache_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_ptr,
    value_ptr,
    block_table_ptr,
    cu_seqlens_kv_ptr,
    token_to_batch_ptr,
    seq_start_ptr,
    k_scale_ptr,
    v_scale_ptr,
    num_heads,
    head_size,
    x,
    max_block_num,
    DEQUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
): ...
def cp_mha_gather_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    token_to_batch: torch.Tensor,
    seq_starts: torch.Tensor,
    dequant: bool,
    kv_cache_layout: str,
    total_tokens: int,
): ...
@triton.jit
def reshape_and_cache_shuffle_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    x,
    k_stride0,
    v_stride0,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr,
    IS_FNUZ: tl.constexpr,
): ...
def reshape_and_cache_shuffle_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
): ...

logger: Incomplete

@dataclass
class AiterFlashAttentionDecodeMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor

@dataclass
class AiterFlashAttentionPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor

@dataclass
class AiterChunkSlidingWindowMetadata:
    swa_seqlens: torch.Tensor
    swa_cu_seqlens: torch.Tensor
    swa_seq_starts: torch.Tensor
    swa_token_to_batch: torch.Tensor
    swa_max_seqlens: int
    swa_total_tokens: int
    swa_workspace: torch.Tensor

@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    num_chunks: int
    total_token_per_batch: list[int]
    swa_metadata: AiterChunkSlidingWindowMetadata | None

@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata

@dataclass
class AiterFlashAttentionMetadata:
    num_actual_tokens: int
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_extends: int
    num_extend_tokens: int
    decode_metadata: AiterFlashAttentionDecodeMetadata | None
    prefill_metadata: AiterFlashAttentionPrefillMetadata | None
    extend_metadata: AiterFlashAttentionChunkPrefillMetadata | None
    use_cascade: bool
    common_prefix_len: int
    total_tokens: int
    k_scale: dict[str, torch.Tensor] | None
    v_scale: dict[str, torch.Tensor] | None

class AiterFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AiterFlashAttentionMetadata]
):
    model_config: Incomplete
    parallel_config: Incomplete
    cache_config: Incomplete
    num_heads_q: Incomplete
    num_heads_kv: Incomplete
    headdim: Incomplete
    block_size: Incomplete
    aot_sliding_window: tuple[int, int] | None
    total_tokens: int
    extend_workspace: Incomplete
    scale: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ): ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterFlashAttentionMetadata: ...
    def build_for_drafting(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int
    ) -> AiterFlashAttentionMetadata: ...
    def use_cascade_attention(self, *args, **kwargs) -> bool: ...

class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool: ...

class AiterFlashAttentionImpl(AttentionImpl):
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
    ) -> None: ...
    def extend_for_sliding_window(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: float,
        v_scale: float,
    ): ...
    def extend_forward(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ): ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def do_kv_cache_update(
        self,
        layer: Attention,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ): ...
