import abc
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata as get_paged_mqa_logits_metadata,
    is_deep_gemm_supported as is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills as split_decodes_and_prefills,
    split_prefill_chunks as split_prefill_chunks,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size as get_total_cp_world_size

logger: Incomplete

class DeepseekV32IndexerBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]: ...

@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int

@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]

@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    use_large_context_topk: bool
    offsets: torch.Tensor | None

@dataclass
class DeepseekV32IndexerMetadata:
    seq_lens: torch.Tensor
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    head_dim: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    decode: DeepSeekV32IndexerDecodeMetadata | None = ...
    prefill: DeepseekV32IndexerPrefillMetadata | None = ...

def kv_spans_from_batches(
    start_seq_loc: torch.Tensor, seq_len_per_batch: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]: ...
def get_max_prefill_buffer_size(vllm_config: VllmConfig): ...

class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport: ...
    max_prefill_buffer_size: Incomplete
    num_speculative_tokens: Incomplete
    num_sms: Incomplete
    decode_lens_buffer: Incomplete
    arange_buffer: Incomplete
    expanded_seq_lens_buffer: Incomplete
    expanded_block_table_buffer: Incomplete
    scheduler_metadata_buffer: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def build_one_prefill_chunk(
        self, reqs_start, reqs_end, query_start_loc_cpu, seq_lens_cpu, block_table
    ): ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata: ...
