import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims as get_mla_dims,
)
from vllm.model_executor.models.deepseek_v2 import Indexer as Indexer
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionMetadata as AttentionMetadata,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
    SparseMLAAttentionImpl as SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index as triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode as reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode as reshape_query_for_spec_decode,
    split_decodes_and_prefills as split_decodes_and_prefills,
    split_prefill_chunks as split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta as FlashMLASchedMeta,
    flash_mla_sparse_fwd as flash_mla_sparse_fwd,
    flash_mla_with_kvcache as flash_mla_with_kvcache,
    get_mla_metadata as get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager as current_workspace_manager,
)

logger: Incomplete
MIN_HEADS_FOR_BF16_PREFILL: int

class FlashMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]: ...
    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def is_mla(cls) -> bool: ...
    @classmethod
    def is_sparse(cls) -> bool: ...
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...

@dataclass
class FlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = ...
    topk_tokens: int = ...
    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: FlashMLASchedMeta
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    class FP8SeparatePrefillDecode:
        @dataclass
        class Decode:
            kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata
            decode_query_len: int

        @dataclass
        class Prefill:
            seq_lens: torch.Tensor
            request_ids: torch.Tensor
            workspace_starts: torch.Tensor
            @dataclass
            class Chunk:
                seq_lens: torch.Tensor
                tokens_slice: slice
                block_table: torch.Tensor
                req_start_idx: int
                workspace_starts: torch.Tensor
                chunk_tot_seqlen: int

            chunks: list[Chunk]
            def __init__(
                self, seq_lens, request_ids, workspace_starts, chunks
            ) -> None: ...
            def __replace__(
                self, *, seq_lens, request_ids, workspace_starts, chunks
            ) -> None: ...

        num_prefills: int
        num_decodes: int
        num_prefill_tokens: int
        num_decode_tokens: int
        decode: Decode | None
        prefill: Prefill | None
        def __init__(
            self,
            num_prefills=...,
            num_decodes=...,
            num_prefill_tokens=...,
            num_decode_tokens=...,
            decode=...,
            prefill=...,
        ) -> None: ...
        def __replace__(
            self,
            *,
            num_prefills=...,
            num_decodes=...,
            num_prefill_tokens=...,
            num_decode_tokens=...,
            decode=...,
            prefill=...,
        ) -> None: ...

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None
    fp8_use_mixed_batch: bool
    def __init__(
        self,
        num_reqs,
        max_query_len,
        max_seq_len,
        num_actual_tokens,
        query_start_loc,
        slot_mapping,
        block_table,
        req_id_per_token,
        block_size=...,
        topk_tokens=...,
        fp8_extra_metadata=...,
        fp8_use_mixed_batch=...,
    ) -> None: ...
    def __replace__(
        self,
        *,
        num_reqs,
        max_query_len,
        max_seq_len,
        num_actual_tokens,
        query_start_loc,
        slot_mapping,
        block_table,
        req_id_per_token,
        block_size=...,
        topk_tokens=...,
        fp8_extra_metadata=...,
        fp8_use_mixed_batch=...,
    ) -> None: ...

def get_prefill_workspace_size(max_model_len: int): ...

class FlashMLASparseMetadataBuilder(AttentionMetadataBuilder[FlashMLASparseMetadata]):
    vllm_config: Incomplete
    layer_names: Incomplete
    kv_cache_spec: Incomplete
    model_config: Incomplete
    device: Incomplete
    num_heads: Incomplete
    mla_dims: Incomplete
    fp8_decode_padded_heads: Incomplete
    topk_tokens: Incomplete
    use_fp8_kv_cache: Incomplete
    topk_tokens_tensor: Incomplete
    max_model_len_tensor: Incomplete
    dummy_block_table: Incomplete
    tile_scheduler_metadata_buffer: Incomplete
    num_splits_buffer: Incomplete
    req_id_per_token_buffer: Incomplete
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
    ) -> FlashMLASparseMetadata: ...

class FlashMLASparseImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    kv_cache_dtype: Incomplete
    kv_lora_rank: int
    softmax_scale: Incomplete
    topk_indices_buffer: torch.Tensor | None
    prefill_padding: Incomplete
    fp8_decode_padded_heads: Incomplete
    prefill_workspace_shape: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Indexer | None = None,
        **mla_args,
    ) -> None: ...
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
