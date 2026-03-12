import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims as get_mla_dims,
)
from vllm.model_executor.models.deepseek_v2 import Indexer as Indexer
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionMetadata as AttentionMetadata,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
    SparseMLAAttentionImpl as SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index as triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType as KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete
FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE: Incomplete

class FlashInferMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["FlashInferMLASparseImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def is_mla(cls) -> bool: ...
    @classmethod
    def is_sparse(cls) -> bool: ...
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool: ...
    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None: ...

@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    block_size: int = ...
    topk_tokens: int = ...

class FlashInferMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashInferMLASparseMetadata]
):
    vllm_config: Incomplete
    layer_names: Incomplete
    kv_cache_spec: Incomplete
    model_config: Incomplete
    device: Incomplete
    mla_dims: Incomplete
    topk_tokens: Incomplete
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
    ) -> FlashInferMLASparseMetadata: ...

class FlashInferMLASparseImpl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    kv_cache_dtype: Incomplete
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    topk_indices_buffer: torch.Tensor | None
    bmm1_scale: float | None
    bmm2_scale: float | None
    supports_quant_query_input: bool
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
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
