import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend as MLACommonBackend,
    MLACommonDecodeMetadata as MLACommonDecodeMetadata,
    MLACommonImpl as MLACommonImpl,
    MLACommonMetadata as MLACommonMetadata,
    MLACommonMetadataBuilder as MLACommonMetadataBuilder,
    QueryLenSupport as QueryLenSupport,
)
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.attention.backend import (
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionType as AttentionType,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode as reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode as reshape_query_for_spec_decode,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta as FlashMLASchedMeta,
    flash_mla_with_kvcache as flash_mla_with_kvcache,
    flash_mla_with_kvcache_fp8 as flash_mla_with_kvcache_fp8,
    get_mla_metadata as get_mla_metadata,
    get_mla_metadata_dense_fp8 as get_mla_metadata_dense_fp8,
    is_flashmla_dense_supported as is_flashmla_dense_supported,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete

class FlashMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]: ...
    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]: ...
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

@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    scheduler_metadata: FlashMLASchedMeta

@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]): ...

class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    query_len_support: ClassVar[QueryLenSupport]
    reorder_batch_threshold: int
    num_q_heads: Incomplete
    cg_buf_tile_scheduler_metadata: Incomplete
    cg_buf_num_splits: Incomplete
    is_fp8_kvcache: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...

class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    can_return_lse_for_decode: bool
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
        **mla_args,
    ) -> None: ...
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
