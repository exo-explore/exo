import torch
from _typeshed import Incomplete
from typing import ClassVar
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend as MLACommonBackend,
    MLACommonImpl as MLACommonImpl,
    MLACommonMetadata as MLACommonMetadata,
    MLACommonMetadataBuilder as MLACommonMetadataBuilder,
    QueryLenSupport as QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionType as AttentionType,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType as KVCacheLayoutType

logger: Incomplete
FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE: Incomplete

class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    query_len_support: ClassVar[QueryLenSupport]

class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]: ...
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
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None: ...

g_fi_workspace: Incomplete

class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
    bmm1_scale: float | None
    bmm2_scale: float | None
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
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
