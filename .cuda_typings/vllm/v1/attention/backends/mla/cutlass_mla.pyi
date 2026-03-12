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
)
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.attention.backend import (
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionType as AttentionType,
    MultipleOf as MultipleOf,
    is_quantized_kv_cache as is_quantized_kv_cache,
)

logger: Incomplete

class CutlassMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]): ...

class CutlassMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["CutlassMLAImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["CutlassMLAMetadataBuilder"]: ...
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool: ...

class SM100Workspace:
    def __init__(self, initial_workspace_size) -> None: ...
    def get_buf(self): ...
    def ensure_size(self, attn_metadata: MLACommonMetadata, num_kv_splits: int): ...

g_sm100_workspace: Incomplete
MAX_HEADS: int

class CutlassMLAImpl(MLACommonImpl[MLACommonMetadata]):
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
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
