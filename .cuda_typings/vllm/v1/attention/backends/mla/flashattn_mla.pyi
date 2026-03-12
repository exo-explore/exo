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
from vllm.utils.math_utils import round_up as round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionType as AttentionType,
    MultipleOf as MultipleOf,
    is_quantized_kv_cache as is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_mla as flash_attn_supports_mla,
    get_flash_attn_version as get_flash_attn_version,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec
from vllm.vllm_flash_attn import (
    flash_attn_varlen_func as flash_attn_varlen_func,
    get_scheduler_metadata as get_scheduler_metadata,
)

logger: Incomplete

class FlashAttnMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]: ...
    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]: ...
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
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: torch.Tensor | None = ...
    max_num_splits: int = ...

@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]): ...

class FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    query_len_support: ClassVar[QueryLenSupport]
    reorder_batch_threshold: int
    max_num_splits: int
    fa_aot_schedule: Incomplete
    use_full_cuda_graph: Incomplete
    max_cudagraph_size: Incomplete
    scheduler_metadata: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...

class FlashAttnMLAImpl(MLACommonImpl[FlashAttnMLAMetadata]):
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
        attn_metadata: FlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
