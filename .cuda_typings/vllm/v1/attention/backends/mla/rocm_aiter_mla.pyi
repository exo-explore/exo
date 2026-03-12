import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend as MLACommonBackend,
    MLACommonDecodeMetadata as MLACommonDecodeMetadata,
    MLACommonImpl as MLACommonImpl,
    MLACommonMetadata as MLACommonMetadata,
    MLACommonMetadataBuilder as MLACommonMetadataBuilder,
    QueryLenSupport as QueryLenSupport,
)
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.attention.backend import (
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    MultipleOf as MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

class AiterMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]: ...

@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    paged_kv_indptr: torch.Tensor | None = ...
    paged_kv_indices: torch.Tensor | None = ...
    paged_kv_last_page_len: torch.Tensor | None = ...
    qo_indptr: torch.Tensor | None = ...
    attn_out_dtype: torch.dtype = ...
    max_qo_len: int | None = ...

class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]): ...

class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    query_len_support: ClassVar[QueryLenSupport]
    compilation_config: Incomplete
    decode_attn_out_dtype: Incomplete
    paged_kv_last_page_len: Incomplete
    paged_kv_indices: Incomplete
    paged_kv_indptr: Incomplete
    qo_indptr: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...

class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    flash_attn_varlen_func: Incomplete
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
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
