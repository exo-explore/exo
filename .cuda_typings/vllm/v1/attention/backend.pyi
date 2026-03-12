import abc
import numpy as np
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, Protocol, TypeVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
)
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.v1.attention.backends.utils import KVCacheLayoutType as KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

class AttentionType(str, Enum):
    DECODER = "decoder"
    ENCODER = "encoder"
    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"

class MultipleOf:
    base: int
    def __init__(self, base: int) -> None: ...

class AttentionBackend(ABC, metaclass=abc.ABCMeta):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]]
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    @abstractmethod
    def get_name() -> str: ...
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]: ...
    @staticmethod
    @abstractmethod
    def get_builder_cls(): ...
    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int: ...
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]: ...
    @classmethod
    def full_cls_name(cls) -> tuple[str, str]: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool: ...
    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool: ...
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool: ...
    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool: ...
    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int: ...
    @classmethod
    def is_mla(cls) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
    @classmethod
    def supports_alibi_sqrt(cls) -> bool: ...
    @classmethod
    def supports_mm_prefix(cls) -> bool: ...
    @classmethod
    def is_sparse(cls) -> bool: ...
    @classmethod
    def supports_per_head_quant_scales(cls) -> bool: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
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
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: DeviceCapability,
        attn_type: str,
    ) -> list[str]: ...
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None: ...

class AttentionMetadata: ...

T = TypeVar("T", bound=AttentionMetadata)

@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = ...
    logits_indices_padded: torch.Tensor | None = ...
    num_logits_indices: int | None = ...
    encoder_seq_lens: torch.Tensor | None = ...
    encoder_seq_lens_cpu: np.ndarray | None = ...
    dcp_local_seq_lens: torch.Tensor | None = ...
    dcp_local_seq_lens_cpu: torch.Tensor | None = ...
    def batch_size(self) -> int: ...
    def naive_query_lens(self) -> torch.Tensor: ...
    def replace(self, **kwargs) -> CommonAttentionMetadata: ...
    @property
    def seq_lens_cpu(self) -> torch.Tensor: ...
    @property
    def num_computed_tokens_cpu(self) -> torch.Tensor: ...
    def compute_num_computed_tokens(self) -> torch.Tensor: ...
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> CommonAttentionMetadata: ...

M = TypeVar("M")

class AttentionCGSupport(Enum):
    ALWAYS = 3
    UNIFORM_BATCH = 2
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    NEVER = 0

class AttentionMetadataBuilder(ABC, Generic[M], metaclass=abc.ABCMeta):
    reorder_batch_threshold: int | None
    supports_update_block_table: bool
    kv_cache_spec: Incomplete
    layer_names: Incomplete
    vllm_config: Incomplete
    device: Incomplete
    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ): ...
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport: ...
    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M: ...
    def update_block_table(
        self, metadata: M, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> M: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M: ...
    def build_for_drafting(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int
    ) -> M: ...
    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool: ...

class AttentionLayer(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...

class AttentionImplBase(ABC, Generic[T]):
    num_heads: int
    head_size: int
    scale: float
    can_return_lse_for_decode: bool
    supports_pcp: bool
    supports_mtp_with_cp_non_trivial_interleave_size: bool
    need_to_return_lse_for_decode: bool
    supports_quant_query_input: bool
    dcp_world_size: int
    dcp_rank: int
    pcp_world_size: int
    pcp_rank: int
    total_cp_world_size: int
    total_cp_rank: int
    def __new__(cls, *args, **kwargs): ...
    def process_weights_after_loading(self, act_dtype: torch.dtype): ...

class AttentionImpl(AttentionImplBase[T], Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = ...,
        kv_sharing_target_layer_name: str | None = None,
    ): ...
    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def fused_output_quant_supported(self, quant_key: QuantKey): ...
    def fused_rope_kvcache_supported(self): ...
    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ): ...

class MLAAttentionImpl(AttentionImplBase[T], Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ): ...
    @abstractmethod
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None: ...
    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None: ...

class SparseMLAAttentionImpl(AttentionImplBase[T], Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ): ...
    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None: ...

def is_quantized_kv_cache(kv_cache_dtype: str) -> bool: ...
def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]: ...
def subclass_attention_backend_with_overrides(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    overrides: dict[str, Any],
) -> type[AttentionBackend]: ...
