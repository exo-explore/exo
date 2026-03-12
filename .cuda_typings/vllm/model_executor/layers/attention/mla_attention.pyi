import abc
import functools
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from typing import ClassVar, Generic, TypeVar
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
    get_current_vllm_config_or_none as get_current_vllm_config_or_none,
)
from vllm.distributed.parallel_state import (
    get_dcp_group as get_dcp_group,
    is_global_first_rank as is_global_first_rank,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.attention.attention import (
    get_attention_context as get_attention_context,
    set_default_quant_scales as set_default_quant_scales,
    should_load_quant_weights as should_load_quant_weights,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer as maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    get_and_maybe_dequant_weights as get_and_maybe_dequant_weights,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    has_flashinfer as has_flashinfer,
    has_nvidia_artifactory as has_nvidia_artifactory,
)
from vllm.utils.math_utils import cdiv as cdiv, round_down as round_down
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
    kv_cache_dtype_str_to_dtype as kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionLayer as AttentionLayer,
    AttentionMetadata as AttentionMetadata,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MLAAttentionImpl as MLAAttentionImpl,
    SparseMLAAttentionImpl as SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as get_flash_attn_version,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens as get_dcp_local_seq_lens,
    get_per_layer_parameters as get_per_layer_parameters,
    infer_global_hyperparameters as infer_global_hyperparameters,
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs as cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce as dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states as merge_attn_states,
)
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    KVCacheSpec as KVCacheSpec,
    MLAAttentionSpec as MLAAttentionSpec,
)

logger: Incomplete

class MLAAttention(nn.Module, AttentionLayerBase):
    num_heads: Incomplete
    scale: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    kv_b_proj: Incomplete
    head_size: Incomplete
    layer_name: Incomplete
    indexer: Incomplete
    num_kv_heads: int
    qk_head_dim: Incomplete
    quant_config: Incomplete
    attn_backend: Incomplete
    kv_cache_dtype: Incomplete
    calculate_kv_scales: Incomplete
    impl: Incomplete
    q_pad_num_heads: Incomplete
    use_direct_call: Incomplete
    kv_cache: Incomplete
    use_sparse: Incomplete
    dcp_a2a: Incomplete
    q_range: Incomplete
    k_range: Incomplete
    v_range: Incomplete
    is_aiter_triton_fp8_bmm_enabled: Incomplete
    is_aiter_triton_fp4_bmm_enabled: Incomplete
    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ) -> None: ...
    @property
    def chunked_prefill_workspace_size(self) -> int: ...
    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...
    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    W_K: Incomplete
    W_K_scale: Incomplete
    W_UV: Incomplete
    W_UK_T: Incomplete
    def process_weights_after_loading(self, act_dtype: torch.dtype): ...
    def calc_kv_scales(
        self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None: ...
    def get_attn_backend(self) -> type[AttentionBackend]: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec: ...

@maybe_transfer_kv_layer
def unified_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> torch.Tensor: ...
def unified_mla_attention_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> torch.Tensor: ...
def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor: ...
def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor: ...
@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None: ...
def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None: ...

class QueryLenSupport(Enum):
    SINGLE_ONLY = "single_only"
    UNIFORM = "uniform"
    VARLEN = "varlen"

is_vllm_fa: bool

def dynamic_per_batched_tensor_quant(x: torch.Tensor, dtype: torch.dtype = ...): ...

class _DecodeConcatQuantFP8(QuantFP8):
    forward_native: Incomplete
    forward_cuda: Incomplete
    forward_hip: Incomplete

CUDNN_WORKSPACE_SIZE: int

class MLACommonBackend(AttentionBackend, metaclass=abc.ABCMeta):
    accept_output_buffer: bool
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]: ...
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
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def is_mla(cls) -> bool: ...

@dataclass
class MLACommonPrefillMetadata:
    @dataclass
    class ChunkedContextMetadata:
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        workspace: torch.Tensor
        token_to_seq: torch.Tensor
        chunk_total_token: list[int]
        padded_local_chunk_seq_lens: list[list[int]] | None = ...
        local_context_lens_allranks: list[list[int]] | None = ...
        padded_local_cu_seq_lens: torch.Tensor | None = ...
        cu_seq_lens_lst: list[list[int]] | None = ...
        chunk_size: int | None = ...

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: ChunkedContextMetadata | None
    query_seq_lens: torch.Tensor | None
    workspace_buffer: torch.Tensor | None
    q_data_type: torch.dtype | None
    output_dtype: torch.dtype | None
    def __init__(
        self,
        block_table,
        query_start_loc,
        max_query_len,
        chunked_context=...,
        query_seq_lens=...,
        workspace_buffer=...,
        q_data_type=...,
        output_dtype=...,
    ) -> None: ...
    def __replace__(
        self,
        *,
        block_table,
        query_start_loc,
        max_query_len,
        chunked_context=...,
        query_seq_lens=...,
        workspace_buffer=...,
        q_data_type=...,
        output_dtype=...,
    ) -> None: ...

@dataclass
class FlashInferPrefillMetadata(MLACommonPrefillMetadata):
    prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = ...
    prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = field(
        default_factory=list
    )

@dataclass
class CudnnPrefillMetadata(MLACommonPrefillMetadata):
    class ChunkedContextMetadata(MLACommonPrefillMetadata.ChunkedContextMetadata):
        seq_lens: torch.Tensor

    cudnn_workspace: torch.Tensor | None
    def __init__(
        self,
        block_table,
        query_start_loc,
        max_query_len,
        chunked_context=...,
        query_seq_lens=...,
        workspace_buffer=...,
        q_data_type=...,
        output_dtype=...,
        cudnn_workspace=...,
    ) -> None: ...
    def __replace__(
        self,
        *,
        block_table,
        query_start_loc,
        max_query_len,
        chunked_context=...,
        query_seq_lens=...,
        workspace_buffer=...,
        q_data_type=...,
        output_dtype=...,
        cudnn_workspace=...,
    ) -> None: ...

@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None

D = TypeVar("D", bound=MLACommonDecodeMetadata)

@dataclass
class MLACommonMetadata(AttentionMetadata, Generic[D]):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    head_dim: int | None = ...
    decode: D | None = ...
    prefill: (
        MLACommonPrefillMetadata
        | FlashInferPrefillMetadata
        | CudnnPrefillMetadata
        | None
    ) = ...
    def __post_init__(self) -> None: ...

M = TypeVar("M", bound=MLACommonMetadata)
A = TypeVar("A", bound=AttentionMetadata)

def is_deepseek_r1_mla_compatible(vllm_config: VllmConfig) -> bool: ...
@functools.cache
def use_flashinfer_prefill() -> bool: ...
@functools.cache
def use_cudnn_prefill() -> bool: ...
@functools.cache
def use_trtllm_ragged_deepseek_prefill() -> bool: ...
@dataclass
class MLADims:
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

def get_mla_dims(model_config: ModelConfig) -> MLADims: ...
@functools.cache
def backend_supports_prefill_query_quantization() -> bool: ...

class MLACommonMetadataBuilder(AttentionMetadataBuilder[M]):
    query_len_support: ClassVar[QueryLenSupport]
    reorder_batch_threshold: int
    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int: ...
    @staticmethod
    def determine_prefill_query_data_type(
        vllm_config: VllmConfig, model_dtype: torch.dtype
    ) -> torch.dtype: ...
    metadata_cls: Incomplete
    kv_cache_spec: Incomplete
    model_config: Incomplete
    compilation_config: Incomplete
    vllm_config: Incomplete
    device: Incomplete
    num_heads: Incomplete
    mla_dims: Incomplete
    aot_schedule: Incomplete
    q_data_type: Incomplete
    dcp_world_size: Incomplete
    dcp_rank: Incomplete
    dcp_local_block_size: Incomplete
    dcp_virtual_block_size: Incomplete
    cp_kv_cache_interleave_size: Incomplete
    page_size: Incomplete
    chunked_prefill_workspace_size: Incomplete
    chunked_prefill_workspace: Incomplete
    prefill_metadata_cls: Incomplete
    cudnn_workspace: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[M] | None = None,
        supports_dcp_with_varlen: bool = False,
    ) -> None: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M: ...

def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list[int],
    local_context_lens_allranks: list[list[int]],
    sum_seq_len: int,
    max_seq_len: int,
    chunk_size: int,
    chunk_idx: int,
    toks: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class MLACommonImpl(MLAAttentionImpl[M], Generic[M], metaclass=abc.ABCMeta):
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    kv_cache_dtype: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    kv_b_proj: Incomplete
    indexer: Incomplete
    q_pad_num_heads: Incomplete
    supports_quant_query_input: bool
    flash_attn_varlen_func: Incomplete
    vllm_flash_attn_version: Incomplete
    dcp_world_size: int
    cp_kv_cache_interleave_size: int
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
    ) -> None: ...
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None: ...
    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
