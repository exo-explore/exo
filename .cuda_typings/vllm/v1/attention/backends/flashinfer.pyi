import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from typing import ClassVar
from typing_extensions import override
from vllm import envs as envs
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    VllmConfig as VllmConfig,
    get_current_vllm_config_or_none as get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType as CacheDType
from vllm.distributed.parallel_state import get_dcp_group as get_dcp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
)
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention as can_use_trtllm_attention,
    use_trtllm_attention as use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.torch_utils import is_strictly_contiguous as is_strictly_contiguous
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType as KVCacheLayoutType,
    get_dcp_local_seq_lens as get_dcp_local_seq_lens,
    get_kv_cache_layout as get_kv_cache_layout,
    get_per_layer_parameters as get_per_layer_parameters,
    infer_global_hyperparameters as infer_global_hyperparameters,
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs as cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce as dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states as merge_attn_states,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer as CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT: Incomplete
FP8_DTYPE: Incomplete
FP4_DTYPE: Incomplete
logger: Incomplete
trtllm_gen_workspace_buffer: Incomplete

def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class BatchDCPPrefillWrapper:
    def __init__(
        self, workspace_buffer: torch.Tensor | None = None, dcp_a2a: bool = False
    ) -> None: ...
    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ): ...
    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ): ...

class FlashInferBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]: ...
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
    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype: ...
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]: ...
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None: ...
    forward_includes_kv_cache_update: bool

@dataclass
class FIPrefill:
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper

@dataclass
class FIDecode:
    wrapper: BatchDecodeWithPagedKVCacheWrapper

@dataclass
class TRTLLMPrefill:
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor
    max_q_len: int
    max_seq_len: int

@dataclass
class TRTLLMDecode:
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int

@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    q_data_type: torch.dtype
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: FIPrefill | TRTLLMPrefill | None
    decode: FIDecode | TRTLLMDecode | None
    use_cascade: bool
    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None

class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    reorder_batch_threshold: int
    cache_config: Incomplete
    model_config: Incomplete
    attention_config: Incomplete
    decode_fixed_split_size: int
    prefill_fixed_split_size: int
    disable_split_kv: bool
    compilation_config: Incomplete
    enable_cuda_graph: Incomplete
    dcp_world_size: Incomplete
    dcp_rank: Incomplete
    dcp_kv_cache_interleave_size: Incomplete
    use_dcp: Incomplete
    dcp_a2a: Incomplete
    num_qo_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    page_size: Incomplete
    cache_dtype: Incomplete
    kv_cache_dtype: Incomplete
    q_data_type: Incomplete
    use_trtllm_decode_attention: Incomplete
    global_hyperparameters: Incomplete
    sm_scale: Incomplete
    window_left: Incomplete
    logits_soft_cap: Incomplete
    has_sinks: Incomplete
    pin_memory: Incomplete
    paged_kv_indptr: Incomplete
    paged_kv_indptr_cpu_buffer: Incomplete
    paged_kv_indices: Incomplete
    paged_kv_last_page_len: Incomplete
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    @override
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport: ...
    def set_workspace_buffer(self, workspace_buffer: torch.Tensor): ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata: ...
    def use_cascade_attention(self, *args, **kwargs) -> bool: ...

class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    alibi_slopes: Incomplete
    sliding_window: Incomplete
    window_left: Incomplete
    kv_cache_dtype: Incomplete
    logits_soft_cap: Incomplete
    kv_sharing_target_layer_name: Incomplete
    num_queries_per_kv: Incomplete
    sinks: torch.Tensor | None
    support_trtllm_attn: Incomplete
    supports_quant_query_input: Incomplete
    bmm1_scale: float | None
    bmm2_scale: float | None
    o_sf_scale: float | None
    dcp_combine: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = ...,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None: ...
    def fused_output_quant_supported(self, quant_key: QuantKey): ...
    def process_weights_after_loading(self, act_dtype: torch.dtype): ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None: ...

def fast_plan_decode(
    self,
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None: ...
