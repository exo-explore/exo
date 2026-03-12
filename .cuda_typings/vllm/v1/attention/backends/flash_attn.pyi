import numpy as np
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
    get_current_vllm_config_or_none as get_current_vllm_config_or_none,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType as CacheDType
from vllm.distributed.parallel_state import get_dcp_group as get_dcp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability as DeviceCapability
from vllm.utils.math_utils import cdiv as cdiv, round_up as round_up
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionImpl as AttentionImpl,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
    is_quantized_kv_cache as is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8 as flash_attn_supports_fp8,
    flash_attn_supports_sinks as flash_attn_supports_sinks,
    flash_attn_varlen_func as flash_attn_varlen_func,
    get_flash_attn_version as get_flash_attn_version,
    get_scheduler_metadata as get_scheduler_metadata,
    is_flash_attn_varlen_func_available as is_flash_attn_varlen_func_available,
    reshape_and_cache_flash as reshape_and_cache_flash,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens as get_dcp_local_seq_lens,
    get_kv_cache_layout as get_kv_cache_layout,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs as cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce as dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import (
    merge_attn_states as merge_attn_states,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete

class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_name() -> str: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
    @classmethod
    def supports_per_head_quant_scales(cls) -> bool: ...
    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]: ...
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
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype: ...
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool: ...
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
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
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    max_dcp_context_kv_len: int | None = ...
    dcp_context_kv_lens: torch.Tensor | None = ...
    scheduler_metadata: torch.Tensor | None = ...
    prefix_scheduler_metadata: torch.Tensor | None = ...
    max_num_splits: int = ...
    causal: bool = ...

class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    supports_update_block_table: bool
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport: ...
    model_config: Incomplete
    parallel_config: Incomplete
    cache_config: Incomplete
    compilation_config: Incomplete
    attention_config: Incomplete
    num_heads_q: Incomplete
    num_heads_kv: Incomplete
    kv_cache_dtype: Incomplete
    headdim: Incomplete
    block_size: Incomplete
    max_num_splits: int
    aot_schedule: Incomplete
    dcp_world_size: Incomplete
    dcp_rank: Incomplete
    cp_kv_cache_interleave_size: Incomplete
    use_full_cuda_graph: Incomplete
    max_cudagraph_size: Incomplete
    scheduler_metadata: Incomplete
    aot_sliding_window: tuple[int, int] | None
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
    ) -> FlashAttentionMetadata: ...
    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata: ...
    def use_cascade_attention(self, *args, **kwargs) -> bool: ...

class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    alibi_slopes: Incomplete
    sliding_window: Incomplete
    kv_cache_dtype: Incomplete
    logits_soft_cap: Incomplete
    kv_sharing_target_layer_name: Incomplete
    num_queries_per_kv: Incomplete
    attn_type: Incomplete
    vllm_flash_attn_version: Incomplete
    batch_invariant_enabled: Incomplete
    sinks: Incomplete
    supports_quant_query_input: bool
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
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None: ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
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

def use_cascade_attention(
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
def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor: ...
