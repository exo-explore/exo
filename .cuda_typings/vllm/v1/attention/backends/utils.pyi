import functools
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout as get_kv_connector_cache_layout,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
    AttentionMetadata as AttentionMetadata,
    CommonAttentionMetadata as CommonAttentionMetadata,
    subclass_attention_backend as subclass_attention_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheSpec as KVCacheSpec,
    MambaSpec as MambaSpec,
)
from vllm.v1.worker.gpu_input_batch import InputBatch as InputBatch

logger: Incomplete
KVCacheLayoutType: Incomplete
PAD_SLOT_ID: int

def is_valid_kv_cache_layout(value: str) -> bool: ...
@functools.lru_cache
def get_kv_cache_layout(): ...
def set_kv_cache_layout(cache_layout: KVCacheLayoutType): ...
@dataclass
class PerLayerParameters:
    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = ...
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)

def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]: ...
def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters: ...
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]: ...
def make_kv_sharing_fast_prefill_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata: ...
def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata, decode_threshold: int = 1
) -> tuple[int, int, int, int, int, int]: ...
def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
) -> tuple[int, int, int, int]: ...
def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]: ...
def reorder_batch_to_split_decodes_and_prefills(
    input_batch: InputBatch,
    scheduler_output: SchedulerOutput,
    decode_threshold: int = 1,
) -> bool: ...
def reshape_query_for_spec_decode(
    query: torch.Tensor, batch_size: int
) -> torch.Tensor: ...
def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor: ...
def subclass_attention_metadata(
    name_prefix: str, metadata_cls: Any, fields: list[tuple[str, Any, Any]]
) -> Any: ...

class KVSharingFastPrefillMetadata(Protocol):
    logits_indices_padded: torch.Tensor | None
    num_logits_indices: int | None

def create_fast_prefill_custom_backend(
    prefix: str, underlying_attn_backend: type[AttentionBackend]
) -> type[AttentionBackend]: ...
def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor, *, device: torch.device
): ...
def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor: ...
def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor: ...
