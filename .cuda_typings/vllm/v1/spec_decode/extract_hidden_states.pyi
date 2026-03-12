import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.kv_transfer import has_kv_transfer_group as has_kv_transfer_group
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.model_loader import get_model as get_model
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher as CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput as KVConnectorOutput
from vllm.v1.worker.dp_utils import (
    coordinate_batch_across_dp as coordinate_batch_across_dp,
)
from vllm.v1.worker.gpu_input_batch import (
    CachedRequestState as CachedRequestState,
    InputBatch as InputBatch,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin as KVConnectorModelRunnerMixin,
)

PADDING_SLOT_ID: int

class ExtractHiddenStatesProposer:
    vllm_config: Incomplete
    device: Incomplete
    dtype: Incomplete
    dp_rank: Incomplete
    model: nn.Module | None
    attn_layer_names: list[str]
    attn_metadata_builder: AttentionMetadataBuilder | None
    max_num_tokens: Incomplete
    hf_config: Incomplete
    num_hidden_states: Incomplete
    hidden_size: Incomplete
    hidden_states: Incomplete
    cudagraph_dispatcher: Incomplete
    def __init__(self, vllm_config: VllmConfig, device) -> None: ...
    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        scheduler_output: SchedulerOutput,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> tuple[torch.Tensor, KVConnectorOutput | None]: ...
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None: ...
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None: ...
    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def load_model(self, target_model: nn.Module) -> None: ...
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None: ...
