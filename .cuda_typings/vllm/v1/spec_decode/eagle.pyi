import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.model_loader import get_model as get_model
from vllm.model_executor.models import supports_multimodal as supports_multimodal
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
)
from vllm.model_executor.models.llama_eagle3 import (
    Eagle3LlamaForCausalLM as Eagle3LlamaForCausalLM,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import triton as triton
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata as CommonAttentionMetadata
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata as TreeAttentionMetadata,
    TreeAttentionMetadataBuilder as TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata as TritonAttentionMetadata,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher as CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    KVCacheConfig as KVCacheConfig,
    UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata as SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID as PADDING_SLOT_ID,
    compute_new_slot_mapping as compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel as copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel as eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel as eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata as eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N as extend_all_queries_by_N,
)
from vllm.v1.utils import CpuGpuBuffer as CpuGpuBuffer
from vllm.v1.worker.dp_utils import (
    coordinate_batch_across_dp as coordinate_batch_across_dp,
)
from vllm.v1.worker.gpu_input_batch import (
    CachedRequestState as CachedRequestState,
    InputBatch as InputBatch,
)
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

logger: Incomplete

class SpecDecodeBaseProposer:
    vllm_config: Incomplete
    speculative_config: Incomplete
    draft_model_config: Incomplete
    method: Incomplete
    pass_hidden_states_to_model: Incomplete
    runner: Incomplete
    device: Incomplete
    dtype: Incomplete
    max_model_len: Incomplete
    dp_rank: Incomplete
    num_speculative_tokens: Incomplete
    hidden_size: Incomplete
    inputs_embeds_size: Incomplete
    parallel_drafting: bool
    extra_slots_per_request: Incomplete
    net_num_new_slots_per_request: Incomplete
    needs_extra_input_slots: Incomplete
    parallel_drafting_token_id: int
    parallel_drafting_hidden_state_tensor: torch.Tensor | None
    use_local_argmax_reduction: bool
    max_num_tokens: Incomplete
    token_arange_np: Incomplete
    mm_registry: Incomplete
    supports_mm_inputs: Incomplete
    draft_attn_groups: list[AttentionGroup]
    kv_cache_gid: int
    eagle3_use_aux_hidden_state: bool
    compilation_config: Incomplete
    cudagraph_dispatcher: Incomplete
    input_ids: Incomplete
    uses_mrope: Incomplete
    uses_xdrope_dim: Incomplete
    draft_uses_xdrope_dim: Incomplete
    mrope_positions: Incomplete
    xdrope_positions: Incomplete
    positions: Incomplete
    hidden_states: Incomplete
    block_size: int
    arange: Incomplete
    is_rejected_token_mask: torch.Tensor | None
    is_masked_token_mask: torch.Tensor | None
    inputs_embeds: Incomplete
    backup_next_token_ids: Incomplete
    allowed_attn_types: tuple | None
    tree_choices: list[tuple[int, ...]]
    cu_drafts_per_level: Incomplete
    child_drafts_per_level: Incomplete
    tree_draft_pos_offsets: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ) -> None: ...
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None: ...
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor: ...
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]: ...
    def model_returns_tuple(self) -> bool: ...
    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor: ...
    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]: ...
    def propose_tree(
        self,
        batch_size: int,
        logits: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[torch.Tensor]: ...
    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]: ...
    def get_model_name(self, model: nn.Module) -> str: ...
    model: Incomplete
    def load_model(self, target_model: nn.Module) -> None: ...
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None: ...
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None: ...
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None: ...

class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None: ...

def compute_probs_and_sample_next_token(
    logits: torch.Tensor, sampling_metadata: SamplingMetadata
) -> tuple[torch.Tensor, torch.Tensor]: ...
