import torch
import torch.nn as nn
from .utils import (
    AttentionGroup as AttentionGroup,
    KVBlockZeroer as KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups as add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache as bind_kv_cache,
    prepare_kernel_block_sizes as prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs as sanity_check_mm_encoder_outputs,
)
from _typeshed import Incomplete
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias
from vllm.compilation.counter import compilation_counter as compilation_counter
from vllm.compilation.cuda_graph import (
    CUDAGraphStat as CUDAGraphStat,
    CUDAGraphWrapper as CUDAGraphWrapper,
)
from vllm.compilation.monitor import (
    set_cudagraph_capturing_enabled as set_cudagraph_capturing_enabled,
)
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    CompilationMode as CompilationMode,
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
    set_current_vllm_config as set_current_vllm_config,
    update_config as update_config,
)
from vllm.config.cache import CacheConfig as CacheConfig
from vllm.distributed.ec_transfer import (
    get_ec_transfer as get_ec_transfer,
    has_ec_transfer as has_ec_transfer,
)
from vllm.distributed.eplb.eplb_state import EplbState as EplbState
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group as get_kv_transfer_group,
    has_kv_transfer_group as has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    copy_kv_blocks as copy_kv_blocks,
)
from vllm.distributed.parallel_state import (
    get_dcp_group as get_dcp_group,
    get_pp_group as get_pp_group,
    get_tp_group as get_tp_group,
    graph_capture as graph_capture,
    is_global_first_rank as is_global_first_rank,
    prepare_communication_buffer_for_model as prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor as BatchDescriptor,
    set_forward_context as set_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.layers import (
    LoRAMapping as LoRAMapping,
    LoRAMappingType as LoRAMappingType,
)
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    MLAAttention as MLAAttention,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer as RoutedExpertsCapturer,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as MRotaryEmbedding,
    XDRotaryEmbedding as XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import get_model_loader as get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload as finalize_layerwise_reload,
    initialize_layerwise_reload as initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsXDRoPE as SupportsXDRoPE,
    is_mixture_of_experts as is_mixture_of_experts,
    supports_eagle3 as supports_eagle3,
    supports_mrope as supports_mrope,
    supports_multimodal_pruning as supports_multimodal_pruning,
    supports_realtime as supports_realtime,
    supports_transcription as supports_transcription,
    supports_xdrope as supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling as VllmModelForPooling,
    is_pooling_model as is_pooling_model,
    is_text_generation_model as is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader as create_offloader,
    get_offloader as get_offloader,
    set_offloader as set_offloader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget as MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs as BatchedTensorInputs,
    MultiModalKwargsItem as MultiModalKwargsItem,
    PlaceholderRange as PlaceholderRange,
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs as group_and_batch_mm_kwargs
from vllm.platforms import current_platform as current_platform
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import SamplingType as SamplingType
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tasks import (
    GenerationTask as GenerationTask,
    PoolingTask as PoolingTask,
    SupportedTask as SupportedTask,
)
from vllm.tracing import instrument as instrument
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.utils.math_utils import cdiv as cdiv, round_up as round_up
from vllm.utils.mem_utils import (
    DeviceMemoryProfiler as DeviceMemoryProfiler,
    format_gib as format_gib,
)
from vllm.utils.nvtx_pytorch_hooks import PytHooks as PytHooks
from vllm.utils.platform_utils import (
    is_pin_memory_available as is_pin_memory_available,
    num_compute_units as num_compute_units,
)
from vllm.utils.torch_utils import (
    get_dtype_size as get_dtype_size,
    kv_cache_dtype_str_to_dtype as kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionMetadata as AttentionMetadata,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    AttentionType as AttentionType,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadataBuilder as GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadataBuilder as Mamba2AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend as create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens as get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills as reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    NewRequestData as NewRequestData,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher as CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    ChunkedLocalAttentionSpec as ChunkedLocalAttentionSpec,
    CrossAttentionSpec as CrossAttentionSpec,
    EncoderOnlyAttentionSpec as EncoderOnlyAttentionSpec,
    FullAttentionSpec as FullAttentionSpec,
    KVCacheConfig as KVCacheConfig,
    KVCacheGroupSpec as KVCacheGroupSpec,
    KVCacheSpec as KVCacheSpec,
    MambaSpec as MambaSpec,
    SlidingWindowSpec as SlidingWindowSpec,
    UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    DraftTokenIds as DraftTokenIds,
    ECConnectorOutput as ECConnectorOutput,
    EMPTY_MODEL_RUNNER_OUTPUT as EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput as KVConnectorOutput,
    LogprobsLists as LogprobsLists,
    LogprobsTensors as LogprobsTensors,
    ModelRunnerOutput as ModelRunnerOutput,
    PoolerOutput as PoolerOutput,
    SamplerOutput as SamplerOutput,
    make_empty_encoder_model_runner_output as make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import (
    PoolingMetadata as PoolingMetadata,
    PoolingStates as PoolingStates,
)
from vllm.v1.sample.logits_processor import (
    LogitsProcessors as LogitsProcessors,
    build_logitsprocs as build_logitsprocs,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor as LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSampler
from vllm.v1.sample.sampler import Sampler as Sampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer as DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer as EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesProposer as ExtractHiddenStatesProposer,
)
from vllm.v1.spec_decode.medusa import MedusaProposer as MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata as SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as NgramProposer
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramProposerGPU as NgramProposerGPU,
    copy_num_valid_draft_tokens as copy_num_valid_draft_tokens,
    update_ngram_gpu_tensors_incremental as update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts as update_scheduler_for_invalid_drafts,
)
from vllm.v1.spec_decode.suffix_decoding import (
    SuffixDecodingProposer as SuffixDecodingProposer,
)
from vllm.v1.structured_output.utils import (
    apply_grammar_bitmask as apply_grammar_bitmask,
)
from vllm.v1.utils import (
    CpuGpuBuffer as CpuGpuBuffer,
    record_function_or_nullcontext as record_function_or_nullcontext,
)
from vllm.v1.worker import mamba_utils as mamba_utils
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility as check_attention_cp_compatibility,
    get_total_cp_world_size as get_total_cp_world_size,
)
from vllm.v1.worker.dp_utils import (
    coordinate_batch_across_dp as coordinate_batch_across_dp,
)
from vllm.v1.worker.ec_connector_model_runner_mixin import (
    ECConnectorModelRunnerMixin as ECConnectorModelRunnerMixin,
)
from vllm.v1.worker.gpu.pool.late_interaction_runner import (
    LateInteractionRunner as LateInteractionRunner,
)
from vllm.v1.worker.gpu_input_batch import (
    CachedRequestState as CachedRequestState,
    InputBatch as InputBatch,
)
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper as UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin as KVConnectorModelRunnerMixin,
)
from vllm.v1.worker.lora_model_runner_mixin import (
    LoRAModelRunnerMixin as LoRAModelRunnerMixin,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices as UBatchSlices,
    check_ubatch_thresholds as check_ubatch_thresholds,
    maybe_create_ubatch_slices as maybe_create_ubatch_slices,
    split_attn_metadata as split_attn_metadata,
)
from vllm.v1.worker.utils import (
    is_residual_scattered_for_sp as is_residual_scattered_for_sp,
)
from vllm.v1.worker.workspace import lock_workspace as lock_workspace

logger: Incomplete
AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict

class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    async_copy_ready_event: Incomplete
    vocab_size: Incomplete
    sampled_token_ids_cpu: Incomplete
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ) -> None: ...
    def get_output(self) -> ModelRunnerOutput: ...

class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    async_copy_ready_event: Incomplete
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ) -> None: ...
    def get_output(self) -> ModelRunnerOutput: ...

class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None

class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    vllm_config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    offload_config: Incomplete
    compilation_config: Incomplete
    lora_config: Incomplete
    load_config: Incomplete
    parallel_config: Incomplete
    scheduler_config: Incomplete
    speculative_config: Incomplete
    observability_config: Incomplete
    device: Incomplete
    pin_memory: Incomplete
    dtype: Incomplete
    kv_cache_dtype: Incomplete
    is_pooling_model: Incomplete
    enable_prompt_embeds: Incomplete
    is_multimodal_raw_input_only_model: Incomplete
    is_multimodal_pruning_enabled: bool
    max_model_len: Incomplete
    calculate_kv_scales: Incomplete
    dcp_world_size: Incomplete
    dcp_rank: Incomplete
    max_num_tokens: Incomplete
    max_num_reqs: Incomplete
    broadcast_pp_output: Incomplete
    num_query_heads: Incomplete
    inputs_embeds_size: Incomplete
    attention_chunk_size: Incomplete
    use_alibi: Incomplete
    cascade_attn_enabled: Incomplete
    is_mm_prefix_lm: Incomplete
    mm_registry: Incomplete
    uses_mrope: Incomplete
    uses_xdrope_dim: Incomplete
    supports_mm_inputs: Incomplete
    max_encoder_len: Incomplete
    use_async_scheduling: Incomplete
    sampler: Incomplete
    eplb_state: EplbState | None
    eep_eplb_suppressed: bool
    kv_caches: list[torch.Tensor]
    cross_layers_kv_cache: torch.Tensor | None
    cross_layers_attn_backend: type[AttentionBackend] | None
    attn_groups: list[list[AttentionGroup]]
    encoder_cache: dict[str, torch.Tensor]
    late_interaction_runner: Incomplete
    use_aux_hidden_state_outputs: bool
    drafter: (
        NgramProposer
        | NgramProposerGPU
        | SuffixDecodingProposer
        | EagleProposer
        | DraftModelProposer
        | MedusaProposer
        | ExtractHiddenStatesProposer
    )
    num_tokens_no_spec_gpu: Incomplete
    token_ids_gpu_tensor: Incomplete
    rejection_sampler: Incomplete
    num_spec_tokens: int
    effective_drafter_max_model_len: Incomplete
    requests: dict[str, CachedRequestState]
    num_prompt_logprobs: dict[str, int]
    comm_stream: Incomplete
    input_batch: Incomplete
    async_output_copy_stream: torch.cuda.Stream | None
    prepare_inputs_event: torch.Event | None
    cudagraph_batch_sizes: Incomplete
    encoder_timing_registry: dict[str, EncoderTimingStats]
    input_ids: Incomplete
    positions: Incomplete
    query_start_loc: Incomplete
    seq_lens: Incomplete
    encoder_seq_lens: Incomplete
    dcp_local_seq_lens: Incomplete
    inputs_embeds: Incomplete
    is_token_ids: Incomplete
    discard_request_mask: Incomplete
    num_decode_draft_tokens: Incomplete
    num_accepted_tokens: Incomplete
    is_mm_embed_buffers: Incomplete
    is_mm_embed_idx: int
    mrope_positions: Incomplete
    xdrope_positions: Incomplete
    intermediate_tensors: IntermediateTensors | None
    arange_np: Incomplete
    shared_kv_cache_layers: dict[str, str]
    kv_sharing_fast_prefill_eligible_layers: set[str]
    kv_sharing_fast_prefill_logits_indices: Incomplete
    uniform_decode_query_len: Incomplete
    cudagraph_dispatcher: Incomplete
    mm_budget: Incomplete
    reorder_batch_threshold: int | None
    runner_only_attn_layers: set[str]
    transfer_event: Incomplete
    sampled_token_ids_pinned_cpu: Incomplete
    valid_sampled_token_count_event: torch.Event | None
    valid_sampled_token_count_copy_stream: torch.cuda.Stream | None
    draft_token_ids_event: torch.Event | None
    draft_token_ids_copy_stream: torch.cuda.Stream | None
    valid_sampled_token_count_cpu: torch.Tensor | None
    draft_token_ids_cpu: torch.Tensor | None
    num_accepted_tokens_event: torch.Event | None
    execute_model_state: ExecuteModelState | None
    kv_connector_output: KVConnectorOutput | None
    mamba_state_idx: dict[str, int]
    layerwise_nvtx_hooks_registered: bool
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    def update_max_model_len(self, max_model_len: int) -> None: ...
    def reset_mm_cache(self) -> None: ...
    def reset_encoder_cache(self) -> None: ...
    def init_fp8_kv_scales(self) -> None: ...
    def get_model(self) -> nn.Module: ...
    def get_supported_generation_tasks(self) -> list[GenerationTask]: ...
    def get_supported_pooling_tasks(self) -> list[PoolingTask]: ...
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors: ...
    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None: ...
    def setup_eplb_from_mapping(
        self, expanded_physical_to_logical: torch.Tensor, old_num_physical_experts: int
    ) -> None: ...
    @contextmanager
    def synchronize_input_prep(self) -> Generator[None]: ...
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None: ...
    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    def propose_draft_token_ids(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor: ...
    def update_config(self, overrides: dict[str, Any]) -> None: ...
    model: Incomplete
    model_memory_usage: Incomplete
    def load_model(self, load_dummy_weights: bool = False) -> None: ...
    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None: ...
    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ): ...
    def profile_run(self) -> None: ...
    def profile_cudagraph_memory(self) -> int: ...
    def capture_model(self) -> int: ...
    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None: ...
    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None: ...
    def calculate_reorder_batch_threshold(self) -> None: ...
    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None: ...
    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]: ...
    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None: ...
    kv_cache_config: Incomplete
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None: ...
    max_num_kv_tokens: Incomplete
    def init_routed_experts_capturer(self) -> None: ...
    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None: ...
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]: ...
    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]: ...
    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ): ...

@dataclass
class EncoderTimingStats:
    encoder_forward_secs: float = ...
    num_encoder_calls: int = ...
    def to_dict(self) -> dict[str, float | int]: ...
