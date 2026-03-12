import functools
import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any, NamedTuple
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group as get_dcp_group,
    get_pp_group as get_pp_group,
    prepare_communication_buffer_for_model as prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor as BatchDescriptor,
    set_forward_context as set_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader import get_model_loader as get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tasks import SupportedTask as SupportedTask
from vllm.utils.mem_utils import (
    DeviceMemoryProfiler as DeviceMemoryProfiler,
    format_gib as format_gib,
)
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE as STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.outputs import (
    DraftTokenIds as DraftTokenIds,
    KVConnectorOutput as KVConnectorOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility as check_attention_cp_compatibility,
)
from vllm.v1.worker.gpu.async_utils import (
    AsyncOutput as AsyncOutput,
    AsyncPoolingOutput as AsyncPoolingOutput,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer as build_slot_mappings_by_layer,
    get_kv_cache_spec as get_kv_cache_spec,
    init_attn_backend as init_attn_backend,
    init_kv_cache as init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables as BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu as async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import (
    prepare_dcp_local_seq_lens as prepare_dcp_local_seq_lens,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor as BatchExecutionDescriptor,
    ModelCudaGraphManager as ModelCudaGraphManager,
    get_uniform_token_count as get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import (
    sync_cudagraph_and_dp_padding as sync_cudagraph_and_dp_padding,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch as InputBatch,
    InputBuffers as InputBuffers,
    combine_sampled_and_draft_tokens as combine_sampled_and_draft_tokens,
    expand_idx_mapping as expand_idx_mapping,
    get_num_sampled_and_rejected as get_num_sampled_and_rejected,
    post_update as post_update,
    post_update_pool as post_update_pool,
    prepare_pos_seq_lens as prepare_pos_seq_lens,
    prepare_prefill_inputs as prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import (
    KVConnector as KVConnector,
    NO_OP_KV_CONNECTOR as NO_OP_KV_CONNECTOR,
    get_kv_connector as get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import LoraState as LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache as EncoderCache
from vllm.v1.worker.gpu.model_states import init_model_state as init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner as PoolingRunner
from vllm.v1.worker.gpu.pp_utils import (
    pp_broadcast as pp_broadcast,
    pp_receive as pp_receive,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput as SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    PromptLogprobsWorker as PromptLogprobsWorker,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler as Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator as init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers as set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    rejection_sample as rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.utils import (
    DraftTokensHandler as DraftTokensHandler,
)
from vllm.v1.worker.gpu.states import RequestState as RequestState
from vllm.v1.worker.gpu.structured_outputs import (
    StructuredOutputsWorker as StructuredOutputsWorker,
)
from vllm.v1.worker.lora_model_runner_mixin import (
    LoRAModelRunnerMixin as LoRAModelRunnerMixin,
)

logger: Incomplete

class GPUModelRunner(LoRAModelRunnerMixin):
    vllm_config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    compilation_config: Incomplete
    lora_config: Incomplete
    load_config: Incomplete
    parallel_config: Incomplete
    scheduler_config: Incomplete
    speculative_config: Incomplete
    observability_config: Incomplete
    device: Incomplete
    dtype: Incomplete
    kv_cache_dtype: Incomplete
    vocab_size: Incomplete
    max_model_len: Incomplete
    max_num_tokens: Incomplete
    max_num_reqs: Incomplete
    use_async_scheduling: Incomplete
    output_copy_stream: Incomplete
    output_copy_event: Incomplete
    pp_size: Incomplete
    use_pp: Incomplete
    is_first_pp_rank: Incomplete
    is_last_pp_rank: Incomplete
    dp_size: Incomplete
    dp_rank: Incomplete
    dcp_size: Incomplete
    use_dcp: Incomplete
    dcp_rank: Incomplete
    cp_interleave: Incomplete
    mm_registry: Incomplete
    supports_mm_inputs: Incomplete
    encoder_cache: Incomplete
    speculator: Incomplete
    num_speculative_steps: int
    use_aux_hidden_state_outputs: bool
    draft_tokens_handler: Incomplete
    req_states: Incomplete
    input_buffers: Incomplete
    sampler: Incomplete
    prompt_logprobs_worker: Incomplete
    decode_query_len: Incomplete
    cudagraph_manager: Incomplete
    structured_outputs_worker: Incomplete
    lora_state: Incomplete
    kv_connector: KVConnector
    is_pooling_model: Incomplete
    pooling_runner: PoolingRunner | None
    execute_model_state: ExecuteModelState | None
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    def update_max_model_len(self, max_model_len: int) -> None: ...
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    model: Incomplete
    model_memory_usage: Incomplete
    model_state: Incomplete
    def load_model(self, *args, **kwargs) -> None: ...
    def get_model(self) -> nn.Module: ...
    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream: ...
    def get_kv_cache_spec(self): ...
    kv_cache_config: Incomplete
    block_tables: Incomplete
    kv_caches: list[torch.Tensor]
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None: ...
    def profile_run(self) -> None: ...
    def reset_mm_cache(self) -> None: ...
    def reset_encoder_cache(self) -> None: ...
    def profile_cudagraph_memory(self) -> int: ...
    def capture_model(self) -> int: ...
    def warmup_for_prefill(self) -> None: ...
    def finish_requests(self, scheduler_output: SchedulerOutput) -> None: ...
    def free_states(self, scheduler_output: SchedulerOutput) -> None: ...
    def add_requests(self, scheduler_output: SchedulerOutput) -> None: ...
    def update_requests(self, scheduler_output: SchedulerOutput) -> None: ...
    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch: ...
    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]: ...
    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]: ...
    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]: ...
    def postprocess(
        self,
        input_batch: InputBatch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None: ...
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None: ...
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None: ...
    def postprocess_pool(self, input_batch: InputBatch) -> None: ...

class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | IntermediateTensors
    aux_hidden_states: list[torch.Tensor] | None
    kv_connector_output: KVConnectorOutput | None
    num_tokens_across_dp: torch.Tensor | None
