from _typeshed import Incomplete
from collections.abc import Iterable
from vllm import envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat as CUDAGraphStat
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata as ECConnectorMetadata,
    ECConnectorRole as ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory as ECConnectorFactory,
)
from vllm.distributed.kv_events import (
    EventPublisherFactory as EventPublisherFactory,
    KVEventBatch as KVEventBatch,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory as KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorRole as KVConnectorRole,
    SupportsHMA as SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata as KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats as KVConnectorStats,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader as RoutedExpertsReader,
)
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalRegistry as MultiModalRegistry,
)
from vllm.multimodal.encoder_budget import MultiModalBudget as MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager as EncoderCacheManager,
    EncoderDecoderCacheManager as EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import (
    KVCacheBlocks as KVCacheBlocks,
    KVCacheManager as KVCacheManager,
)
from vllm.v1.core.kv_cache_metrics import (
    KVCacheMetricsCollector as KVCacheMetricsCollector,
)
from vllm.v1.core.sched.interface import (
    PauseState as PauseState,
    SchedulerInterface as SchedulerInterface,
)
from vllm.v1.core.sched.output import (
    CachedRequestData as CachedRequestData,
    GrammarOutput as GrammarOutput,
    NewRequestData as NewRequestData,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue as RequestQueue,
    SchedulingPolicy as SchedulingPolicy,
    create_request_queue as create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop as check_stop, remove_all as remove_all
from vllm.v1.engine import (
    EngineCoreEventType as EngineCoreEventType,
    EngineCoreOutput as EngineCoreOutput,
    EngineCoreOutputs as EngineCoreOutputs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics as ModelMetrics, PerfStats as PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats as PrefixCacheStats,
    SchedulerStats as SchedulerStats,
)
from vllm.v1.outputs import (
    DraftTokenIds as DraftTokenIds,
    KVConnectorOutput as KVConnectorOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.request import (
    Request as Request,
    RequestStatus as RequestStatus,
    StreamingUpdate as StreamingUpdate,
)
from vllm.v1.spec_decode.metrics import SpecDecodingStats as SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager as StructuredOutputManager
from vllm.v1.utils import (
    record_function_or_nullcontext as record_function_or_nullcontext,
)

logger: Incomplete

class Scheduler(SchedulerInterface):
    vllm_config: Incomplete
    scheduler_config: Incomplete
    cache_config: Incomplete
    lora_config: Incomplete
    kv_cache_config: Incomplete
    kv_events_config: Incomplete
    parallel_config: Incomplete
    log_stats: Incomplete
    observability_config: Incomplete
    kv_metrics_collector: KVCacheMetricsCollector | None
    structured_output_manager: Incomplete
    is_encoder_decoder: Incomplete
    finished_req_ids_dict: dict[int, set[str]] | None
    prev_step_scheduled_req_ids: set[str]
    max_num_running_reqs: Incomplete
    max_num_scheduled_tokens: Incomplete
    max_model_len: Incomplete
    enable_kv_cache_events: Incomplete
    connector: Incomplete
    connector_prefix_cache_stats: PrefixCacheStats | None
    recompute_kv_load_failures: bool
    kv_event_publisher: Incomplete
    ec_connector: Incomplete
    block_size: Incomplete
    dcp_world_size: Incomplete
    pcp_world_size: Incomplete
    requests: dict[str, Request]
    policy: Incomplete
    waiting: Incomplete
    skipped_waiting: Incomplete
    running: list[Request]
    finished_req_ids: set[str]
    num_waiting_for_streaming_input: int
    finished_recving_kv_req_ids: set[str]
    failed_recving_kv_req_ids: set[str]
    supports_mm_inputs: Incomplete
    mm_budget: Incomplete
    max_num_encoder_input_tokens: Incomplete
    encoder_cache_manager: Incomplete
    use_eagle: bool
    num_spec_tokens: int
    num_lookahead_tokens: Incomplete
    kv_cache_manager: Incomplete
    use_pp: Incomplete
    use_v2_model_runner: Incomplete
    has_mamba_layers: Incomplete
    needs_kv_cache_zeroing: Incomplete
    need_mamba_block_aligned_split: Incomplete
    perf_metrics: ModelMetrics | None
    routed_experts_reader: Incomplete
    max_num_kv_tokens: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = ...,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None: ...
    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]: ...
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None: ...
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None: ...
    def get_request_counts(self) -> tuple[int, int]: ...
    def add_request(self, request: Request) -> None: ...
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]: ...
    @property
    def pause_state(self) -> PauseState: ...
    def set_pause_state(self, pause_state: PauseState) -> None: ...
    def get_num_unfinished_requests(self) -> int: ...
    def has_finished_requests(self) -> bool: ...
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    def reset_connector_cache(self) -> bool: ...
    def reset_encoder_cache(self) -> None: ...
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None: ...
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None: ...
    def shutdown(self) -> None: ...
    def get_kv_connector(self) -> KVConnectorBase_V1 | None: ...
