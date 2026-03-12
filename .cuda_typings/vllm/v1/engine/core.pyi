import threading
import zmq
from _typeshed import Incomplete
from collections import deque
from collections.abc import Callable as Callable
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any
from vllm.config import ParallelConfig as ParallelConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group as stateless_destroy_torch_distributed_process_group,
)
from vllm.envs import enable_envs_cache as enable_envs_cache
from vllm.logger import init_logger as init_logger
from vllm.logging_utils.dump_input import dump_engine_exception as dump_engine_exception
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS as POOLING_TASKS, SupportedTask as SupportedTask
from vllm.tracing import (
    instrument as instrument,
    maybe_init_worker_tracer as maybe_init_worker_tracer,
)
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value as maybe_register_config_serialize_by_value,
)
from vllm.utils.gc_utils import (
    freeze_gc_heap as freeze_gc_heap,
    maybe_attach_gc_debug_callback as maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name as get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket as make_zmq_socket
from vllm.utils.system_utils import (
    decorate_logs as decorate_logs,
    set_process_title as set_process_title,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash as BlockHash,
    generate_scheduler_kv_cache_config as generate_scheduler_kv_cache_config,
    get_kv_cache_configs as get_kv_cache_configs,
    get_request_block_hasher as get_request_block_hasher,
    init_none_hash as init_none_hash,
)
from vllm.v1.core.sched.interface import (
    PauseState as PauseState,
    SchedulerInterface as SchedulerInterface,
)
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.engine import (
    EEPNotificationType as EEPNotificationType,
    EEP_NOTIFICATION_CALL_ID as EEP_NOTIFICATION_CALL_ID,
    EngineCoreOutput as EngineCoreOutput,
    EngineCoreOutputs as EngineCoreOutputs,
    EngineCoreRequest as EngineCoreRequest,
    EngineCoreRequestType as EngineCoreRequestType,
    FinishReason as FinishReason,
    PauseMode as PauseMode,
    ReconfigureDistributedRequest as ReconfigureDistributedRequest,
    ReconfigureRankType as ReconfigureRankType,
    UtilityOutput as UtilityOutput,
    UtilityResult as UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata as EngineHandshakeMetadata,
    EngineZmqAddresses as EngineZmqAddresses,
    get_device_indices as get_device_indices,
)
from vllm.v1.executor import Executor as Executor
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats as SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput as ModelRunnerOutput
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus
from vllm.v1.serial_utils import (
    MsgpackDecoder as MsgpackDecoder,
    MsgpackEncoder as MsgpackEncoder,
)
from vllm.v1.structured_output import StructuredOutputManager as StructuredOutputManager
from vllm.v1.utils import compute_iteration_details as compute_iteration_details

logger: Incomplete
HANDSHAKE_TIMEOUT_MINS: int

class EngineCore:
    vllm_config: Incomplete
    log_stats: Incomplete
    model_executor: Incomplete
    available_gpu_memory_for_kv_cache: int
    structured_output_manager: Incomplete
    scheduler: SchedulerInterface
    use_spec_decode: Incomplete
    mm_registry: Incomplete
    mm_receiver_cache: Incomplete
    batch_queue_size: Incomplete
    batch_queue: (
        deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
    )
    is_ec_consumer: Incomplete
    is_pooling_model: Incomplete
    request_block_hasher: Callable[[Request], list[BlockHash]] | None
    step_fn: Incomplete
    async_scheduling: Incomplete
    aborts_queue: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ) -> None: ...
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def add_request(self, request: Request, request_wave: int = 0): ...
    def abort_requests(self, request_ids: list[str]): ...
    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput): ...
    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput): ...
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]: ...
    def post_step(self, model_executed: bool) -> None: ...
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]: ...
    def shutdown(self) -> None: ...
    def profile(self, is_start: bool = True, profile_prefix: str | None = None): ...
    def reset_mm_cache(self) -> None: ...
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    def reset_encoder_cache(self) -> None: ...
    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None: ...
    def resume_scheduler(self) -> None: ...
    def is_scheduler_paused(self) -> bool: ...
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future: ...
    def wake_up(self, tags: list[str] | None = None): ...
    def is_sleeping(self) -> bool: ...
    def execute_dummy_batch(self) -> None: ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None: ...
    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]: ...
    def preprocess_add_request(
        self, request: EngineCoreRequest
    ) -> tuple[Request, int]: ...

class EngineCoreProc(EngineCore):
    ENGINE_CORE_DEAD: bytes
    addresses: EngineZmqAddresses
    input_queue: Incomplete
    output_queue: Incomplete
    engine_index: Incomplete
    engines_running: bool
    client_count: Incomplete
    has_coordinator: Incomplete
    frontend_stats_publish_address: Incomplete
    publish_dp_lb_stats: Incomplete
    process_input_queue_block: bool
    output_thread: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        *,
        engine_index: int = 0,
    ) -> None: ...
    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses: ...
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs): ...
    def has_work(self) -> bool: ...
    def run_busy_loop(self) -> None: ...
    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ): ...
    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ): ...
    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None: ...

class DPEngineCoreProc(EngineCoreProc):
    step_counter: int
    current_wave: int
    last_counts: Incomplete
    eep_scaling_state: ElasticEPScalingState | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ) -> None: ...
    def shutdown(self) -> None: ...
    def add_request(self, request: Request, request_wave: int = 0): ...
    def resume_scheduler(self) -> None: ...
    process_input_queue_block: bool
    engines_running: Incomplete
    def run_busy_loop(self) -> None: ...
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None: ...
    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ): ...

class EngineCoreActorMixin:
    addresses: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ) -> None: ...
    def wait_for_init(self) -> None: ...
    def run(self) -> None: ...

class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ) -> None: ...

class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ) -> None: ...
