import queue
import threading
from _typeshed import Incomplete
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from functools import cached_property as cached_property
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    destroy_distributed_environment as destroy_distributed_environment,
    destroy_model_parallel as destroy_model_parallel,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle as Handle,
    MessageQueue as MessageQueue,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator as KVOutputAggregator,
)
from vllm.distributed.parallel_state import (
    get_dcp_group as get_dcp_group,
    get_dp_group as get_dp_group,
    get_ep_group as get_ep_group,
    get_inner_dp_world_group as get_inner_dp_world_group,
    get_pcp_group as get_pcp_group,
    get_pp_group as get_pp_group,
    get_tp_group as get_tp_group,
    model_parallel_is_initialized as model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache as enable_envs_cache
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.tracing import (
    instrument as instrument,
    maybe_init_worker_tracer as maybe_init_worker_tracer,
)
from vllm.utils.network_utils import (
    get_distributed_init_method as get_distributed_init_method,
    get_ip as get_ip,
    get_loopback_ip as get_loopback_ip,
    get_open_port as get_open_port,
)
from vllm.utils.system_utils import (
    decorate_logs as decorate_logs,
    get_mp_context as get_mp_context,
    set_process_title as set_process_title,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.executor.abstract import (
    Executor as Executor,
    FailureCallback as FailureCallback,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    DraftTokenIds as DraftTokenIds,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase as WorkerWrapperBase

logger: Incomplete

class FutureWrapper(Future):
    futures_queue: Incomplete
    aggregate: Incomplete
    def __init__(
        self,
        futures_queue: deque[tuple["FutureWrapper", Callable]],
        aggregate: Callable = ...,
    ) -> None: ...
    def result(self, timeout=None): ...
    def wait_for_response(self, get_response: Callable): ...

class MultiprocExecutor(Executor):
    supports_pp: bool
    monitor_workers: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, monitor_workers: bool = True
    ) -> None: ...
    def start_worker_monitor(self, inline: bool = False) -> None: ...
    failure_callback: Incomplete
    def register_failure_callback(self, callback: FailureCallback): ...
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]: ...
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]: ...
    def execute_dummy_batch(self) -> None: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any: ...
    shutting_down: bool
    rpc_broadcast_mq: Incomplete
    response_mqs: Incomplete
    def shutdown(self) -> None: ...
    def check_health(self) -> None: ...
    @cached_property
    def max_concurrent_batches(self) -> int: ...

@dataclass
class UnreadyWorkerProcHandle:
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = ...

@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue | None
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = ...
    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> WorkerProcHandle: ...

class WorkerProc:
    READY_STR: str
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None
    rank: Incomplete
    worker: Incomplete
    use_async_scheduling: Incomplete
    async_output_queue: queue.Queue
    async_output_copy_thread: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ) -> None: ...
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle: ...
    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle: ...
    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]: ...
    def shutdown(self) -> None: ...
    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event): ...
    @staticmethod
    def worker_main(*args, **kwargs) -> None: ...
    class ResponseStatus(Enum):
        SUCCESS = ...
        FAILURE = ...

    def enqueue_output(self, output: Any): ...
    def handle_output(self, output: Any): ...
    def async_output_busy_loop(self) -> None: ...
    def worker_busy_loop(self) -> None: ...
    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None: ...

def set_multiprocessing_worker_envs() -> None: ...
