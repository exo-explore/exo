from asyncio import CancelledError, Lock, Task, create_task
from asyncio import Queue as AsyncQueue
from contextlib import asynccontextmanager
from logging import Logger, LogRecord
from queue import Queue as PQueue
from typing import Callable, Sequence

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from master.commands import ExternalCommand
from master.env import MasterEnvironmentSchema
from master.logging import (
    MasterCommandRunnerNotRunningLogEntry,
    MasterStateManagerStoppedLogEntry,
    MasterUninitializedLogEntry,
)
from master.router import QueueMapping
from master.state_manager.sync import SyncStateManagerMapping
from shared.constants import EXO_MASTER_STATE
from shared.logger import (
    FilterLogByType,
    LogEntryType,
    attach_to_queue,
    configure_logger,
    create_queue_listener,
    log,
)
from shared.types.events.common import (
    Apply,
    EventCategory,
    EventFromEventLog,
    EventPublisher,
    State,
)
from shared.types.models.common import ModelId
from shared.types.models.model import ModelInfo
from shared.types.states.master import MasterState
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import Instance


# Restore State
def get_master_state(logger: Logger) -> MasterState:
    if EXO_MASTER_STATE.exists():
        with open(EXO_MASTER_STATE, "r") as f:
            return MasterState.model_validate_json(f.read())
    else:
        log(logger, MasterUninitializedLogEntry())
        return MasterState()


# FastAPI Dependencies
def check_env_vars_defined(data: object, logger: Logger) -> MasterEnvironmentSchema:
    if not isinstance(data, MasterEnvironmentSchema):
        raise RuntimeError("Environment Variables Not Found")
    return data


def get_master_state_dependency(data: object, logger: Logger) -> MasterState:
    if not isinstance(data, MasterState):
        raise RuntimeError("Master State Not Found")
    return data


# Safety on Apply.
def safely_apply[T: EventCategory](
    state: State[T], apply_fn: Apply[T], events: Sequence[EventFromEventLog[T]]
) -> State[T]:
    sorted_events = sorted(events, key=lambda event: event.idx_in_log)
    state = state.model_copy()
    for event in sorted_events:
        if event.idx_in_log <= state.last_event_applied_idx:
            continue
        state.last_event_applied_idx = event.idx_in_log
        state = apply_fn(state, event)
    return state


class MasterEventLoop:
    """Thread-safe manager for MasterState with independent event loop."""

    def __init__(
        self,
        initial_state: MasterState,
        push_events_to_queue: Callable[[QueueMapping], None],
        event_publisher: EventPublisher[EventCategory],
        state_managers: SyncStateManagerMapping,
        logger: Logger,
    ):
        self._state = initial_state
        self._state_lock = Lock()
        self._event_queues: QueueMapping
        self._command_runner: ...
        self._command_run_task: Task[None] | None = None
        self._command_queue: AsyncQueue[ExternalCommand] = AsyncQueue()
        self._response_queue: AsyncQueue[Response | StreamingResponse] = AsyncQueue()
        self._state_managers: SyncStateManagerMapping
        self._state_global_lock: Lock = Lock()
        self._push_events_to_queue: Callable[[QueueMapping], None]
        self._event_fetch_task: Task[None] | None = None
        self._logger = logger

    @property
    def _is_command_runner_running(self) -> bool:
        return self._command_run_task is not None and not self._command_run_task.done()

    @property
    def _is_event_fetcher_running(self) -> bool:
        return self._event_fetch_task is not None and not self._event_fetch_task.done()

    async def send_command(
        self, command: ExternalCommand
    ) -> Response | StreamingResponse:
        """Send a command to the background event loop."""
        if self._is_command_runner_running:
            await self._command_queue.put(command)
            return await self._response_queue.get()
        else:
            log(self._logger, MasterCommandRunnerNotRunningLogEntry())
            raise RuntimeError("Command Runner Is Not Running")

    async def start(self) -> None:
        """Start the background event loop."""

        async def fetch_and_apply_events() -> None:
            while True:
                async with self._state_global_lock:
                    for state in self._state_managers.values():
                        self._push_events_to_queue(self._event_queues)
                        safely_apply(
                            state, apply_fn, self._event_queues[state.event_category]
                        )

        self._event_fetch_task = create_task(fetch_and_apply_events())
        self._command_run_task = create_task(self._command_runner())

    async def stop(self) -> None:
        """Stop the background event loop and persist state."""
        if not self._is_command_runner_running or not self._is_event_fetcher_running:
            raise RuntimeError("Command Runner Is Not Running")

        assert self._command_run_task is not None and self._event_fetch_task is not None

        for service in [self._event_fetch_task, self._command_run_task]:
            service.cancel()
            try:
                await service
            except CancelledError:
                pass

        log(self._logger, MasterStateManagerStoppedLogEntry())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = configure_logger("master")

    telemetry_queue: PQueue[LogRecord] = PQueue()
    metrics_queue: PQueue[LogRecord] = PQueue()
    cluster_queue: PQueue[LogRecord] = PQueue()

    attach_to_queue(
        logger,
        [
            FilterLogByType(log_types={LogEntryType.telemetry}),
        ],
        telemetry_queue,
    )
    attach_to_queue(
        logger,
        [
            FilterLogByType(log_types={LogEntryType.metrics}),
        ],
        metrics_queue,
    )
    attach_to_queue(
        logger,
        [
            FilterLogByType(log_types={LogEntryType.cluster}),
        ],
        cluster_queue,
    )

    # TODO: Add handlers
    telemetry_listener = create_queue_listener(telemetry_queue, [])
    metrics_listener = create_queue_listener(metrics_queue, [])
    cluster_listener = create_queue_listener(cluster_queue, [])

    telemetry_listener.start()
    metrics_listener.start()
    cluster_listener.start()

    initial_state = get_master_state(logger)
    app.state.master_event_loop = MasterEventLoop(
        initial_state, None, None, None, logger
    )
    await app.state.master_event_loop.start()

    yield

    await app.state.master_event_loop.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/topology/control_plane")
def get_control_plane_topology():
    return {"message": "Hello, World!"}


@app.get("/topology/data_plane")
def get_data_plane_topology():
    return {"message": "Hello, World!"}


@app.get("/instances/list")
def list_instances():
    return {"message": "Hello, World!"}


@app.post("/instances/create")
def create_instance(model_id: ModelId) -> InstanceId: ...


@app.get("/instance/{instance_id}/read")
def get_instance(instance_id: InstanceId) -> Instance: ...


@app.delete("/instance/{instance_id}/delete")
def remove_instance(instance_id: InstanceId) -> None: ...


@app.get("/model/{model_id}/metadata")
def get_model_data(model_id: ModelId) -> ModelInfo: ...


@app.post("/model/{model_id}/instances")
def get_instances_by_model(model_id: ModelId) -> list[Instance]: ...
