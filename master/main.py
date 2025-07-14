from asyncio import CancelledError, Lock, Queue, Task, create_task
from contextlib import asynccontextmanager
from enum import Enum
from logging import Logger, LogRecord
from typing import Annotated, Literal

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, TypeAdapter

from master.env import MasterEnvironmentSchema
from master.event_routing import AsyncUpdateStateFromEvents
from master.logging import (
    MasterCommandReceivedLogEntry,
    MasterInvalidCommandReceivedLogEntry,
    MasterUninitializedLogEntry,
)
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
    Event,
    EventCategories,
    EventFetcherProtocol,
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


class BaseExternalCommand[T: str](BaseModel):
    command_type: T


class ChatCompletionNonStreamingCommand(
    BaseExternalCommand[Literal["chat_completion_non_streaming"]]
):
    command_type: Literal["chat_completion_non_streaming"] = (
        "chat_completion_non_streaming"
    )


ExternalCommand = Annotated[
    ChatCompletionNonStreamingCommand, Field(discriminator="command_type")
]
ExternalCommandParser: TypeAdapter[ExternalCommand] = TypeAdapter(ExternalCommand)


class MasterBackgroundServices(str, Enum):
    MAIN_LOOP = "main_loop"


class StateManager[T: EventCategories]:
    state: State[T]
    queue: Queue[Event[T]]
    manager: AsyncUpdateStateFromEvents[T]

    def __init__(
        self,
        state: State[T],
        queue: Queue[Event[T]],
    ) -> None: ...


class MasterStateManager:
    """Thread-safe manager for MasterState with independent event loop."""

    def __init__(
        self,
        initial_state: MasterState,
        event_processor: EventFetcherProtocol[EventCategories],
        event_publisher: EventPublisher[EventCategories],
        logger: Logger,
    ):
        self._state = initial_state
        self._state_lock = Lock()
        self._command_queue: Queue[ExternalCommand] = Queue()
        self._services: dict[MasterBackgroundServices, Task[None]] = {}
        self._logger = logger

    async def read_state(self) -> MasterState:
        """Get a thread-safe snapshot of the current state."""
        async with self._state_lock:
            return self._state.model_copy(deep=True)

    async def send_command(
        self, command: ExternalCommand
    ) -> Response | StreamingResponse:
        """Send a command to the background event loop."""
        if self._services[MasterBackgroundServices.MAIN_LOOP]:
            self._command_queue.put(command)
            return Response(status_code=200)
        else:
            raise RuntimeError("State manager is not running")

    async def start(self) -> None:
        """Start the background event loop."""
        for service in MasterBackgroundServices:
            match service:
                case MasterBackgroundServices.MAIN_LOOP:
                    if self._services[service]:
                        raise RuntimeError("State manager is already running")
                    self._services[service]: Task[None] = create_task(
                        self._event_loop()
                    )
                    log(self._logger, MasterStateManagerStartedLogEntry())
                case _:
                    raise ValueError(f"Unknown service: {service}")

    async def stop(self) -> None:
        """Stop the background event loop and persist state."""
        if not self._services[MasterBackgroundServices.MAIN_LOOP]:
            raise RuntimeError("State manager is not running")

        for service in self._services.values():
            service.cancel()
            try:
                await service
            except CancelledError:
                pass

        log(self._logger, MasterStateManagerStoppedLogEntry())

    async def _event_loop(self) -> None:
        """Independent event loop for processing commands and mutating state."""
        while True:
            try:
                async with self._state_lock:
                    match EventCategories:
                        case EventCategories.InstanceEventTypes:
                            events_one = self._event_processor.get_events_to_apply(
                                self._state.data_plane_network_state
                            )
                        case EventCategories.InstanceEventTypes:
                            events_one = self._event_processor.get_events_to_apply(
                                self._state.control_plane_network_state
                            )
                        case _:
                            raise ValueError(
                                f"Unknown event category: {event_category}"
                            )
                command = self._command_queue.get(timeout=5.0)
                match command:
                    case ChatCompletionNonStreamingCommand():
                        log(
                            self._logger,
                            MasterCommandReceivedLogEntry(
                                command_name=command.command_type
                            ),
                        )
                    case _:
                        log(
                            self._logger,
                            MasterInvalidCommandReceivedLogEntry(
                                command_name=command.command_type
                            ),
                        )
            except CancelledError:
                break
            except Exception as e:
                log(self._logger, MasterStateManagerErrorLogEntry(error=str(e)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = configure_logger("master")

    telemetry_queue: Queue[LogRecord] = Queue()
    metrics_queue: Queue[LogRecord] = Queue()
    cluster_queue: Queue[LogRecord] = Queue()

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
    app.state.master_state_manager = MasterStateManager(initial_state, logger)
    await app.state.master_state_manager.start()

    yield

    await app.state.master_state_manager.stop()


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
