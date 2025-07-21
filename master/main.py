from contextlib import asynccontextmanager
from logging import Logger, LogRecord
from queue import Queue as PQueue
from typing import Literal

from fastapi import FastAPI

from master.env import MasterEnvironmentSchema
from master.logging import (
    MasterUninitializedLogEntry,
)
from shared.constants import EXO_MASTER_STATE
from shared.event_loops.main import NodeEventLoopProtocol
from shared.logger import (
    FilterLogByType,
    LogEntryType,
    attach_to_queue,
    configure_logger,
    create_queue_listener,
    log,
)
from shared.types.events.common import (
    EventCategoryEnum,
)
from shared.types.models.common import ModelId
from shared.types.models.model import ModelInfo
from shared.types.state import State
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import Instance


# Restore State
def get_state(logger: Logger) -> State:
    if EXO_MASTER_STATE.exists():
        with open(EXO_MASTER_STATE, "r") as f:
            return State.model_validate_json(f.read())
    else:
        log(logger, MasterUninitializedLogEntry())
        return State()


# FastAPI Dependencies
def check_env_vars_defined(data: object, logger: Logger) -> MasterEnvironmentSchema:
    if not isinstance(data, MasterEnvironmentSchema):
        raise RuntimeError("Environment Variables Not Found")
    return data


def get_state_dependency(data: object, logger: Logger) -> State:
    if not isinstance(data, State):
        raise RuntimeError("Master State Not Found")
    return data


# What The Master Cares About
MasterEventCategories = (
    Literal[EventCategoryEnum.MutatesTopologyState]
    | Literal[EventCategoryEnum.MutatesTaskState]
    | Literal[EventCategoryEnum.MutatesTaskSagaState]
    | Literal[EventCategoryEnum.MutatesRunnerStatus]
    | Literal[EventCategoryEnum.MutatesInstanceState]
    | Literal[EventCategoryEnum.MutatesNodePerformanceState]
)


# Takes Care Of All States And Events Related To The Master
class MasterEventLoopProtocol(NodeEventLoopProtocol[MasterEventCategories]): ...


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

    # TODO: Add Handlers For Pushing Logs To Remote Services
    telemetry_listener = create_queue_listener(telemetry_queue, [])
    metrics_listener = create_queue_listener(metrics_queue, [])
    cluster_listener = create_queue_listener(cluster_queue, [])

    telemetry_listener.start()
    metrics_listener.start()
    cluster_listener.start()

    # # Get validated environment
    # env = get_validated_env(MasterEnvironmentSchema, logger)
    
    # # Initialize event log manager (creates both worker and global event DBs)
    # event_log_config = EventLogConfig()  # Uses default config
    # event_log_manager = EventLogManager(
    #     config=event_log_config,
    #     logger=logger
    # )
    # await event_log_manager.initialize()
    
    # # Store for use in API handlers
    # app.state.event_log_manager = event_log_manager
    
    # # Initialize forwarder if configured
    # if env.FORWARDER_BINARY_PATH:
    #     forwarder_supervisor = ForwarderSupervisor(
    #         forwarder_binary_path=env.FORWARDER_BINARY_PATH,
    #         logger=logger
    #     )
    #     # Start as replica by default (until elected)
    #     await forwarder_supervisor.start_as_replica()
        
    #     # Create election callbacks for Rust election system
    #     election_callbacks = ElectionCallbacks(
    #         forwarder_supervisor=forwarder_supervisor,
    #         logger=logger
    #     )
        
    #     # Make callbacks available for Rust code to invoke
    #     app.state.election_callbacks = election_callbacks
        
    #     # Log status
    #     logger.info(
    #         f"Forwarder supervisor initialized. Running: {forwarder_supervisor.is_running}"
    #     )
    # else:
    #     logger.warning("No forwarder binary path configured")
    #     forwarder_supervisor = None
    # initial_state = get_master_state(logger)
    # app.state.master_event_loop = MasterEventLoop()
    # await app.state.master_event_loop.start()

    yield

    # await app.state.master_event_loop.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/topology")
def get_topology():
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
