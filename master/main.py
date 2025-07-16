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


# What The Master Cares About
MasterEventCategories = (
    Literal[EventCategoryEnum.MutatesControlPlaneState]
    | Literal[EventCategoryEnum.MutatesTaskState]
    | Literal[EventCategoryEnum.MutatesTaskSagaState]
    | Literal[EventCategoryEnum.MutatesRunnerStatus]
    | Literal[EventCategoryEnum.MutatesInstanceState]
    | Literal[EventCategoryEnum.MutatesNodePerformanceState]
    | Literal[EventCategoryEnum.MutatesDataPlaneState]
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

    # initial_state = get_master_state(logger)
    # app.state.master_event_loop = MasterEventLoop()
    # await app.state.master_event_loop.start()

    yield

    # await app.state.master_event_loop.stop()


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
