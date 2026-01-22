from datetime import datetime
from typing import Any, cast

from pydantic import Field, field_validator

from exo.plugins.type_registry import event_registry, instance_registry, task_registry
from exo.shared.topology import Connection
from exo.shared.types.chunks import GenerationChunk, InputImageChunk
from exo.shared.types.common import CommandId, Id, NodeId, SessionId
from exo.shared.types.tasks import BaseTask, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import BaseInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


@event_registry.register
class TestEvent(BaseEvent):
    __test__ = False


@event_registry.register
class TaskCreated(BaseEvent):
    task_id: TaskId
    task: BaseTask

    @field_validator("task", mode="before")
    @classmethod
    def validate_task(cls, v: Any) -> BaseTask:  # noqa: ANN401  # pyright: ignore[reportAny]
        return cast(BaseTask, task_registry.deserialize(v))  # pyright: ignore[reportAny]


@event_registry.register
class TaskAcknowledged(BaseEvent):
    task_id: TaskId


@event_registry.register
class TaskDeleted(BaseEvent):
    task_id: TaskId


@event_registry.register
class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


@event_registry.register
class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


@event_registry.register
class InstanceCreated(BaseEvent):
    instance: BaseInstance

    @field_validator("instance", mode="before")
    @classmethod
    def validate_instance(cls, v: Any) -> BaseInstance:  # noqa: ANN401  # pyright: ignore[reportAny]
        return cast(BaseInstance, instance_registry.deserialize(v))  # pyright: ignore[reportAny]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


@event_registry.register
class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


@event_registry.register
class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


@event_registry.register
class RunnerDeleted(BaseEvent):
    runner_id: RunnerId


@event_registry.register
class NodeTimedOut(BaseEvent):
    node_id: NodeId


# TODO: bikeshed this name
@event_registry.register
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


@event_registry.register
class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


@event_registry.register
class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


@event_registry.register
class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


@event_registry.register
class TopologyEdgeCreated(BaseEvent):
    conn: Connection


@event_registry.register
class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


# Union type for Pydantic validation - tries each type in order
Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | RunnerDeleted
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
)


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: BaseEvent

    @field_validator("event", mode="before")
    @classmethod
    def validate_event(cls, v: Any) -> BaseEvent:  # noqa: ANN401  # pyright: ignore[reportAny]
        return cast(BaseEvent, event_registry.deserialize(v))  # pyright: ignore[reportAny]


class ForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: BaseEvent

    @field_validator("event", mode="before")
    @classmethod
    def validate_event(cls, v: Any) -> BaseEvent:  # noqa: ANN401  # pyright: ignore[reportAny]
        return cast(BaseEvent, event_registry.deserialize(v))  # pyright: ignore[reportAny]
