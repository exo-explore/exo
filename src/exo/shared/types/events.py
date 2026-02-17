import base64
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, final

from pydantic import BeforeValidator, Field, PlainSerializer

from exo.shared.topology import Connection
from exo.shared.types.chunks import GenerationChunk, InputImageChunk
from exo.shared.types.common import CommandId, Id, MetaInstanceId, NodeId, SessionId
from exo.shared.types.meta_instance import MetaInstance
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import CamelCaseModel, FrozenModel, TaggedModel


def _decode_base64_bytes(v: bytes | str) -> bytes:
    if isinstance(v, bytes):
        return v
    return base64.b64decode(v)


def _encode_base64_bytes(v: bytes) -> str:
    return base64.b64encode(v).decode("ascii")


Base64Bytes = Annotated[
    bytes,
    BeforeValidator(_decode_base64_bytes),
    PlainSerializer(_encode_base64_bytes, return_type=str),
]
"""bytes that serialize to/from base64 strings in JSON.

Needed because TaggedModel's wrap validator converts JSON→Python validation
context, which breaks strict-mode bytes deserialization from JSON strings.
"""


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    __test__ = False


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    task_id: TaskId


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId
    failure_error: str | None = None


class MetaInstanceCreated(BaseEvent):
    meta_instance: MetaInstance


class MetaInstanceDeleted(BaseEvent):
    meta_instance_id: MetaInstanceId


@final
class MetaInstancePlacementFailed(BaseEvent):
    meta_instance_id: MetaInstanceId
    reason: str


@final
class InstanceRetrying(BaseEvent):
    """Runners failed but retry count is below the limit — restart runners, keep instance."""

    instance_id: InstanceId
    meta_instance_id: MetaInstanceId
    failure_error: str


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(BaseEvent):
    runner_id: RunnerId


class NodeTimedOut(BaseEvent):
    node_id: NodeId


# TODO: bikeshed this name
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


@final
class TraceEventData(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


@final
class TracesCollected(BaseEvent):
    task_id: TaskId
    rank: int
    traces: list[TraceEventData]


@final
class TracesMerged(BaseEvent):
    task_id: TaskId
    traces: list[TraceEventData]


@final
class JacclSideChannelData(BaseEvent):
    """A runner's local contribution to a JACCL SideChannel all_gather round."""

    instance_id: InstanceId
    runner_id: RunnerId
    sequence: int
    data: Base64Bytes


@final
class JacclSideChannelGathered(BaseEvent):
    """Gathered result of a JACCL SideChannel all_gather round."""

    instance_id: InstanceId
    sequence: int
    gathered_data: Mapping[RunnerId, Base64Bytes]


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | InstanceRetrying
    | MetaInstanceCreated
    | MetaInstanceDeleted
    | MetaInstancePlacementFailed
    | RunnerStatusUpdated
    | RunnerDeleted
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
    | JacclSideChannelData
    | JacclSideChannelGathered
)


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class ForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event
