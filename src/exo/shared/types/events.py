from datetime import datetime
from enum import Enum
from typing import Literal, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Connection
from exo.shared.types.chunks import Chunk, InputImageChunk
from exo.shared.types.common import CommandId, Id, ModelId, NodeId, SessionId, SystemId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


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


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


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
    chunk: Chunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


class CustomModelCardAdded(BaseEvent):
    model_card: ModelCard


class CustomModelCardDeleted(BaseEvent):
    model_id: ModelId


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


class InstanceLinkCreated(BaseEvent):
    link: InstanceLink


class InstanceLinkDeleted(BaseEvent):
    link_id: InstanceLinkId


class DrafterPlacementDegradationReason(str, Enum):
    """Why placement could not honour a model's ``drafter_eligible_nodes``.

    Surfaced on :class:`DrafterPlacementDegraded` so the operator can see
    *why* their asymmetric drafter placement was downgraded to legacy
    single-device (or no) drafter, without crawling worker logs.
    """

    NoEligibleNodeAvailable = "NoEligibleNodeAvailable"
    """No eligible node is alive in the topology (eligibility list refers
    to nodes that are missing/timed-out)."""

    AllEligibleNodesInTargetCycle = "AllEligibleNodesInTargetCycle"
    """Every listed eligible node is already a target rank, so there's no
    spare host to land the drafter on."""

    NoReachablePathFromTargetRankZero = "NoReachablePathFromTargetRankZero"
    """``MlxRing`` requires a socket connection from target rank 0 to the
    drafter node; ``MlxJaccl`` requires an RDMA edge. None of the
    eligible nodes provided one."""

    InsufficientDrafterMemory = "InsufficientDrafterMemory"
    """The first reachable eligible node lacks enough RAM for the chosen
    drafter weights."""


@final
class DrafterPlacementDegraded(BaseEvent):
    """Loud-but-graceful telemetry: asymmetric drafter requested, denied.

    Emitted by the master when a model card declares
    ``drafter_eligible_nodes`` but the placement layer cannot satisfy
    the asymmetric topology. The corresponding ``InstanceCreated`` is
    still emitted in the same step -- the user's request still
    completes, just without the asymmetric speedup -- so the operator
    sees both events and knows their cluster needs adjusting (e.g.
    bring an eligible node online, free its RAM, fix the network
    edge).

    State transition: pass-through. No state mutation; this exists
    purely for dashboard/CLI surfacing.
    """

    model_id: ModelId
    instance_id: InstanceId | None = None
    target_node_ids: list[NodeId]
    eligible_nodes: list[NodeId]
    reason: DrafterPlacementDegradationReason
    fallback: Literal["single_device_drafter", "no_drafter"]
    detail: str = ""


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
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
    | CustomModelCardAdded
    | CustomModelCardDeleted
    | InstanceLinkCreated
    | InstanceLinkDeleted
    | DrafterPlacementDegraded
)


class IndexedEvent(FrozenModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class GlobalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event


class LocalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: SystemId
    session: SessionId
    event: Event
