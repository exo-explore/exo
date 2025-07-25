import types
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import Field

from shared.constants import get_error_reporting_message
from shared.topology import Connection, ConnectionProfile, NodePerformanceProfile
from shared.types.common import NodeId
from shared.types.events.chunks import CommandId, GenerationChunk
from shared.types.tasks import Task, TaskId, TaskStatus
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import Instance
from shared.types.worker.runners import RunnerId, RunnerStatus

if TYPE_CHECKING:
    pass

from pydantic import BaseModel

from shared.types.common import ID


class EventId(ID):
    """
    Newtype around `ID`
    """


# Event base-class boilerplate (you should basically never touch these)
# Only very specialised registry or serialisation/deserialization logic might need know about these

class _EventType(str, Enum):
    """
    Here are all the unique kinds of events that can be sent over the network.
    """

    # Task Events
    TaskCreated = "TaskCreated"
    TaskStateUpdated = "TaskStateUpdated"
    TaskDeleted = "TaskDeleted"

    # Streaming Events
    ChunkGenerated = "ChunkGenerated"

    # Instance Events
    InstanceCreated = "InstanceCreated"
    InstanceDeleted = "InstanceDeleted"
    InstanceActivated = "InstanceActivated"
    InstanceDeactivated = "InstanceDeactivated"
    InstanceReplacedAtomically = "InstanceReplacedAtomically"

    # Runner Status Events
    RunnerStatusUpdated = "RunnerStatusUpdated"
    RunnerDeleted = "RunnerDeleted"

    # Node Performance Events
    NodePerformanceMeasured = "NodePerformanceMeasured"

    # Topology Events
    TopologyEdgeCreated = "TopologyEdgeCreated"
    TopologyEdgeReplacedAtomically = "TopologyEdgeReplacedAtomically"
    TopologyEdgeDeleted = "TopologyEdgeDeleted"
    WorkerStatusUpdated = "WorkerStatusUpdated"

    # # Timer Events
    # TimerCreated = "TimerCreated"
    # TimerFired = "TimerFired"


class _BaseEvent[T: _EventType](BaseModel):
    """
    This is the event base-class, to please the Pydantic gods.
    PLEASE don't use this for anything unless you know why you are doing so,
    instead just use the events union :)
    """

    event_type: T
    event_id: EventId = EventId()

    def check_event_was_sent_by_correct_node(self, origin_id: NodeId) -> bool:
        """Check if the event was sent by the correct node.

        This is a placeholder implementation that always returns True.
        Subclasses can override this method to implement specific validation logic.
        """
        return True


class TaskCreated(_BaseEvent[_EventType.TaskCreated]):
    event_type: Literal[_EventType.TaskCreated] = _EventType.TaskCreated
    task_id: TaskId
    task: Task


class TaskDeleted(_BaseEvent[_EventType.TaskDeleted]):
    event_type: Literal[_EventType.TaskDeleted] = _EventType.TaskDeleted
    task_id: TaskId


class TaskStateUpdated(_BaseEvent[_EventType.TaskStateUpdated]):
    event_type: Literal[_EventType.TaskStateUpdated] = _EventType.TaskStateUpdated
    task_id: TaskId
    task_status: TaskStatus


class InstanceCreated(_BaseEvent[_EventType.InstanceCreated]):
    event_type: Literal[_EventType.InstanceCreated] = _EventType.InstanceCreated
    instance: Instance


class InstanceActivated(_BaseEvent[_EventType.InstanceActivated]):
    event_type: Literal[_EventType.InstanceActivated] = _EventType.InstanceActivated
    instance_id: InstanceId


class InstanceDeactivated(_BaseEvent[_EventType.InstanceDeactivated]):
    event_type: Literal[_EventType.InstanceDeactivated] = _EventType.InstanceDeactivated
    instance_id: InstanceId


class InstanceDeleted(_BaseEvent[_EventType.InstanceDeleted]):
    event_type: Literal[_EventType.InstanceDeleted] = _EventType.InstanceDeleted
    instance_id: InstanceId


class InstanceReplacedAtomically(_BaseEvent[_EventType.InstanceReplacedAtomically]):
    event_type: Literal[_EventType.InstanceReplacedAtomically] = _EventType.InstanceReplacedAtomically
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(_BaseEvent[_EventType.RunnerStatusUpdated]):
    event_type: Literal[_EventType.RunnerStatusUpdated] = _EventType.RunnerStatusUpdated
    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(_BaseEvent[_EventType.RunnerDeleted]):
    event_type: Literal[_EventType.RunnerDeleted] = _EventType.RunnerDeleted
    runner_id: RunnerId


class NodePerformanceMeasured(_BaseEvent[_EventType.NodePerformanceMeasured]):
    event_type: Literal[_EventType.NodePerformanceMeasured] = _EventType.NodePerformanceMeasured
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerStatusUpdated(_BaseEvent[_EventType.WorkerStatusUpdated]):
    event_type: Literal[_EventType.WorkerStatusUpdated] = _EventType.WorkerStatusUpdated
    node_id: NodeId
    node_state: NodeStatus


class ChunkGenerated(_BaseEvent[_EventType.ChunkGenerated]):
    event_type: Literal[_EventType.ChunkGenerated] = _EventType.ChunkGenerated
    command_id: CommandId
    chunk: GenerationChunk


class TopologyEdgeCreated(_BaseEvent[_EventType.TopologyEdgeCreated]):
    event_type: Literal[_EventType.TopologyEdgeCreated] = _EventType.TopologyEdgeCreated
    edge: Connection


class TopologyEdgeReplacedAtomically(_BaseEvent[_EventType.TopologyEdgeReplacedAtomically]):
    event_type: Literal[_EventType.TopologyEdgeReplacedAtomically] = _EventType.TopologyEdgeReplacedAtomically
    edge: Connection
    edge_profile: ConnectionProfile


class TopologyEdgeDeleted(_BaseEvent[_EventType.TopologyEdgeDeleted]):
    event_type: Literal[_EventType.TopologyEdgeDeleted] = _EventType.TopologyEdgeDeleted
    edge: Connection

_Event = Union[
    TaskCreated,
    TaskStateUpdated,
    TaskDeleted,
    InstanceCreated,
    InstanceActivated,
    InstanceDeactivated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    RunnerStatusUpdated,
    RunnerDeleted,
    NodePerformanceMeasured,
    WorkerStatusUpdated,
    ChunkGenerated,
    TopologyEdgeCreated,
    TopologyEdgeReplacedAtomically,
    TopologyEdgeDeleted,
]
"""
Un-annotated union of all events. Only used internally to create the registry.
For all other usecases, use the annotated union of events :class:`Event` :)
"""


def _check_event_type_consistency():
    # Grab enum values from members
    member_enum_values = [m for m in _EventType]

    # grab enum values from the union => scrape the type annotation
    union_enum_values: list[_EventType] = []
    union_classes = list(get_args(_Event))
    for cls in union_classes:  # pyright: ignore[reportAny]
        assert issubclass(cls, object), (
            f"{get_error_reporting_message()}",
            f"The class {cls} is NOT a subclass of {object}."
        )

        # ensure the first base parameter is ALWAYS _BaseEvent
        base_cls = list(types.get_original_bases(cls))
        assert len(base_cls) >= 1 and issubclass(base_cls[0], object) \
               and issubclass(base_cls[0], _BaseEvent), (
            f"{get_error_reporting_message()}",
            f"The class {cls} does NOT inherit from {_BaseEvent} {get_origin(base_cls[0])}."
        )

        # grab type hints and extract the right values from it
        cls_hints = get_type_hints(cls)
        assert "event_type" in cls_hints and \
               get_origin(cls_hints["event_type"]) is Literal, (  # pyright: ignore[reportAny]
            f"{get_error_reporting_message()}",
            f"The class {cls} is missing a {Literal}-annotated `event_type` field."
        )

        # make sure the value is an instance of `_EventType`
        enum_value = list(get_args(cls_hints["event_type"]))
        assert len(enum_value) == 1 and isinstance(enum_value[0], _EventType), (
            f"{get_error_reporting_message()}",
            f"The `event_type` of {cls} has a non-{_EventType} literal-type."
        )
        union_enum_values.append(enum_value[0])

    # ensure there is a 1:1 bijection between the two
    for m in member_enum_values:
        assert m in union_enum_values, (
            f"{get_error_reporting_message()}",
            f"There is no event-type registered for {m} in {_Event}."
        )
        union_enum_values.remove(m)
    assert len(union_enum_values) == 0, (
        f"{get_error_reporting_message()}",
        f"The following events have multiple event types defined in {_Event}: {union_enum_values}."
    )


_check_event_type_consistency()



Event = Annotated[_Event, Field(discriminator="event_type")]
"""Type of events, a discriminated union."""

# class TimerCreated(_BaseEvent[_EventType.TimerCreated]):
#     event_type: Literal[_EventType.TimerCreated] = _EventType.TimerCreated
#     timer_id: TimerId
#     delay_seconds: float
#
#
# class TimerFired(_BaseEvent[_EventType.TimerFired]):
#     event_type: Literal[_EventType.TimerFired] = _EventType.TimerFired
#     timer_id: TimerId