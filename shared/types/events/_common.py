import types
import typing
from enum import Enum
from typing import TYPE_CHECKING

from shared.constants import get_error_reporting_message

from ._events import _Event  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    pass

from pydantic import BaseModel

from shared.types.common import NewUUID, NodeId


class EventId(NewUUID):
    """
    Newtype around `NewUUID`
    """


class CommandId(NewUUID):
    """
    Newtype around `NewUUID` for command IDs
    """


# Event base-class boilerplate (you should basically never touch these)
# Only very specialised registry or serialisation/deserialization logic might need know about these

class _EventType(str, Enum):
    """
    Here are all the unique kinds of events that can be sent over the network.
    """

    # Task Saga Events
    MLXInferenceSagaPrepare = "MLXInferenceSagaPrepare"
    MLXInferenceSagaStartPrepare = "MLXInferenceSagaStartPrepare"

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

    # Node Performance Events
    NodePerformanceMeasured = "NodePerformanceMeasured"

    # Topology Events
    TopologyEdgeCreated = "TopologyEdgeCreated"
    TopologyEdgeReplacedAtomically = "TopologyEdgeReplacedAtomically"
    TopologyEdgeDeleted = "TopologyEdgeDeleted"
    WorkerConnected = "WorkerConnected"
    WorkerStatusUpdated = "WorkerStatusUpdated"
    WorkerDisconnected = "WorkerDisconnected"

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
    


def _check_event_type_consistency():
    # Grab enum values from members
    member_enum_values = [m for m in _EventType]

    # grab enum values from the union => scrape the type annotation
    union_enum_values: list[_EventType] = []
    union_classes = list(typing.get_args(_Event))
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
            f"The class {cls} does NOT inherit from {_BaseEvent} {typing.get_origin(base_cls[0])}."
        )

        # grab type hints and extract the right values from it
        cls_hints = typing.get_type_hints(cls)
        assert "event_type" in cls_hints and \
               typing.get_origin(cls_hints["event_type"]) is typing.Literal, (  # pyright: ignore[reportAny]
            f"{get_error_reporting_message()}",
            f"The class {cls} is missing a {typing.Literal}-annotated `event_type` field."
        )

        # make sure the value is an instance of `_EventType`
        enum_value = list(typing.get_args(cls_hints["event_type"]))
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

