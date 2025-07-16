from types import UnionType
from typing import Annotated, Any, Mapping, Type, get_args

from pydantic import Field, TypeAdapter

from shared.constants import get_error_reporting_message
from shared.types.events.common import (
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
    Event,
    EventCategories,
    EventTypes,
    InstanceEventTypes,
    NodePerformanceEventTypes,
    RunnerStatusEventTypes,
    StreamingEventTypes,
    TaskEventTypes,
    TaskSagaEventTypes,
)
from shared.types.events.events import (
    ChunkGenerated,
    DataPlaneEdgeCreated,
    DataPlaneEdgeDeleted,
    DataPlaneEdgeReplacedAtomically,
    InstanceCreated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
    NodePerformanceMeasured,
    RunnerStatusUpdated,
    TaskCreated,
    TaskDeleted,
    TaskStateUpdated,
    WorkerConnected,
    WorkerDisconnected,
    WorkerStatusUpdated,
)

"""
class EventTypeNames(StrEnum):
    TaskEventType = auto()
    InstanceEvent = auto()
    NodePerformanceEvent = auto()
    ControlPlaneEvent = auto()
    StreamingEvent = auto()
    DataPlaneEvent = auto()
    TimerEvent = auto()
    MLXEvent = auto()

check_event_categories_are_defined_for_all_event_types(EVENT_TYPE_ENUMS, EventTypeNames)
"""
EventRegistry: Mapping[EventTypes, Type[Any]] = {
    TaskEventTypes.TaskCreated: TaskCreated,
    TaskEventTypes.TaskStateUpdated: TaskStateUpdated,
    TaskEventTypes.TaskDeleted: TaskDeleted,
    InstanceEventTypes.InstanceCreated: InstanceCreated,
    InstanceEventTypes.InstanceDeleted: InstanceDeleted,
    InstanceEventTypes.InstanceReplacedAtomically: InstanceReplacedAtomically,
    RunnerStatusEventTypes.RunnerStatusUpdated: RunnerStatusUpdated,
    NodePerformanceEventTypes.NodePerformanceMeasured: NodePerformanceMeasured,
    ControlPlaneEventTypes.WorkerConnected: WorkerConnected,
    ControlPlaneEventTypes.WorkerStatusUpdated: WorkerStatusUpdated,
    ControlPlaneEventTypes.WorkerDisconnected: WorkerDisconnected,
    StreamingEventTypes.ChunkGenerated: ChunkGenerated,
    DataPlaneEventTypes.DataPlaneEdgeCreated: DataPlaneEdgeCreated,
    DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically: DataPlaneEdgeReplacedAtomically,
    DataPlaneEventTypes.DataPlaneEdgeDeleted: DataPlaneEdgeDeleted,
    TaskSagaEventTypes.MLXInferenceSagaPrepare: MLXInferenceSagaPrepare,
    TaskSagaEventTypes.MLXInferenceSagaStartPrepare: MLXInferenceSagaStartPrepare,
}


# Sanity Check.
def check_registry_has_all_event_types() -> None:
    event_types: tuple[EventTypes, ...] = get_args(EventTypes)
    missing_event_types = set(event_types) - set(EventRegistry.keys())

    assert not missing_event_types, (
        f"{get_error_reporting_message()}"
        f"There's an event missing from the registry: {missing_event_types}"
    )


def check_union_of_all_events_is_consistent_with_registry(
    registry: Mapping[EventTypes, Type[Any]], union_type: UnionType
) -> None:
    type_of_each_registry_entry = set(registry.values())
    type_of_each_entry_in_union = set(get_args(union_type))
    missing_from_union = type_of_each_registry_entry - type_of_each_entry_in_union

    assert not missing_from_union, (
        f"{get_error_reporting_message()}"
        f"Event classes in registry are missing from all_events union: {missing_from_union}"
    )

    extra_in_union = type_of_each_entry_in_union - type_of_each_registry_entry

    assert not extra_in_union, (
        f"{get_error_reporting_message()}"
        f"Event classes in all_events union are missing from registry: {extra_in_union}"
    )


AllEvents = (
    TaskCreated
    | TaskStateUpdated
    | TaskDeleted
    | InstanceCreated
    | InstanceDeleted
    | InstanceReplacedAtomically
    | RunnerStatusUpdated
    | NodePerformanceMeasured
    | WorkerConnected
    | WorkerStatusUpdated
    | WorkerDisconnected
    | ChunkGenerated
    | DataPlaneEdgeCreated
    | DataPlaneEdgeReplacedAtomically
    | DataPlaneEdgeDeleted
    | MLXInferenceSagaPrepare
    | MLXInferenceSagaStartPrepare
)

# Run the sanity check
check_union_of_all_events_is_consistent_with_registry(EventRegistry, AllEvents)


_EventType = Annotated[AllEvents, Field(discriminator="event_type")]
EventParser: TypeAdapter[Event[EventCategories]] = TypeAdapter(_EventType)
