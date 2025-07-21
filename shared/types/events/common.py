from enum import Enum, StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Literal,
    NamedTuple,
    Protocol,
    Sequence,
    cast,
)

if TYPE_CHECKING:
    pass

from pydantic import BaseModel, Field, model_validator

from shared.types.common import NewUUID, NodeId
from shared.types.events.sanity_checking import (
    assert_literal_union_covers_enum,
    check_event_type_union_is_consistent_with_registry,
)


class EventId(NewUUID):
    pass


class TimerId(NewUUID):
    pass


# Here are all the unique kinds of events that can be sent over the network.
# I've defined them in different enums for clarity, but they're all part of the same set of possible events.
class TaskSagaEventTypes(str, Enum):
    MLXInferenceSagaPrepare = "MLXInferenceSagaPrepare"
    MLXInferenceSagaStartPrepare = "MLXInferenceSagaStartPrepare"


class TaskEventTypes(str, Enum):
    TaskCreated = "TaskCreated"
    TaskStateUpdated = "TaskStateUpdated"
    TaskDeleted = "TaskDeleted"


class StreamingEventTypes(str, Enum):
    ChunkGenerated = "ChunkGenerated"


class InstanceEventTypes(str, Enum):
    InstanceCreated = "InstanceCreated"
    InstanceDeleted = "InstanceDeleted"
    InstanceActivated = "InstanceActivated"
    InstanceDeactivated = "InstanceDeactivated"
    InstanceReplacedAtomically = "InstanceReplacedAtomically"


class RunnerStatusEventTypes(str, Enum):
    RunnerStatusUpdated = "RunnerStatusUpdated"


class NodePerformanceEventTypes(str, Enum):
    NodePerformanceMeasured = "NodePerformanceMeasured"


class TopologyEventTypes(str, Enum):
    TopologyEdgeCreated = "TopologyEdgeCreated"
    TopologyEdgeReplacedAtomically = "TopologyEdgeReplacedAtomically"
    TopologyEdgeDeleted = "TopologyEdgeDeleted"
    WorkerConnected = "WorkerConnected"
    WorkerStatusUpdated = "WorkerStatusUpdated"
    WorkerDisconnected = "WorkerDisconnected"


class TimerEventTypes(str, Enum):
    TimerCreated = "TimerCreated"
    TimerFired = "TimerFired"


# Registry of all event type enums
EVENT_TYPE_ENUMS = [
    TaskEventTypes,
    StreamingEventTypes,
    InstanceEventTypes,
    RunnerStatusEventTypes,
    NodePerformanceEventTypes,
    TopologyEventTypes,
    TimerEventTypes,
    TaskSagaEventTypes,
]


# Here's the set of all possible events.
EventTypes = (
    TaskEventTypes
    | StreamingEventTypes
    | InstanceEventTypes
    | RunnerStatusEventTypes
    | NodePerformanceEventTypes
    | TopologyEventTypes
    | TimerEventTypes
    | TaskSagaEventTypes
)


check_event_type_union_is_consistent_with_registry(EVENT_TYPE_ENUMS, EventTypes)


class EventCategoryEnum(StrEnum):
    MutatesTaskState = "MutatesTaskState"
    MutatesTaskSagaState = "MutatesTaskSagaState"
    MutatesRunnerStatus = "MutatesRunnerStatus"
    MutatesInstanceState = "MutatesInstanceState"
    MutatesNodePerformanceState = "MutatesNodePerformanceState"
    MutatesTopologyState = "MutatesTopologyState"


EventCategory = (
    Literal[EventCategoryEnum.MutatesTopologyState]
    | Literal[EventCategoryEnum.MutatesTaskState]
    | Literal[EventCategoryEnum.MutatesTaskSagaState]
    | Literal[EventCategoryEnum.MutatesRunnerStatus]
    | Literal[EventCategoryEnum.MutatesInstanceState]
    | Literal[EventCategoryEnum.MutatesNodePerformanceState]
    | Literal[EventCategoryEnum.MutatesTopologyState]
)

EventCategories = FrozenSet[EventCategory]

assert_literal_union_covers_enum(EventCategory, EventCategoryEnum)


EventTypeT = EventTypes  # Type Alias placeholder; generic parameter will override


class BaseEvent[
    SetMembersT: EventCategories | EventCategory,
    EventTypeLitT: EventTypes = EventTypes,
](BaseModel):
    event_type: EventTypeLitT
    event_category: SetMembersT
    event_id: EventId = EventId()

    def check_event_was_sent_by_correct_node(self, origin_id: NodeId) -> bool:
        """Check if the event was sent by the correct node.
        
        This is a placeholder implementation that always returns True.
        Subclasses can override this method to implement specific validation logic.
        """
        return True


class EventFromEventLog[SetMembersT: EventCategories | EventCategory](BaseModel):
    event: BaseEvent[SetMembersT]
    origin: NodeId
    idx_in_log: int = Field(gt=0)

    @model_validator(mode="after")
    def check_event_was_sent_by_correct_node(
        self,
    ) -> "EventFromEventLog[SetMembersT]":
        if self.event.check_event_was_sent_by_correct_node(self.origin):
            return self
        raise ValueError("Invalid Event: Origin ID Does Not Match")


def narrow_event_type[T: EventCategory, Q: EventCategories | EventCategory](
    event: BaseEvent[Q],
    target_category: T,
) -> BaseEvent[T]:
    if target_category not in event.event_category:
        raise ValueError(f"Event Does Not Contain Target Category {target_category}")

    narrowed_event = event.model_copy(update={"event_category": {target_category}})
    return cast(BaseEvent[T], narrowed_event)


def narrow_event_from_event_log_type[
    T: EventCategory,
    Q: EventCategories | EventCategory,
](
    event: EventFromEventLog[Q],
    target_category: T,
) -> EventFromEventLog[T]:
    if target_category not in event.event.event_category:
        raise ValueError(f"Event Does Not Contain Target Category {target_category}")
    narrowed_event = event.model_copy(
        update={"event": narrow_event_type(event.event, target_category)}
    )

    return cast(EventFromEventLog[T], narrowed_event)


class State[EventCategoryT: EventCategory](BaseModel):
    event_category: EventCategoryT
    last_event_applied_idx: int = Field(default=0, ge=0)


# Definitions for Type Variables
type Saga[EventCategoryT: EventCategory] = Callable[
    [State[EventCategoryT], EventFromEventLog[EventCategoryT]],
    Sequence[BaseEvent[EventCategories]],
]
type Apply[EventCategoryT: EventCategory] = Callable[
    [State[EventCategoryT], EventFromEventLog[EventCategoryT]],
    State[EventCategoryT],
]


class StateAndEvent[EventCategoryT: EventCategory](NamedTuple):
    state: State[EventCategoryT]
    event: EventFromEventLog[EventCategoryT]


type EffectHandler[EventCategoryT: EventCategory] = Callable[
    [StateAndEvent[EventCategoryT], State[EventCategoryT]], None
]
type EventPublisher = Callable[[BaseEvent[Any]], None]


# A component that can publish events
class EventPublisherProtocol(Protocol):
    def send(self, events: Sequence[BaseEvent[EventCategories]]) -> None: ...


# A component that can fetch events to apply
class EventFetcherProtocol[EventCategoryT: EventCategory](Protocol):
    def get_events_to_apply(
        self, state: State[EventCategoryT]
    ) -> Sequence[BaseEvent[EventCategoryT]]: ...


# A component that can get the effect handler for a saga
def get_saga_effect_handler[EventCategoryT: EventCategory](
    saga: Saga[EventCategoryT], event_publisher: EventPublisher
) -> EffectHandler[EventCategoryT]:
    def effect_handler(state_and_event: StateAndEvent[EventCategoryT]) -> None:
        trigger_state, trigger_event = state_and_event
        for event in saga(trigger_state, trigger_event):
            event_publisher(event)

    return lambda state_and_event, _: effect_handler(state_and_event)


def get_effects_from_sagas[EventCategoryT: EventCategory](
    sagas: Sequence[Saga[EventCategoryT]],
    event_publisher: EventPublisher,
) -> Sequence[EffectHandler[EventCategoryT]]:
    return [get_saga_effect_handler(saga, event_publisher) for saga in sagas]


type IdemKeyGenerator[EventCategoryT: EventCategory] = Callable[
    [State[EventCategoryT], int], Sequence[EventId]
]


class CommandId(NewUUID):
    pass


class CommandTypes(str, Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class Command[
    EventCategoryT: EventCategories | EventCategory,
    CommandType: CommandTypes,
](BaseModel):
    command_type: CommandType
    command_id: CommandId


type Decide[EventCategoryT: EventCategory, CommandTypeT: CommandTypes] = Callable[
    [State[EventCategoryT], Command[EventCategoryT, CommandTypeT]],
    Sequence[BaseEvent[EventCategoryT]],
]
