# components.py defines the small event functions, adapters etc. 
# this name could probably be improved.

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    pass

from typing import Callable

from pydantic import BaseModel, Field, model_validator

from shared.types.common import NodeId
from shared.types.events._events import Event
from shared.types.state import State


class EventFromEventLog[T: Event](BaseModel):
    event: T
    origin: NodeId
    idx_in_log: int = Field(gt=0)

    @model_validator(mode="after")
    def check_event_was_sent_by_correct_node(
        self,
    ) -> "EventFromEventLog[T]":
        if self.event.check_event_was_sent_by_correct_node(self.origin):
            return self
        raise ValueError("Invalid Event: Origin ID Does Not Match")



type Apply = Callable[
    [State, Event],
    State
]