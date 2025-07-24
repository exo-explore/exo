# ruff: noqa: F403
# ruff: noqa: F405

# Note: we are implementing internal details here, so importing private stuff is fine!!!
from pydantic import TypeAdapter

from shared.types.events.components import EventFromEventLog

from ._apply import Event, apply
from ._common import *
from ._events import *

EventParser: TypeAdapter[Event] = TypeAdapter(Event)
"""Type adaptor to parse :class:`Event`s."""

__all__ = ["Event", "EventParser", "apply", "EventFromEventLog"]
