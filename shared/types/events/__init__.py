# ruff: noqa: F403
# ruff: noqa: F405

# Note: we are implementing internal details here, so importing private stuff is fine!!!
from pydantic import TypeAdapter

from ._events import *
from .components import EventFromEventLog

EventParser: TypeAdapter[Event] = TypeAdapter(Event)
"""Type adaptor to parse :class:`Event`s."""

__all__ = ["Event", "EventParser", "EventFromEventLog"]
