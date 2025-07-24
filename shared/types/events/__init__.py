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

# Event type consistency check - runs after all imports are complete
def _check_event_type_consistency():
    import types
    import typing

    from shared.constants import get_error_reporting_message

    from ._common import _BaseEvent, _EventType  # pyright: ignore[reportPrivateUsage]
    from ._events import _Event  # pyright: ignore[reportPrivateUsage]
    
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
