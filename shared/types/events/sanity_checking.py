from enum import StrEnum
from types import UnionType
from typing import Any, Mapping, Set, Type, cast, get_args

from pydantic.fields import FieldInfo

from shared.constants import get_error_reporting_message
from shared.types.events.common import EventType


def assert_event_union_covers_registry[TEnum: StrEnum](
    literal_union: UnionType,
) -> None:
    """
    Ensure that our union of events (AllEventsUnion) has one member per element of Enum
    """
    enum_values: Set[str] = {member.value for member in EventType}

    def _flatten(tp: UnionType) -> Set[str]:
        values: Set[str] = set()
        args = get_args(tp)  # Get event classes from the union
        for arg in args:  # type: ignore[reportAny]
            # Cast to type since we know these are class types
            event_class = cast(type[Any], arg)
            # Each event class is a Pydantic model with model_fields
            if hasattr(event_class, 'model_fields'):
                model_fields = cast(dict[str, FieldInfo], event_class.model_fields)
                if 'event_type' in model_fields:
                    # Get the default value of the event_type field
                    event_type_field: FieldInfo = model_fields['event_type']
                    if hasattr(event_type_field, 'default'):
                        default_value = cast(EventType, event_type_field.default)
                        # The default is an EventType enum member, get its value
                        values.add(default_value.value)
        return values

    literal_values: Set[str] = _flatten(literal_union)

    assert enum_values == literal_values, (
        f"{get_error_reporting_message()}"
        f"The values of the enum {EventType} are not covered by the literal union {literal_union}.\n"
        f"These are the missing values: {enum_values - literal_values}\n"
        f"These are the extra values: {literal_values - enum_values}\n"
    )

def check_union_of_all_events_is_consistent_with_registry(
    registry: Mapping[EventType, Type[Any]], union_type: UnionType
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

def check_registry_has_all_event_types(event_registry: Mapping[EventType, Type[Any]]) -> None:
    event_types: tuple[EventType, ...] = get_args(EventType)
    missing_event_types = set(event_types) - set(event_registry.keys())

    assert not missing_event_types, (
        f"{get_error_reporting_message()}"
        f"There's an event missing from the registry: {missing_event_types}"
    )

# TODO: Check all events have an apply function.
# probably in a different place though.