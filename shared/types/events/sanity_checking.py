from enum import Enum, StrEnum
from types import UnionType
from typing import Any, LiteralString, Sequence, Set, Type, get_args

from shared.constants import get_error_reporting_message


def check_event_type_union_is_consistent_with_registry(
    event_type_enums: Sequence[Type[Enum]], event_types: UnionType
) -> None:
    """Assert that every enum value from _EVENT_TYPE_ENUMS satisfies EventTypes."""

    event_types_inferred_from_union = set(get_args(event_types))

    event_types_inferred_from_registry = [
        member for enum_class in event_type_enums for member in enum_class
    ]

    # Check that each registry value belongs to one of the types in the union
    for tag_of_event_type in event_types_inferred_from_registry:
        event_type = type(tag_of_event_type)
        assert event_type in event_types_inferred_from_union, (
            f"{get_error_reporting_message()}"
            f"There's a mismatch between the registry of event types and the union of possible event types."
            f"The enum value {tag_of_event_type} for type {event_type} is not covered by {event_types_inferred_from_union}."
        )


def check_event_categories_are_defined_for_all_event_types(
    event_definitions: Sequence[Type[Enum]], event_categories: Type[StrEnum]
) -> None:
    """Assert that the event category names are consistent with the event type enums."""

    expected_category_tags: list[str] = [
        enum_class.__name__ for enum_class in event_definitions
    ]
    tag_of_event_categories: list[str] = list(event_categories.__members__.values())
    assert tag_of_event_categories == expected_category_tags, (
        f"{get_error_reporting_message()}"
        f"The values of the enum EventCategories are not named after the event type enums."
        f"These are the missing categories: {set(expected_category_tags) - set(tag_of_event_categories)}"
        f"These are the extra categories: {set(tag_of_event_categories) - set(expected_category_tags)}"
    )


def assert_literal_union_covers_enum[TEnum: StrEnum](
    literal_union: UnionType,
    enum_type: Type[TEnum],
) -> None:
    enum_values: Set[Any] = {member.value for member in enum_type}

    def _flatten(tp: UnionType) -> Set[Any]:
        values: Set[Any] = set()
        args: tuple[LiteralString, ...] = get_args(tp)
        for arg in args:
            payloads: tuple[TEnum, ...] = get_args(arg)
            for payload in payloads:
                values.add(payload.value)
        return values

    literal_values: Set[Any] = _flatten(literal_union)

    assert enum_values == literal_values, (
        f"{get_error_reporting_message()}"
        f"The values of the enum {enum_type} are not covered by the literal union {literal_union}.\n"
        f"These are the missing values: {enum_values - literal_values}\n"
        f"These are the extra values: {literal_values - enum_values}\n"
    )
