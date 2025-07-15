from enum import StrEnum
from typing import Any, Mapping, Type


def check_keys_in_map_match_enum_values[TEnum: StrEnum](
    mapping_type: Type[Mapping[Any, Any]],
    enum: Type[TEnum],
) -> None:
    mapping_keys = set(mapping_type.__annotations__.keys())
    category_values = set(e.value for e in enum)
    assert mapping_keys == category_values, (
        f"StateDomainMapping keys {mapping_keys} do not match EventCategories values {category_values}"
    )
