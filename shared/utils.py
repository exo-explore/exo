from typing import Any, Type, TypeVar

T = TypeVar("T")


def ensure_type(obj: Any, expected_type: Type[T]) -> T:  # type: ignore
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(obj)}")  # type: ignore
    return obj
