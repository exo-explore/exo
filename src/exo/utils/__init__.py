from typing import Any, Type

from .phantom import PhantomData


def ensure_type[T](obj: Any, expected_type: Type[T]) -> T:  # type: ignore
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(obj)}")  # type: ignore
    return obj


def todo[T](
    msg: str = "This code has not been implemented yet.",
    _phantom: PhantomData[T] = None,
) -> T:
    raise NotImplementedError(msg)
