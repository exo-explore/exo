from collections.abc import Callable, Iterable
from typing import Any, Type, TypeGuard

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


def fold[T, U](acc: T, fn: Callable[[T, U], T], iterator: Iterable[U]) -> T:
    for it in iterator:
        acc = fn(acc, it)
    return acc


def _filter_none[U](item: U | None) -> TypeGuard[U]:
    return item is not None


def fmap[T, U](fn: Callable[[T], U | None], iterator: Iterable[T]) -> Iterable[U]:
    return filter(_filter_none, map(fn, iterator))
