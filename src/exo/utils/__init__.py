from typing import Any, Callable, Iterable, Iterator, Type, TypeGuard

from .phantom import PhantomData

STDIN_FD = 0
STDOUT_FD = 1
STDERR_FD = 2
STDIO_FDS = (STDIN_FD, STDOUT_FD, STDERR_FD)


def ensure_type[T](obj: Any, expected_type: Type[T]) -> T:  # type: ignore
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(obj)}")  # type: ignore
    return obj


def todo[T](
    msg: str = "This code has not been implemented yet.",
    _phantom: PhantomData[T] = None,
) -> T:
    raise NotImplementedError(msg)


def not_none[T](t: T | None) -> TypeGuard[T]:
    return t is not None


def fmap[T, U](f: Callable[[T], U | None], s: Iterable[T]) -> Iterator[U]:
    return filter(not_none, map(f, s))
