from collections.abc import Callable, Iterable
from typing import TypeGuard


def fmap[T, U](c: Callable[[T], U | None], i: Iterable[T]) -> Iterable[U]:
    def is_real(it: U | None) -> TypeGuard[U]:
        return it is not None

    return filter(is_real, map(c, i))
