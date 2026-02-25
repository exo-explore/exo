from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Unpack

from anyio import create_task_group
from anyio.abc import TaskGroup as TaskGroupABC


@dataclass
class TaskGroup:
    _tg: TaskGroupABC | None = field(default=None, init=False)
    _queued: list[tuple[Any, Any, Any]] | None = field(default_factory=list, init=False)

    def is_running(self) -> bool:
        return self._tg is not None

    def cancel_tasks(self):
        assert self._tg
        self._tg.cancel_scope.cancel()

    def cancel_called(self) -> bool:
        assert self._tg
        return self._tg.cancel_scope.cancel_called

    def start_soon[*T](
        self,
        func: Callable[[Unpack[T]], Awaitable[Any]],
        *args: Unpack[T],
        name: object = None,
    ) -> None:
        assert self._tg is not None
        assert self._queued is None
        self._tg.start_soon(func, *args, name=name)

    def queue[*T](
        self,
        func: Callable[[Unpack[T]], Awaitable[Any]],
        *args: Unpack[T],
        name: object = None,
    ) -> None:
        assert self._tg is None
        assert self._queued is not None
        self._queued.append((func, args, name))

    async def __aenter__(self) -> TaskGroupABC:
        assert self._tg is None
        assert self._queued is not None
        self._tg = create_task_group()
        r = await self._tg.__aenter__()
        for func, args, name in self._queued:  # pyright: ignore[reportAny]
            self._tg.start_soon(func, *args, name=name)  # pyright: ignore[reportAny]
        self._queued = None
        return r

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the task group context waiting for all tasks to finish."""
        assert self._tg is not None, "aenter sets self.lazy, so it exists when we aexit"
        assert self._queued is None
        return await self._tg.__aexit__(exc_type, exc_val, exc_tb)
