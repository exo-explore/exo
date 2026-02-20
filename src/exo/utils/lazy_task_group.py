# pyright: reportAny=false
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Unpack

from anyio import CancelScope, create_task_group
from anyio.abc import TaskGroup


class LazyTaskGroup(TaskGroup):
    lazy: TaskGroup | None
    queued: list[tuple[Any, Any, Any]]

    def __init__(self) -> None:
        self.queued = []
        self.lazy = None
        self.cancel_scope = CancelScope()

    async def start(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: object,
        name: object = None,
    ) -> Any:
        if self.lazy is None:
            raise RuntimeError(
                "LazyTaskGroup.start() requires an active task group; "
                "use start_soon() before entering the context."
            )

        return await self.lazy.start(func, *args, name=name)

    def start_soon[*T](
        self,
        func: Callable[[Unpack[T]], Awaitable[Any]],
        *args: Unpack[T],
        name: object = None,
    ) -> None:
        if self.lazy is None:
            self.queued.append((func, args, name))
        else:
            self.lazy.start_soon(func, *args, name=name)

    async def __aenter__(self) -> TaskGroup:
        self.lazy = create_task_group()
        r = await self.lazy.__aenter__()
        if self.cancel_scope.cancel_called:
            r.cancel_scope.cancel()
        self.cancel_scope = r.cancel_scope
        for func, args, name in self.queued:
            self.lazy.start_soon(func, *args, name=name)
        self.queued.clear()
        return r

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the task group context waiting for all tasks to finish."""
        assert self.lazy is not None, (
            "aenter sets self.lazy, so it exists when we aexit"
        )
        return await self.lazy.__aexit__(exc_type, exc_val, exc_tb)
