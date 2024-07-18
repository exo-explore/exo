import os
import asyncio
from typing import Any, Callable, Coroutine, TypeVar, Optional, Dict, Generic, Tuple

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

T = TypeVar('T')
K = TypeVar('K')

class AsyncCallback(Generic[T]):
    def __init__(self) -> None:
        self.condition: asyncio.Condition = asyncio.Condition()
        self.result: Optional[Tuple[T, ...]] = None
        self.observers: list[Callable[..., None]] = []

    async def wait(self,
                   check_condition: Callable[..., bool],
                   timeout: Optional[float] = None) -> Tuple[T, ...]:
        async with self.condition:
            await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
            assert self.result is not None  # for type checking
            return self.result

    def on_next(self, callback: Callable[..., None]) -> None:
        self.observers.append(callback)

    def set(self, *args: T) -> None:
        self.result = args
        for observer in self.observers:
            observer(*args)
        asyncio.create_task(self.notify())

    async def notify(self) -> None:
        async with self.condition:
            self.condition.notify_all()

class AsyncCallbackSystem(Generic[K, T]):
    def __init__(self) -> None:
        self.callbacks: Dict[K, AsyncCallback[T]] = {}

    def register(self, name: K) -> AsyncCallback[T]:
        if name not in self.callbacks:
            self.callbacks[name] = AsyncCallback[T]()
        return self.callbacks[name]

    def deregister(self, name: K) -> None:
        if name in self.callbacks:
            del self.callbacks[name]

    def trigger(self, name: K, *args: T) -> None:
        if name in self.callbacks:
            self.callbacks[name].set(*args)

    def trigger_all(self, *args: T) -> None:
        for callback in self.callbacks.values():
            callback.set(*args)
