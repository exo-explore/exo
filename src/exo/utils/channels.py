from math import inf
from typing import Self

from anyio import ClosedResourceError, WouldBlock
from anyio.streams.memory import (
    MemoryObjectReceiveStream as AnyioReceiver,
)
from anyio.streams.memory import (
    MemoryObjectSendStream as AnyioSender,
)
from anyio.streams.memory import (
    MemoryObjectStreamState as AnyioState,
)


class Sender[T](AnyioSender[T]):
    def clone_receiver(self) -> "Receiver[T]":
        """Constructs a Receiver using a Senders shared state - similar to calling Receiver.clone() without needing the receiver"""
        if self._closed:
            raise ClosedResourceError
        return Receiver(_state=self._state)


class Receiver[T](AnyioReceiver[T]):
    def clone_sender(self) -> Sender[T]:
        """Constructs a Sender using a Receivers shared state - similar to calling Sender.clone() without needing the sender"""
        if self._closed:
            raise ClosedResourceError
        return Sender(_state=self._state)

    def collect(self) -> list[T]:
        """Collect all currently available items from this receiver"""
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out

    async def receive_at_least(self, n: int) -> list[T]:
        out: list[T] = []
        out.append(await self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(await self.receive())
            out.extend(self.collect())
        return out

    def __enter__(self) -> Self:
        return self


class channel[T]:  # noqa: N801
    def __new__(cls, max_buffer_size: float = inf) -> tuple[Sender[T], Receiver[T]]:
        if max_buffer_size != inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        state = AnyioState[T](max_buffer_size)
        return Sender(_state=state), Receiver(_state=state)
