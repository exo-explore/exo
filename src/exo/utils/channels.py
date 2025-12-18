import multiprocessing as mp
from dataclasses import dataclass, field
from math import inf
from multiprocessing.synchronize import Event
from queue import Empty, Full
from types import TracebackType
from typing import Self

from anyio import (
    CapacityLimiter,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    to_thread,
)
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
    def clone(self) -> "Sender[T]":
        if self._closed:
            raise ClosedResourceError
        return Sender(_state=self._state)

    def clone_receiver(self) -> "Receiver[T]":
        """Constructs a Receiver using a Senders shared state - similar to calling Receiver.clone() without needing the receiver"""
        if self._closed:
            raise ClosedResourceError
        return Receiver(_state=self._state)


class Receiver[T](AnyioReceiver[T]):
    def clone(self) -> "Receiver[T]":
        if self._closed:
            raise ClosedResourceError
        return Receiver(_state=self._state)

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


class _MpEndOfStream:
    pass


class MpState[T]:
    def __init__(self, max_buffer_size: float):
        if max_buffer_size == inf:
            max_buffer_size = 0
        assert isinstance(max_buffer_size, int), (
            "State should only ever be constructed with an integer or math.inf size."
        )

        self.max_buffer_size: float = max_buffer_size
        self.buffer: mp.Queue[T | _MpEndOfStream] = mp.Queue(max_buffer_size)
        self.closed: Event = mp.Event()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpSender[T]:
    """
    An interprocess channel, mimicing the Anyio structure.
    It should be noted that none of the clone methods are implemented for simplicity, for now.
    """

    _state: MpState[T] = field()

    def send_nowait(self, item: T) -> None:
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self._state.buffer.put(item, block=False)
        except Full:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def send(self, item: T) -> None:
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self.send_nowait(item)
        except WouldBlock:
            # put anyway, blocking
            self._state.buffer.put(item, block=True)

    async def send_async(self, item: T) -> None:
        await to_thread.run_sync(self.send, item, limiter=CapacityLimiter(1))

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        self._state.buffer.put(_MpEndOfStream())
        self._state.buffer.close()

    # == unique to Mp channels ==
    def join(self) -> None:
        """Ensure any queued messages are resolved before continuing"""
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # == context manager support ==
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpReceiver[T]:
    """
    An interprocess channel, mimicing the Anyio structure.
    It should be noted that none of the clone methods are implemented for simplicity, for now.
    """

    _state: MpState[T] = field()

    def receive_nowait(self) -> T:
        if self._state.closed.is_set():
            raise ClosedResourceError

        try:
            item = self._state.buffer.get(block=False)
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream
            return item
        except Empty:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def receive(self) -> T:
        try:
            return self.receive_nowait()
        except WouldBlock:
            item = self._state.buffer.get()
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream from None
            return item

    # nb: this function will not cancel particularly well
    async def receive_async(self) -> T:
        return await to_thread.run_sync(self.receive, limiter=CapacityLimiter(1))

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        self._state.buffer.close()

    # == unique to Mp channels ==
    def join(self) -> None:
        """Block until all enqueued messages are drained off our side of the buffer"""
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # == iterator support ==
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            return self.receive()
        except EndOfStream:
            raise StopIteration from None

    # == async iterator support ==
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive_async()
        except EndOfStream:
            raise StopAsyncIteration from None

    # == context manager support ==
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

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

    def receive_at_least(self, n: int) -> list[T]:
        out: list[T] = []
        out.append(self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(self.receive())
            out.extend(self.collect())
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


class channel[T]:  # noqa: N801
    """Create a pair of asynchronous channels for communicating within the same process"""

    def __new__(cls, max_buffer_size: float = inf) -> tuple[Sender[T], Receiver[T]]:
        if max_buffer_size != inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        state = AnyioState[T](max_buffer_size)
        return Sender(_state=state), Receiver(_state=state)


class mp_channel[T]:  # noqa: N801
    """Create a pair of synchronous channels for interprocess communication"""

    # max buffer size uses math.inf to represent an unbounded queue, and 0 to represent a yet unimplemented "unbuffered" queue.
    def __new__(cls, max_buffer_size: float = inf) -> tuple[MpSender[T], MpReceiver[T]]:
        if (
            max_buffer_size == 0
            or max_buffer_size != inf
            and not isinstance(max_buffer_size, int)
        ):
            raise ValueError(
                "max_buffer_size must be either an integer or math.inf. 0-sized buffers are not supported by multiprocessing"
            )
        state = MpState[T](max_buffer_size)
        return MpSender(_state=state), MpReceiver(_state=state)
