import contextlib
import multiprocessing as mp
import pickle
import select
from dataclasses import dataclass, field
from functools import wraps
from inspect import iscoroutinefunction
from math import inf
from multiprocessing.reduction import ForkingPickler
from multiprocessing.synchronize import Event
from queue import Empty, Full
from types import CoroutineType, TracebackType
from typing import Any, Callable, NoReturn, Protocol, Self, cast, overload, override

from anyio import (
    BrokenResourceError,
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

try:
    from anyio.streams.memory import (
        MemoryObjectStreamState,
    )
    from anyio.streams.memory import (
        MemoryObjectStreamState as AnyioState,
    )
except ImportError:

    class MemoryObjectStreamState:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: object) -> object:
            return cls

    AnyioState = MemoryObjectStreamState  # type: ignore[misc, assignment]


@dataclass(eq=False)
class ErrorOverride:
    closed_resource_error: type[ClosedResourceError] = field(
        default=ClosedResourceError,
    )
    broken_resource_error: type[BrokenResourceError] = field(
        default=BrokenResourceError,
    )
    end_of_stream: type[EndOfStream] = field(
        default=EndOfStream,
    )
    would_block: type[WouldBlock] = field(
        default=WouldBlock,
    )

    @overload
    def patch[**P, R](
        self,
        fn: Callable[P, CoroutineType[Any, Any, R]],
        /,
    ) -> Callable[P, CoroutineType[Any, Any, R]]: ...

    @overload
    def patch[**P, R](
        self,
        fn: Callable[P, R],
        /,
    ) -> Callable[P, R]: ...

    def patch[**P, R](self, fn: Callable[P, Any], /) -> Callable[P, Any]:
        """
        Returns a function with all these exceptions replaced by their overrides
        """

        if iscoroutinefunction(fn):
            async_fn = cast(Callable[P, CoroutineType[Any, Any, R]], fn)

            @wraps(async_fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return await async_fn(*args, **kwargs)
                except ClosedResourceError as e:
                    self._raise_replace(self.closed_resource_error, e)
                except BrokenResourceError as e:
                    self._raise_replace(self.broken_resource_error, e)
                except EndOfStream as e:
                    self._raise_replace(self.end_of_stream, e)
                except WouldBlock as e:
                    self._raise_replace(self.would_block, e)

            return async_wrapper
        else:
            sync_fn = cast(Callable[P, R], fn)

            @wraps(sync_fn)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return sync_fn(*args, **kwargs)
                except ClosedResourceError as e:
                    self._raise_replace(self.closed_resource_error, e)
                except BrokenResourceError as e:
                    self._raise_replace(self.broken_resource_error, e)
                except EndOfStream as e:
                    self._raise_replace(self.end_of_stream, e)
                except WouldBlock as e:
                    self._raise_replace(self.would_block, e)

            return sync_wrapper

    @staticmethod
    def _raise_replace(replacement: type[BaseException], e: BaseException) -> NoReturn:
        if isinstance(e, replacement):
            raise
        raise replacement() from e


class Sender[T](AnyioSender[T]):
    def __init__(
        self,
        state: MemoryObjectStreamState[T],
        error_override_config: ErrorOverride | None,
    ):
        super().__init__(_state=state)

        # patch the methods we want to override errors for
        #
        # NOTE: it is very important that new methods which are added,
        #       and which can throw, are patched in this block
        if (e := error_override_config) is not None:
            # new methods of this class
            self.clone_receiver = e.patch(self.clone_receiver)

            # overridden methods
            self.clone = e.patch(self.clone)

            # parent methods
            self.send_nowait = e.patch(self.send_nowait)
            self.send = e.patch(self.send)
            self.close = e.patch(self.close)
            self.aclose = e.patch(self.aclose)
            self.statistics = e.patch(self.statistics)

        self.err_config = error_override_config

    @override
    def clone(self) -> "Sender[T]":
        if self._closed:
            raise ClosedResourceError
        return Sender(self._state, self.err_config)

    def clone_receiver(self) -> "Receiver[T]":
        """Constructs a Receiver using a Senders shared state - similar to calling Receiver.clone() without needing the receiver"""
        if self._closed:
            raise ClosedResourceError
        return Receiver(self._state, self.err_config)


class Receiver[T](AnyioReceiver[T]):
    def __init__(
        self,
        state: MemoryObjectStreamState[T],
        error_override_config: ErrorOverride | None,
    ):
        super().__init__(_state=state)

        # patch the methods we want to override errors for
        #
        # NOTE: it is very important that new methods which are added,
        #       and which can throw, are patched in this block
        if (e := error_override_config) is not None:
            # new methods of this class
            self.clone_sender = e.patch(self.clone_sender)
            self.collect = e.patch(self.collect)
            self.receive_at_least = e.patch(self.receive_at_least)

            # overridden methods
            self.clone = e.patch(self.clone)

            # parent methods
            self.receive_nowait = e.patch(self.receive_nowait)
            self.receive = e.patch(self.receive)
            self.close = e.patch(self.close)
            self.aclose = e.patch(self.aclose)
            self.statistics = e.patch(self.statistics)

        self.err_config = error_override_config

    @override
    def clone(self) -> "Receiver[T]":
        if self._closed:
            raise ClosedResourceError
        return Receiver(self._state, self.err_config)

    def clone_sender(self) -> Sender[T]:
        """Constructs a Sender using a Receivers shared state - similar to calling Sender.clone() without needing the sender"""
        if self._closed:
            raise ClosedResourceError
        return Sender(self._state, self.err_config)

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

    @override
    def __enter__(self) -> Self:
        return self


class _MpEndOfStream:
    pass


class _MpBuffer[T](Protocol):
    """The part of multiprocessing.Queue that the Mp channels use"""

    def put(self, obj: T, block: bool = True, timeout: float | None = None) -> None: ...
    def put_nowait(self, obj: T) -> None: ...
    def get(self, block: bool = True, timeout: float | None = None) -> T: ...
    def close(self) -> None: ...
    def join_thread(self) -> None: ...


class _ImmediateQueue[T]:
    """A multiprocessing queue whose put has written the item out of the process when it returns.

    multiprocessing.Queue.put only hands the item to a feeder thread, which writes it out later
    and needs the GIL to do so. Native code that holds the GIL while it blocks (MLX's distributed
    init does, while it waits for the other ranks to connect) stops that thread, so what was put
    just before such a call stays in the process until the call returns. This queue writes from
    the thread that puts, so nothing can hold it back. In exchange put blocks while the pipe is
    full: the other end must keep reading.
    """

    def __init__(self) -> None:
        self._reader, self._writer = mp.Pipe(duplex=False)
        self._read_lock = mp.Lock()
        self._write_lock = mp.Lock()
        self._closed = False

    def put(self, obj: T, block: bool = True, timeout: float | None = None) -> None:
        if self._closed:
            raise ValueError("Queue is closed")
        payload = ForkingPickler.dumps(obj)
        if not self._write_lock.acquire(block, timeout):
            raise Full
        try:
            # Without blocking, write only while the pipe has room, which is all a small
            # item (such as the end-of-stream marker) needs. (poll, as select can't take
            # descriptors numbered above 1023.)
            if not block:
                writable = select.poll()
                writable.register(self._writer, select.POLLOUT)
                if not writable.poll(0):
                    raise Full
            self._writer.send_bytes(payload)
        finally:
            self._write_lock.release()

    def put_nowait(self, obj: T) -> None:
        self.put(obj, block=False)

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        if self._closed:
            raise ValueError("Queue is closed")
        if not self._read_lock.acquire(block, timeout):
            raise Empty
        try:
            if (not block or timeout is not None) and not self._reader.poll(
                timeout if block else 0
            ):
                raise Empty
            payload = self._reader.recv_bytes()
        finally:
            self._read_lock.release()
        return cast(T, pickle.loads(payload))

    def close(self) -> None:
        self._closed = True
        self._reader.close()
        self._writer.close()

    def join_thread(self) -> None:
        """Nothing to wait for: every item put has been written already"""


class MpState[T]:
    def __init__(self, max_buffer_size: float, send_immediately: bool = False):
        if max_buffer_size == inf:
            max_buffer_size = 0
        assert isinstance(max_buffer_size, int), (
            "State should only ever be constructed with an integer or math.inf size."
        )

        self.max_buffer_size: float = max_buffer_size
        self.buffer: _MpBuffer[T | _MpEndOfStream]
        if send_immediately:
            assert max_buffer_size == 0, "Channels that send immediately are unbounded"
            self.buffer = _ImmediateQueue()
        else:
            self.buffer = mp.Queue(max_buffer_size)
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
        await to_thread.run_sync(
            self.send, item, limiter=CapacityLimiter(1), abandon_on_cancel=True
        )

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        with contextlib.suppress(Exception):
            self._state.buffer.put_nowait(_MpEndOfStream())
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

    def __getstate__(self) -> dict[str, Any]:
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
            try:
                item = self._state.buffer.get()
            except (TypeError, OSError):
                # Queue pipe can get closed while we are blocked on get().
                # The underlying connection._handle becomes None, causing
                # TypeError in read(handle, remaining).
                raise ClosedResourceError from None
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream from None
            return item

    async def receive_async(self) -> T:
        return await to_thread.run_sync(
            self.receive, limiter=CapacityLimiter(1), abandon_on_cancel=True
        )

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        with contextlib.suppress(Exception):
            self._state.buffer.put_nowait(_MpEndOfStream())
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

    def __new__(
        cls,
        max_buffer_size: float = inf,
        error_override_config: ErrorOverride | None = None,
    ) -> tuple[Sender[T], Receiver[T]]:
        if max_buffer_size != inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        state = AnyioState[T](max_buffer_size)
        return Sender(state, error_override_config), Receiver(
            state, error_override_config
        )


class mp_channel[T]:  # noqa: N801
    """Create a pair of synchronous channels for interprocess communication"""

    # max buffer size uses math.inf to represent an unbounded queue, and 0 to represent a yet unimplemented "unbuffered" queue.
    #
    # With send_immediately, send writes the item out of the process before it returns, instead
    # of leaving that to a background thread: use it for a sender that may then block in native
    # code holding the GIL, which stops that thread. Such a send blocks while the pipe is full,
    # so the receiver must keep reading.
    def __new__(
        cls, max_buffer_size: float = inf, *, send_immediately: bool = False
    ) -> tuple[MpSender[T], MpReceiver[T]]:
        if (
            max_buffer_size == 0
            or max_buffer_size != inf
            and not isinstance(max_buffer_size, int)
        ):
            raise ValueError(
                "max_buffer_size must be either an integer or math.inf. 0-sized buffers are not supported by multiprocessing"
            )
        state = MpState[T](max_buffer_size, send_immediately)
        return MpSender(_state=state), MpReceiver(_state=state)
