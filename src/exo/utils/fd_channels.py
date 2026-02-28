import contextlib
import os
import struct
from dataclasses import dataclass
from typing import Any, Self

from anyio import (
    BrokenResourceError,
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
from pydantic import TypeAdapter


def _encode_message(data: bytes) -> bytes:
    """Encode a message with its length prefix."""
    return struct.pack("!I", len(data)) + data


def _decode_message(data: bytes) -> tuple[bytes, bytes]:
    """Decode a message from a buffer, returning (message, remaining_buffer)."""
    if len(data) < 4:
        return b"", data
    length = struct.unpack("!I", data[:4])[0]
    if len(data) < 4 + length:
        return b"", data
    return data[4 : 4 + length], data[4 + length :]


class FdSender[T]:
    """A sender that writes JSON-encoded Pydantic models to a file descriptor."""

    def __init__(
        self,
        fd: int,
        type_adapter: TypeAdapter[T],
        max_buffer_size: int = 1024 * 1024,
    ) -> None:
        self._fd = fd
        self._type_adapter = type_adapter
        self._max_buffer_size = max_buffer_size
        self._closed = False
        self._write_buffer = bytearray()

    def send_nowait(self, item: T) -> None:
        """Send an item without blocking."""
        if self._closed:
            raise ClosedResourceError

        data = self._type_adapter.dump_json(item)
        message = _encode_message(data)

        if len(self._write_buffer) + len(message) > self._max_buffer_size:
            raise WouldBlock

        self._write_buffer.extend(message)
        self._flush()

    def send(self, item: T) -> None:
        """Send an item, blocking if necessary."""
        if self._closed:
            raise ClosedResourceError

        data = self._type_adapter.dump_json(item)
        message = _encode_message(data)
        self._write_buffer.extend(message)
        self._flush_blocking()

    async def send_async(self, item: T) -> None:
        """Send an item asynchronously."""
        await to_thread.run_sync(self.send, item)

    def _flush(self) -> None:
        """Flush the write buffer non-blocking."""
        if not self._write_buffer:
            return

        try:
            written = os.write(self._fd, self._write_buffer)
            self._write_buffer = self._write_buffer[written:]
        except (OSError, BrokenPipeError) as e:
            raise BrokenResourceError from e

    def _flush_blocking(self) -> None:
        """Flush the write buffer, blocking until complete."""
        while self._write_buffer:
            self._flush()

    def close(self) -> None:
        """Close the sender."""
        if not self._closed:
            self._closed = True
            try:
                self._flush_blocking()
                os.close(self._fd)
            except (OSError, BrokenPipeError):
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class FdReceiver[T]:
    """A receiver that reads JSON-encoded Pydantic models from a file descriptor."""

    def __init__(
        self,
        fd: int,
        type_adapter: TypeAdapter[T],
        max_buffer_size: int = 1024 * 1024,
    ) -> None:
        self._fd = fd
        self._type_adapter = type_adapter
        self._max_buffer_size = max_buffer_size
        self._closed = False
        self._read_buffer = bytearray()

    def receive_nowait(self) -> T:
        """Receive an item without blocking."""
        if self._closed:
            raise ClosedResourceError

        # Try to read any available data
        self._read_available()

        # Try to decode a message
        message, remaining = _decode_message(bytes(self._read_buffer))
        if message:
            self._read_buffer = bytearray(remaining)
            return self._type_adapter.validate_json(message)

        raise WouldBlock

    def receive(self) -> T:
        """Receive an item, blocking if necessary."""
        if self._closed:
            raise ClosedResourceError

        while True:
            # Try to decode a message
            message, remaining = _decode_message(bytes(self._read_buffer))
            if message:
                self._read_buffer = bytearray(remaining)
                return self._type_adapter.validate_json(message)

            # Read more data
            try:
                chunk = os.read(self._fd, 8192)
                if not chunk:
                    # EOF
                    self.close()
                    raise EndOfStream
                self._read_buffer.extend(chunk)
            except (OSError, BrokenPipeError) as e:
                self.close()
                raise BrokenResourceError from e

    async def receive_async(self) -> T:
        """Receive an item asynchronously."""
        return await to_thread.run_sync(self.receive)

    def _read_available(self) -> None:
        """Read any available data without blocking."""
        try:
            import fcntl

            # Set non-blocking mode temporarily
            flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
            fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                chunk = os.read(self._fd, 8192)
                if chunk:
                    self._read_buffer.extend(chunk)
            except BlockingIOError:
                pass
            finally:
                fcntl.fcntl(self._fd, fcntl.F_SETFL, flags)
        except ImportError:
            # Windows doesn't have fcntl, just try reading
            pass

    def close(self) -> None:
        """Close the receiver."""
        if not self._closed:
            self._closed = True
            with contextlib.suppress(OSError, BrokenPipeError):
                os.close(self._fd)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            return self.receive()
        except EndOfStream:
            raise StopIteration from None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive_async()
        except EndOfStream:
            raise StopAsyncIteration from None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def collect(self) -> list[T]:
        """Collect all currently available items from this receiver."""
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out


@dataclass(eq=False)
class FdChannel:
    """A pair of file descriptors for bidirectional communication."""

    sender_fd: int
    receiver_fd: int

    def close(self) -> None:
        """Close both file descriptors."""
        for fd in (self.sender_fd, self.receiver_fd):
            with contextlib.suppress(OSError, BrokenPipeError):
                os.close(fd)


def fd_channel_pair() -> tuple[FdChannel, FdChannel]:
    """
    Create a pair of connected file descriptor channels.
    Returns (parent_channel, child_channel).
    """
    # Create two pipes: one for parent->child, one for child->parent
    parent_to_child_read, parent_to_child_write = os.pipe()
    child_to_parent_read, child_to_parent_write = os.pipe()

    # Parent writes to parent_to_child_write, reads from child_to_parent_read
    parent_channel = FdChannel(
        sender_fd=parent_to_child_write,
        receiver_fd=child_to_parent_read,
    )

    # Child reads from parent_to_child_read, writes to child_to_parent_write
    child_channel = FdChannel(
        sender_fd=child_to_parent_write,
        receiver_fd=parent_to_child_read,
    )

    return parent_channel, child_channel


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
        """Constructs a Sender using a Receivers shared state - similar to calling Sender.clone() without needing the receiver"""
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
