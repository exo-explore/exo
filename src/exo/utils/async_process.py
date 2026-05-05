from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Mapping
from multiprocessing.process import BaseProcess
from multiprocessing.resource_sharer import DupFd
from signal import Signals
from typing import Any, Iterable, final

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    create_memory_object_stream,
    to_thread,
    wait_readable,
)
from anyio.abc import (
    ByteReceiveStream,
    ByteSendStream,
    ObjectReceiveStream,
    ObjectSendStream,
)
from anyio.abc import (
    Process as AnyioProcess,
)

_STDOUT_FD = 1
_STDERR_FD = 2
_READ_CHUNK_SIZE = 64 * 1024


@final
class MemoryByteReceiveStream(ByteReceiveStream):
    def __init__(self, receive_stream: ObjectReceiveStream[bytes]) -> None:
        self._receive_stream = receive_stream
        self._buffer = bytearray()

    async def receive(self, max_bytes: int = _READ_CHUNK_SIZE) -> bytes:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        if self._buffer:
            chunk = bytes(self._buffer[:max_bytes])
            del self._buffer[:max_bytes]
            return chunk

        chunk = await self._receive_stream.receive()
        if len(chunk) <= max_bytes:
            return chunk

        self._buffer.extend(chunk[max_bytes:])
        return chunk[:max_bytes]

    async def aclose(self) -> None:
        self._buffer.clear()
        await self._receive_stream.aclose()


@final
class AsyncSpawnProcess(AnyioProcess):
    def __init__(
            self,
            target: Callable[..., object] | None = None,
            name: str | None = None,
            args: Iterable[Any] = (),
            kwargs: Mapping[str, Any] = {},
            *,
            daemon: bool | None = None,
            stream_buffer_size: int = 16,
    ) -> None:
        if stream_buffer_size <= 0:
            raise ValueError("stream_buffer_size must be positive")

        # setup state
        self._target = target
        self._name = name
        self._args = args
        self._kwargs = kwargs
        self._daemon = daemon
        self._stream_buffer_size = stream_buffer_size

        # lifecycle state
        self._process: BaseProcess | None = None
        self._pid: int | None = None
        self._stdout: MemoryByteReceiveStream | None = None
        self._stderr: MemoryByteReceiveStream | None = None
        self._drain_tasks: tuple[asyncio.Task[None], ...] = ()
        self._closed = False
        self._returncode: int | None = None

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("process has been closed")
        if self._process is not None:
            raise RuntimeError("process has already been started")

        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        stdout_send, stdout_receive = create_memory_object_stream[bytes](
            self._stream_buffer_size
        )
        stderr_send, stderr_receive = create_memory_object_stream[bytes](
            self._stream_buffer_size
        )
        drain_tasks: tuple[asyncio.Task[None], ...] = ()

        try:
            process = mp.get_context("spawn").Process(
                target=_run_with_captured_stdio,
                name=self._name,
                args=(
                    DupFd(stdout_write_fd),
                    DupFd(stderr_write_fd),
                    self._target,
                    *self._args,
                ),
                kwargs=self._kwargs,
                daemon=self._daemon,
            )
            process.start()
            pid = process.pid
            if pid is None:
                raise RuntimeError("started process has no pid")

            # important to close parent write-side FD to prevent hangs
            _close_fd(stdout_write_fd)
            _close_fd(stderr_write_fd)

            # drain the read-end FD into async memory-buffer stream
            drain_tasks = (
                asyncio.create_task(_drain_fd(stdout_read_fd, stdout_send)),
                asyncio.create_task(_drain_fd(stderr_read_fd, stderr_send)),
            )

            self._process = process
            self._pid = pid
            self._stdout = MemoryByteReceiveStream(stdout_receive)
            self._stderr = MemoryByteReceiveStream(stderr_receive)
            self._drain_tasks = drain_tasks
        except BaseException:
            await _cancel_drain_tasks(drain_tasks)
            for stream in (stdout_send, stderr_send, stdout_receive, stderr_receive):
                with contextlib.suppress(Exception):
                    await stream.aclose()
            for fd in (
                    stdout_read_fd,
                    stdout_write_fd,
                    stderr_read_fd,
                    stderr_write_fd,
            ):
                _close_fd(fd)
            raise

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._process is None:
            return

        with CancelScope(shield=True) as scope:
            await self.stdout.aclose()
            await self.stderr.aclose()
            await _cancel_drain_tasks(self._drain_tasks)

            scope.shield = False
            try:
                await self.wait()
            except BaseException:
                scope.shield = True
                self.kill()
                await self.wait()
                raise
            finally:
                with contextlib.suppress(ValueError):
                    self._process.close()

    async def wait(self) -> int:
        if self._returncode is not None:
            return self._returncode

        process = self.process
        await to_thread.run_sync(process.join)
        exitcode = process.exitcode
        if exitcode is None:
            raise RuntimeError("process exited without an exit code")

        self._returncode = exitcode
        return exitcode

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def send_signal(self, signal: Signals) -> None:
        os.kill(self.pid, signal)

    @property
    def pid(self) -> int:
        if self._pid is None:
            raise RuntimeError("process has not been started")
        return self._pid

    @property
    def returncode(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        if self._process is None:
            return None

        with contextlib.suppress(ValueError):
            return self._process.exitcode
        return None

    @property
    def stdin(self) -> ByteSendStream:
        # todo: implement this if ever needed
        raise NotImplementedError("custom stdin streams have not yet been implemented")

    @property
    def stdout(self) -> ByteReceiveStream:
        if self._stdout is None:
            raise RuntimeError("process has not been started")
        return self._stdout

    @property
    def stderr(self) -> ByteReceiveStream:
        if self._stderr is None:
            raise RuntimeError("process has not been started")
        return self._stderr

    @property
    def process(self) -> BaseProcess:
        if self._process is None:
            raise RuntimeError("process has not been started")
        return self._process


# Spawn-mode multiprocessing requires a module-level target that can be pickled.
def _run_with_captured_stdio(
        stdout: DupFd,
        stderr: DupFd,
        target: Callable[..., object],
        *target_args: object,
        **target_kwargs: object,
) -> None:
    stdout_fd = stdout.detach()
    stderr_fd = stderr.detach()

    try:
        os.dup2(stdout_fd, _STDOUT_FD)
        os.dup2(stderr_fd, _STDERR_FD)
    finally:
        for fd in (stdout_fd, stderr_fd):
            if fd not in (_STDOUT_FD, _STDERR_FD):
                _close_fd(fd)

    faulthandler.enable(file=sys.stderr, all_threads=True)
    target(*target_args, **target_kwargs)


async def _drain_fd(fd: int, send_stream: ObjectSendStream[bytes]) -> None:
    try:
        while True:
            await wait_readable(fd)
            chunk = os.read(fd, _READ_CHUNK_SIZE)
            if not chunk:
                return
            await send_stream.send(chunk)
    except (BrokenPipeError, BrokenResourceError, ClosedResourceError):
        pass
    finally:
        _close_fd(fd)
        await send_stream.aclose()


async def _cancel_drain_tasks(tasks: tuple[asyncio.Task[None], ...]) -> None:
    if not tasks:
        return

    for task in tasks:
        task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if result is None or isinstance(result, asyncio.CancelledError):
            continue
        raise result


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)
