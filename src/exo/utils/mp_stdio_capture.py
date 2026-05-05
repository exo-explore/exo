from __future__ import annotations

import contextlib
import faulthandler
import io
import multiprocessing as mp
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.context import (
    DefaultContext,
    ForkContext,
    ForkServerContext,
    SpawnContext,
)
from multiprocessing.process import BaseProcess
from multiprocessing.resource_sharer import DupFd
from types import TracebackType
from typing import Literal, Self, TextIO, final

from anyio import create_task_group, to_thread

_STDOUT_FD = 1
_STDERR_FD = 2
_READ_CHUNK_SIZE = 64 * 1024

MultiprocessingStartMethod = Literal["fork", "forkserver", "spawn"]
ProcessContext = DefaultContext | ForkContext | ForkServerContext | SpawnContext


@final
@dataclass(frozen=True)
class ChildFileDescriptor:
    """A parent fd wrapper that can be detached as a valid fd in a spawned child."""

    _wrapped: DupFd

    @classmethod
    def from_parent_fd(cls, fd: int) -> Self:
        if fd < 0:
            raise ValueError("file descriptor must be non-negative")
        return cls(DupFd(fd))

    def detach(self) -> int:
        return self._wrapped.detach()


@final
@dataclass(frozen=True)
class ChildStdio:
    """Child-side stdio redirection installed at the top of the process target."""

    stdout: ChildFileDescriptor
    stderr: ChildFileDescriptor

    def install(self) -> None:
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            if stream is not None:
                with contextlib.suppress(ValueError):
                    stream.flush()

        stdout_fd = self.stdout.detach()
        stderr_fd = self.stderr.detach()

        try:
            os.dup2(stdout_fd, _STDOUT_FD)
            os.dup2(stderr_fd, _STDERR_FD)
        finally:
            _close_unless_standard_fd(stdout_fd)
            _close_unless_standard_fd(stderr_fd)

        stdout = _open_text_stream_for_fd(_STDOUT_FD, sys.stdout)
        stderr = _open_text_stream_for_fd(_STDERR_FD, sys.stderr)
        sys.stdout = stdout
        sys.stderr = stderr
        sys.__stdout__ = stdout
        sys.__stderr__ = stderr

        for stream in (stdout, stderr):
            with contextlib.suppress(ValueError):
                stream.reconfigure(line_buffering=True, write_through=True)

        faulthandler.enable(file=stderr, all_threads=True)


@final
@dataclass(frozen=True)
class CapturedOutput:
    stdout: bytes
    stderr: bytes

    def stdout_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.stdout.decode(encoding, errors=errors)

    def stderr_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.stderr.decode(encoding, errors=errors)


@final
@dataclass(frozen=True)
class CapturedProcessResult:
    exitcode: int | None
    output: CapturedOutput


@final
@dataclass(frozen=True)
class CapturedProcessOptions:
    process_name: str | None = None
    daemon: bool | None = None
    start_method: MultiprocessingStartMethod | None = None
    max_capture_bytes: int | None = None

    def create_process[**P](
        self,
        target: Callable[P, object],
        *target_args: P.args,
        **target_kwargs: P.kwargs,
    ) -> CapturedMpProcess:
        return _create_captured_process(
            self,
            target,
            *target_args,
            **target_kwargs,
        )


@final
@dataclass(eq=False)
class CaptureBuffer:
    max_bytes: int | None = None
    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive when provided")

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return

        with self._lock:
            self._buffer.extend(chunk)
            if self.max_bytes is not None and len(self._buffer) > self.max_bytes:
                del self._buffer[: len(self._buffer) - self.max_bytes]

    def snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._buffer)

    def snapshot_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.snapshot().decode(encoding, errors=errors)


@final
@dataclass(eq=False)
class PipeStdioCapture:
    _stdout_read_fd: int
    _stderr_read_fd: int
    stdout: CaptureBuffer
    stderr: CaptureBuffer
    _closed: bool = field(default=False, init=False)

    async def drain(self) -> None:
        async with create_task_group() as task_group:
            task_group.start_soon(_drain_fd, self._stdout_read_fd, self.stdout)
            task_group.start_soon(_drain_fd, self._stderr_read_fd, self.stderr)
        self._closed = True

    def snapshot(self) -> CapturedOutput:
        return CapturedOutput(
            stdout=self.stdout.snapshot(),
            stderr=self.stderr.snapshot(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_fd(self._stdout_read_fd)
        _close_fd(self._stderr_read_fd)


@final
@dataclass(eq=False)
class CapturedMpProcess:
    process: BaseProcess
    capture: PipeStdioCapture
    _stdout_write_fd: int
    _stderr_write_fd: int
    _started: bool = field(default=False, init=False)
    _parent_write_fds_closed: bool = field(default=False, init=False)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("process has already been started")

        try:
            self.process.start()
        except BaseException:
            self.close_parent_fds()
            raise

        self._started = True
        self.close_parent_write_fds()

    async def wait(self, timeout: float | None = None) -> int | None:
        await to_thread.run_sync(self.process.join, timeout)
        return self.process.exitcode

    async def run(self) -> CapturedProcessResult:
        self.start()
        async with create_task_group() as task_group:
            task_group.start_soon(self.capture.drain)
            await self.wait()

        return CapturedProcessResult(
            exitcode=self.process.exitcode,
            output=self.capture.snapshot(),
        )

    def close_parent_write_fds(self) -> None:
        if self._parent_write_fds_closed:
            return
        self._parent_write_fds_closed = True
        _close_fd(self._stdout_write_fd)
        _close_fd(self._stderr_write_fd)

    def close_parent_fds(self) -> None:
        self.close_parent_write_fds()
        self.capture.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close_parent_fds()


def create_captured_process[**P](
    target: Callable[P, object],
    *target_args: P.args,
    **target_kwargs: P.kwargs,
) -> CapturedMpProcess:
    return _create_captured_process(
        CapturedProcessOptions(),
        target,
        *target_args,
        **target_kwargs,
    )


def _create_captured_process[**P](
    options: CapturedProcessOptions,
    target: Callable[P, object],
    *target_args: P.args,
    **target_kwargs: P.kwargs,
) -> CapturedMpProcess:
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()

    try:
        child_stdio = ChildStdio(
            stdout=ChildFileDescriptor.from_parent_fd(stdout_write_fd),
            stderr=ChildFileDescriptor.from_parent_fd(stderr_write_fd),
        )

        process_context: ProcessContext
        # Current multiprocessing stubs return BaseContext for union input, and
        # BaseContext does not expose Process. Keep the literal narrowing here.
        if options.start_method is None:
            process_context = mp.get_context()
        elif options.start_method == "fork":
            process_context = mp.get_context("fork")
        elif options.start_method == "forkserver":
            process_context = mp.get_context("forkserver")
        else:
            process_context = mp.get_context("spawn")

        process = process_context.Process(
            name=options.process_name,
            target=_run_with_captured_stdio,
            args=(child_stdio, target, *target_args),
            kwargs=target_kwargs,
            daemon=options.daemon,
        )

        return CapturedMpProcess(
            process=process,
            capture=PipeStdioCapture(
                _stdout_read_fd=stdout_read_fd,
                _stderr_read_fd=stderr_read_fd,
                stdout=CaptureBuffer(max_bytes=options.max_capture_bytes),
                stderr=CaptureBuffer(max_bytes=options.max_capture_bytes),
            ),
            _stdout_write_fd=stdout_write_fd,
            _stderr_write_fd=stderr_write_fd,
        )
    except BaseException:
        for fd in (stdout_read_fd, stdout_write_fd, stderr_read_fd, stderr_write_fd):
            _close_fd(fd)
        raise


# Spawn-mode multiprocessing requires a module-level target that can be pickled.
def _run_with_captured_stdio[**P](
    stdio: ChildStdio,
    target: Callable[P, object],
    *target_args: P.args,
    **target_kwargs: P.kwargs,
) -> None:
    stdio.install()
    target(*target_args, **target_kwargs)


async def _drain_fd(fd: int, buffer: CaptureBuffer) -> None:
    try:
        while True:
            chunk = await to_thread.run_sync(os.read, fd, _READ_CHUNK_SIZE)
            if not chunk:
                return
            buffer.append(chunk)
    finally:
        _close_fd(fd)


def _open_text_stream_for_fd(fd: int, fallback: TextIO) -> io.TextIOWrapper:
    encoding = fallback.encoding or "utf-8"
    errors = fallback.errors or "replace"
    return os.fdopen(
        fd,
        mode="w",
        buffering=1,
        encoding=encoding,
        errors=errors,
        closefd=False,
    )


def _close_unless_standard_fd(fd: int) -> None:
    if fd not in (_STDOUT_FD, _STDERR_FD):
        _close_fd(fd)


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)
