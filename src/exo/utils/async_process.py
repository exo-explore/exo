from __future__ import annotations

import contextlib
import faulthandler
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.resource_sharer import DupFd
from signal import Signals
from typing import final

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    Event,
    Lock,
    move_on_after,
    sleep,
    wait_readable,
)

from exo.utils.channels import Sender, channel, Receiver
from exo.utils.task_group import TaskGroup

_STDOUT_FD = 1
_STDERR_FD = 2
_READ_CHUNK_SIZE = 64 * 1024
_TERMINATE_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 5.0


@final
class AsyncSpawnProcess:
    @staticmethod
    def context() -> SpawnContext:
        return mp.get_context("spawn")

    def __init__(
            self,
            target: Callable[..., object] | None = None,
            name: str | None = None,
            args: Iterable[object] = (),
            kwargs: Mapping[str, object] | None = None,
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
        self._stdout: Receiver[bytes] | None = None
        self._stderr: Receiver[bytes] | None = None
        self._tg = TaskGroup()
        self._started = Event()
        self._stopped = Event()
        self._wait_lock = Lock()
        self._start_error: BaseException | None = None
        self._has_stopped = False
        self._closed = False
        self._exitcode: int | None = None

    async def run(self) -> None:
        if self._closed:
            raise RuntimeError("process has been closed")
        if self._process is not None:
            raise RuntimeError("process has already been started")

        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        stdout_tx, stdout_rx = channel[bytes]()
        stderr_tx, stderr_rx = channel[bytes]()

        try:
            process = self.context().Process(
                target=_run_with_captured_stdio,
                name=self._name,
                args=(
                    DupFd(stdout_write_fd),
                    DupFd(stderr_write_fd),
                    self._target,
                    *self._args,
                ),
                kwargs={} if self._kwargs is None else self._kwargs,
                daemon=self._daemon,
            )
            process.start()
            pid = process.pid
            if pid is None:
                raise RuntimeError("started process has no pid")

            # important to close parent write-side FD to prevent hangs
            _close_fd(stdout_write_fd)
            _close_fd(stderr_write_fd)

            self._process = process
            self._pid = pid
            self._stdout = stdout_rx
            self._stderr = stderr_rx
            self._started.set()
        except BaseException as exc:
            self._start_error = exc
            self._started.set()
            self._has_stopped = True
            self._stopped.set()
            for ch in (stdout_tx, stdout_rx, stderr_tx, stderr_rx):
                with contextlib.suppress(Exception):
                    await ch.aclose()
            for fd in (
                    stdout_read_fd,
                    stdout_write_fd,
                    stderr_read_fd,
                    stderr_write_fd,
            ):
                _close_fd(fd)
            raise

        try:
            async with self._tg as tg:
                tg.start_soon(_drain_fd, stdout_read_fd, stdout_tx)
                tg.start_soon(_drain_fd, stderr_read_fd, stderr_tx)
                await self.wait()
        finally:
            try:
                with CancelScope(shield=True):
                    await self._terminate_if_still_alive()
            finally:
                self._has_stopped = True
                self._stopped.set()

    async def wait_started(self) -> None:
        await self._started.wait()
        if self._start_error is not None:
            raise self._start_error

    async def wait_stopped(self) -> None:
        await self._stopped.wait()

    def shutdown(self) -> None:
        if not self._has_stopped and self._tg.is_running():
            self._tg.cancel_tasks()

    async def aclose(self) -> None:
        if self._closed:
            return

        self.shutdown()
        if self._process is not None and not self._has_stopped:
            await self.wait_stopped()
        self.close()

    async def wait(self) -> int:
        if self._exitcode is not None:
            return self._exitcode

        async with self._wait_lock:
            if self._exitcode is not None:
                return self._exitcode

            process = self.process
            while True:
                exitcode = process.exitcode
                if exitcode is not None:
                    process.join(0)
                    self._exitcode = exitcode
                    return exitcode

                await sleep(0.01)

    def terminate(self) -> None:
        self.process.terminate()

    def is_alive(self) -> bool:
        if self._process is None:
            return False
        with contextlib.suppress(ValueError):
            return self._process.is_alive()
        return False

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)

    def close(self) -> None:
        self._closed = True
        if self._process is None:
            return
        with contextlib.suppress(ValueError):
            self._process.close()

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
    def exitcode(self) -> int | None:
        if self._exitcode is not None:
            return self._exitcode
        if self._process is None:
            return None

        with contextlib.suppress(ValueError):
            exitcode = self._process.exitcode
            if exitcode is not None:
                self._exitcode = exitcode
            return exitcode
        return None

    # TODO: maybe in the future if needed, create stdin that is also installed,
    #       and a ByteSendStream handle is provided for it :)

    @property
    def stdout(self) -> Receiver[bytes]:
        if self._stdout is None:
            raise RuntimeError("process has not been started")
        return self._stdout

    @property
    def stderr(self) -> Receiver[bytes]:
        if self._stderr is None:
            raise RuntimeError("process has not been started")
        return self._stderr

    @property
    def process(self) -> BaseProcess:
        if self._process is None:
            raise RuntimeError("process has not been started")
        return self._process

    async def _terminate_if_still_alive(self) -> None:
        process = self._process
        if process is None:
            return

        if self.exitcode is not None:
            return

        with contextlib.suppress(ValueError):
            if process.is_alive():
                process.terminate()
                with move_on_after(_TERMINATE_GRACE_SECONDS):
                    await self.wait()

            if self.exitcode is not None or not process.is_alive():
                return

            process.kill()
            with move_on_after(_KILL_GRACE_SECONDS):
                await self.wait()

            if self.exitcode is not None or not process.is_alive():
                return

            raise RuntimeError(f"process {self.pid} is still alive after SIGKILL")


# Spawn-mode multiprocessing requires a module-level target that can be pickled.
def _run_with_captured_stdio(
        stdout: DupFd,
        stderr: DupFd,
        target: Callable[..., object] | None,
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
    if target is not None:
        target(*target_args, **target_kwargs)


async def _drain_fd(fd: int, send_stream: Sender[bytes]) -> None:
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


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)
