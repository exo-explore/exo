from __future__ import annotations

import contextlib
import faulthandler
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from multiprocessing.process import BaseProcess
from multiprocessing.resource_sharer import DupFd
from typing import final

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    Event,
    Lock,
    create_task_group,
    move_on_after,
    sleep,
    wait_readable,
)

from exo.utils.channels import Receiver, Sender, channel

_STDOUT_FD = 1
_STDERR_FD = 2
_READ_CHUNK_SIZE = 64 * 1024
_TERMINATE_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 5.0


@final
class AsyncProcess:
    def __init__(
            self,
            target: Callable[..., object] | None = None,
            name: str | None = None,
            args: Iterable[object] = (),
            kwargs: Mapping[str, object] | None = None,
            *,
            daemon: bool | None = None,
    ) -> None:
        # setup state
        self._target = target
        self._name = name
        self._args = args
        self._kwargs = kwargs
        self._daemon = daemon

        # lifecycle state
        self._process: BaseProcess | None = None
        self._pid: int | None = None
        self._stdout_tx, self._stdout_rx = channel[bytes]()
        self._stderr_tx, self._stderr_rx = channel[bytes]()
        self._started = Event()
        self._stopped = Event()
        self._run_cancel_scope: CancelScope | None = None
        self._run_called = False
        self._wait_lock = Lock()
        self._start_error: BaseException | None = None
        self._closed = False
        self._exitcode: int | None = None

    async def run(self) -> None:
        if self._run_called:
            raise RuntimeError("process has already been started")
        self._run_called = True

        stdout_read_fd: int | None = None
        stdout_write_fd: int | None = None
        stderr_read_fd: int | None = None
        stderr_write_fd: int | None = None

        try:
            if self._closed:
                raise RuntimeError("process has been closed")

            with CancelScope() as run_cancel_scope:
                self._run_cancel_scope = run_cancel_scope
                stdout_read_fd, stdout_write_fd = os.pipe()
                stderr_read_fd, stderr_write_fd = os.pipe()

                process = mp.Process(
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
                self._started.set()

                async with create_task_group() as tg:
                    tg.start_soon(_drain_fd, stdout_read_fd, self._stdout_tx)
                    tg.start_soon(_drain_fd, stderr_read_fd, self._stderr_tx)
                    await self.wait()
        except BaseException as exc:
            self._start_error = exc
            self._started.set()
            self._stopped.set()
            for ch in (self._stdout_tx, self._stdout_rx, self._stderr_tx, self._stderr_rx):
                with contextlib.suppress(Exception):
                    await ch.aclose()
            for fd in (stdout_read_fd, stdout_write_fd, stderr_read_fd, stderr_write_fd):
                if fd is not None:
                    _close_fd(fd)
            raise
        finally:
            try:
                with CancelScope(shield=True):
                    await self._terminate_if_still_alive()
            finally:
                for fd in (stdout_read_fd, stderr_read_fd):
                    if fd is not None:
                        _close_fd(fd)
                for tx in (self._stdout_tx, self._stderr_tx):
                    with contextlib.suppress(Exception):
                        await tx.aclose()
                self._run_cancel_scope = None
                self._stopped.set()

    async def stop(self) -> None:
        if self._closed:
            return

        if not self._run_called:
            raise RuntimeError("process has not been started")
        if self._run_cancel_scope is not None:
            self._run_cancel_scope.cancel()
        await self._stopped.wait()
        self._close()

    async def aclose(self) -> None:
        await self.stop()

    async def wait(self) -> int:
        if self._exitcode is not None:
            return self._exitcode

        async with self._wait_lock:
            if self._exitcode is not None:
                return self._exitcode

            await self._started.wait()
            if self._start_error is not None:
                raise self._start_error
            if self._process is None:
                raise RuntimeError("process has not been started")
            process = self._process
            while True:
                exitcode = process.exitcode
                if exitcode is not None:
                    process.join(0)
                    self._exitcode = exitcode
                    return exitcode

                await sleep(0.01)

    def _close(self) -> None:
        self._closed = True
        for ch in (self._stdout_tx, self._stdout_rx, self._stderr_tx, self._stderr_rx):
            with contextlib.suppress(Exception):
                ch.close()
        if self._process is None:
            return
        with contextlib.suppress(ValueError):
            self._process.close()

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
        return self._stdout_rx

    @property
    def stderr(self) -> Receiver[bytes]:
        return self._stderr_rx

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


async def _drain_fd(fd: int, tx: Sender[bytes]) -> None:
    try:
        while True:
            await wait_readable(fd)
            chunk = os.read(fd, _READ_CHUNK_SIZE)
            if not chunk:
                return
            await tx.send(chunk)
    except (BrokenPipeError, BrokenResourceError, ClosedResourceError):
        pass
    finally:
        _close_fd(fd)
        await tx.aclose()


def _close_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)
