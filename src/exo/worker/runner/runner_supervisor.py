import contextlib
import json
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
)
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream
from loguru import logger
from pydantic import TypeAdapter

from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_args import RunnerCliArgs
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Sender
from exo.utils.fd_channels import FdReceiver, FdSender
from exo.utils.task_group import TaskGroup

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5

# Type adapters for union types
event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
task_adapter: TypeAdapter[Task] = TypeAdapter(Task)
task_id_adapter: TypeAdapter[TaskId] = TypeAdapter(TaskId)


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    process: Process
    initialize_timeout: float
    _ev_recv: FdReceiver[Event]
    _task_sender: FdSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: FdSender[TaskId]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        # Create pipes for each communication direction:
        # - Events: child -> parent (child writes, parent reads)
        # - Tasks: parent -> child (parent writes, child reads)
        # - Cancels: parent -> child (parent writes, child reads)
        event_read_fd, event_write_fd = os.pipe()  # Child writes, parent reads
        task_read_fd, task_write_fd = os.pipe()  # Parent writes, child reads
        cancel_read_fd, cancel_write_fd = os.pipe()  # Parent writes, child reads

        # Build CLI args with only the FDs the child needs
        args = RunnerCliArgs(
            bound_instance=bound_instance,
            event_fd=event_write_fd,  # Child writes events here
            task_fd=task_read_fd,  # Child reads tasks from here
            cancel_fd=cancel_read_fd,  # Child reads cancels from here
            log_level="INFO",
        )

        # File descriptors to pass to the child process
        pass_fds = [event_write_fd, task_read_fd, cancel_read_fd]

        shard_metadata = bound_instance.bound_shard

        # Create FD-based channels for the parent side using TypeAdapters
        ev_recv = FdReceiver[Event](event_read_fd, event_adapter)
        task_sender = FdSender[Task](task_write_fd, task_adapter)
        cancel_sender = FdSender[TaskId](cancel_write_fd, task_id_adapter)

        # Close the child-side FDs in the parent process
        with contextlib.suppress(OSError):
            os.close(event_write_fd)
        with contextlib.suppress(OSError):
            os.close(task_read_fd)
        with contextlib.suppress(OSError):
            os.close(cancel_read_fd)

        # Start the process using anyio.open_process
        # Check if running in a frozen environment (e.g., PyInstaller)
        is_frozen = os.environ.get("EXO_FROZEN") == "1" or getattr(sys, "frozen", False)

        if is_frozen:
            # In frozen builds, use the same executable with a special flag
            # The __main__.py handles this by checking sys.argv
            cmd = [
                sys.executable,
                "--exo-runner",
                json.dumps(args.model_dump(mode="json")),
            ]
        else:
            # In normal Python, use -m to run the module
            cmd = [
                sys.executable,
                "-m",
                "exo.worker.runner.bootstrap",
                json.dumps(args.model_dump(mode="json")),
            ]

        process = await anyio.open_process(cmd, pass_fds=pass_fds)

        return cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            process=process,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )

    async def run(self):
        async with self._tg as tg:
            tg.start_soon(self._watch_runner)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._read_stderr)

    async def _read_stderr(self):
        """Read and log stderr from the runner process."""
        if self.process.stderr:
            async with TextReceiveStream(self.process.stderr) as stderr_stream:
                async for line in stderr_stream:
                    logger.debug(f"Runner stderr: {line.rstrip()}")

    async def shutdown(self):
        logger.info("Runner supervisor shutting down")
        self._tg.cancel_tasks()
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        with contextlib.suppress(ClosedResourceError):
            self._ev_recv.close()
        with contextlib.suppress(ClosedResourceError):
            self._task_sender.close()
        with contextlib.suppress(ClosedResourceError):
            self._event_sender.close()
        with contextlib.suppress(ClosedResourceError):
            self._cancel_sender.send(TaskId("CANCEL_CURRENT_TASK"))
        with contextlib.suppress(ClosedResourceError):
            self._cancel_sender.close()

        # Try graceful termination
        if self.process.returncode is None:
            self.process.terminate()
            try:
                with anyio.fail_after(5):
                    await self.process.wait()
            except TimeoutError:
                logger.warning("Runner process didn't shutdown gracefully, killing")
                self.process.kill()
                try:
                    with anyio.fail_after(1):
                        await self.process.wait()
                except TimeoutError:
                    logger.critical("Runner process didn't respond to SIGKILL")

    async def start_task(self, task: Task):
        if task.task_id in self.pending:
            logger.warning(
                f"Skipping invalid task {task} as it has already been submitted"
            )
            return
        if task.task_id in self.completed:
            logger.warning(
                f"Skipping invalid task {task} as it has already been completed"
            )
            return
        logger.info(f"Starting task {task}")
        event = anyio.Event()
        self.pending[task.task_id] = event
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        await event.wait()

    async def cancel_task(self, task_id: TaskId):
        if task_id in self.completed:
            logger.info(f"Unable to cancel {task_id} as it has been completed")
            return
        self.cancelled.add(task_id)
        with anyio.move_on_after(0.5) as scope:
            await self._cancel_sender.send_async(task_id)
        if scope.cancel_called:
            logger.error("RunnerSupervisor cancel pipe blocked")
            await self._check_runner(TimeoutError("cancel pipe blocked"))

    async def _forward_events(self):
        try:
            with self._ev_recv as events:
                async for event in events:
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                    if isinstance(event, TaskAcknowledged):
                        self.pending.pop(event.task_id).set()
                        continue
                    if (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status == TaskStatus.Complete
                    ):
                        # If a task has just been completed, we should be working on it.
                        assert isinstance(
                            self.status,
                            (
                                RunnerRunning,
                                RunnerWarmingUp,
                                RunnerLoading,
                                RunnerConnecting,
                                RunnerShuttingDown,
                            ),
                        )
                        self.completed.add(event.task_id)
                    await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError) as e:
            await self._check_runner(e)
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    def __del__(self) -> None:
        if self.process.returncode is None:
            logger.critical("RunnerSupervisor was not stopped cleanly.")
            with contextlib.suppress(Exception):
                self.process.kill()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if self.process.returncode is not None:
                    await self._check_runner(RuntimeError("Runner found to be dead"))

    async def _check_runner(self, e: Exception) -> None:
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info("Checking runner's status")

        returncode = self.process.returncode

        if returncode is None:
            # Process is still running, try to wait for it
            self.process.terminate()
            try:
                with anyio.fail_after(5):
                    await self.process.wait()
                    returncode = self.process.returncode
            except TimeoutError:
                pass

        logger.info(f"Runner exited with return code {returncode}")

        if returncode == 0:
            return

        if isinstance(returncode, int) and returncode < 0:
            sig = -returncode
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"returncode={returncode}"

        logger.opt(exception=e).error(f"Runner terminated with {cause}")

        try:
            self.status = RunnerFailed(error_message=f"Terminated ({cause})")
            with anyio.CancelScope(shield=True):
                await self._event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=self.bound_instance.bound_runner_id,
                        runner_status=RunnerFailed(
                            error_message=f"Terminated ({cause})"
                        ),
                    )
                )
        except (ClosedResourceError, BrokenResourceError):
            logger.warning(
                "Event sender already closed, unable to report runner failure"
            )
        await self.shutdown()
