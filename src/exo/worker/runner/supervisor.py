import codecs
import contextlib
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import anyio
from anyio import (
    AsyncFile,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
)
from anyio.streams.text import TextReceiveStream
from loguru import logger

from exo.shared.constants import (
    EXO_RUNNER_LOG_DIR,
)
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ImageEdits,
    ImageGeneration,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
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
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Receiver, Sender, mp_channel
from exo.utils.fs import ensure_parent_directory_exists
from exo.utils.task_group import TaskGroup
from exo.worker.runner.bootstrap import RunnerTerminationError, entrypoint
from exo.worker.runner.diagnostics import (
    RunnerDiagnosticCollector,
    RunnerUnknown,
)

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


@dataclass(eq=False)
class RunnerStdioHandler:
    _bound_instance: BoundInstance
    _stdout_rx: Receiver[bytes]
    _stderr_rx: Receiver[bytes]
    _stderr_log: AsyncFile[str]
    diagnostics: RunnerDiagnosticCollector = field(
        default_factory=RunnerDiagnosticCollector
    )

    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        stdout_rx: Receiver[bytes],
        stderr_rx: Receiver[bytes],
        runner_log_dir: Path = EXO_RUNNER_LOG_DIR,
    ) -> Self:
        # create file in log_dir/<instanceID>/<runnerID>.stderr.log
        stderr_log_path = (
            runner_log_dir
            / bound_instance.instance.instance_id
            / f"{bound_instance.bound_runner_id}.stderr.log"
        )
        ensure_parent_directory_exists(stderr_log_path)
        stderr_log = await anyio.open_file(stderr_log_path, "w+")

        self = cls(
            _bound_instance=bound_instance,
            _stdout_rx=stdout_rx,
            _stderr_rx=stderr_rx,
            _stderr_log=stderr_log,
        )
        return self

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._handle_stdout)
                tg.start_soon(self._handle_stderr)
        finally:
            with CancelScope(shield=True):
                await self._stderr_log.aclose()

    async def _handle_stdout(self):
        # We don't expect anything in stdout so even reading this at all is going
        # to be quite weird; hence handle it by logging error and the received chunk

        rx = TextReceiveStream(self._stdout_rx, encoding="utf-8", errors="replace")
        try:
            async with rx:
                async for chunk in rx:
                    logger.error(f"Unexpected runner stdout chunk: {chunk}")
        except (ClosedResourceError, BrokenResourceError):
            logger.warning("Runner stdio stream closed before clean EOF")

    async def _handle_stderr(self):
        # The diagnostic collector is deliberately line-level for now. It records
        # bounded stderr context and known failure anchors; the supervisor
        # correlates those hints with the runner exit status before surfacing an
        # error.

        # not using TextReceiveStream because it doesn't do final=True handling on errors
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending_line = ""

        async def handle_line(line: str):
            # preserve whitespace for later log-mining
            line = line.removesuffix("\r")
            if not line:
                return

            # Send to logger & error recovery task
            logger.warning(f"Runner stderr: {line}")
            self.diagnostics.record_line(line)

        async def handle_text(text: str):
            nonlocal pending_line

            if not text:
                return

            await self._stderr_log.write(text)
            await self._stderr_log.flush()

            # newline buffering
            pending_line += text
            lines = pending_line.split("\n")
            pending_line = lines.pop()

            for line in lines:
                await handle_line(line)

        try:
            with self._stderr_rx:
                async for chunk in self._stderr_rx:
                    await handle_text(decoder.decode(chunk, final=False))
        except (ClosedResourceError, BrokenResourceError):
            logger.warning("Runner stdio stream closed before clean EOF")
        finally:
            with CancelScope(shield=True):
                await handle_text(decoder.decode(b"", final=True))
                await self._stderr_log.flush()

                if pending_line:
                    await handle_line(pending_line)
                    pending_line = ""


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: AsyncProcess
    _runner_stdio_handler: RunnerStdioHandler
    initialize_timeout: float
    _ev_recv: MpReceiver[Event | RunnerTerminationError]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
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
        ev_send, ev_recv = mp_channel[Event | RunnerTerminationError]()
        task_sender, task_recv = mp_channel[Task]()
        cancel_sender, cancel_recv = mp_channel[TaskId]()

        runner_process = AsyncProcess(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                cancel_recv,
                logger,
            ),
            daemon=True,
        )
        runner_stdio_handler = await RunnerStdioHandler.create(
            bound_instance=bound_instance,
            stdout_rx=runner_process.stdout,
            stderr_rx=runner_process.stderr,
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            _runner_stdio_handler=runner_stdio_handler,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )

        return self

    async def run(self):
        try:
            async with self._tg as tg:
                # start the process itself & handle its stdout/stderr
                await tg.start(self.runner_process.run)
                tg.start_soon(self._runner_stdio_handler.run)

                tg.start_soon(self._watch_runner)
                tg.start_soon(self._forward_events)
        finally:
            logger.info("Runner supervisor shutting down")
            if not self._cancel_watch_runner.cancel_called:
                self._cancel_watch_runner.cancel()
            with contextlib.suppress(ClosedResourceError):
                self._ev_recv.close()
            with contextlib.suppress(ClosedResourceError):
                self._task_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._event_sender.close()
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.send(CANCEL_ALL_TASKS)
            with contextlib.suppress(ClosedResourceError):
                self._cancel_sender.close()

            with anyio.CancelScope(shield=True):
                await self.runner_process.stop()
                logger.info(
                    f"Runner process successfully terminated: {self.runner_process.exitcode}"
                )

    def shutdown(self):
        self._tg.cancel_tasks()

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
        self.in_progress[task.task_id] = task
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            self.in_progress.pop(task.task_id, None)
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        await event.wait()

    async def cancel_task(self, task_id: TaskId):
        if task_id in self.completed:
            logger.info(f"Unable to cancel {task_id} as it has been completed")
            self.cancelled.add(task_id)
            return
        self.cancelled.add(task_id)
        with anyio.move_on_after(0.5) as scope:
            try:
                await self._cancel_sender.send_async(task_id)
            except ClosedResourceError:
                # typically occurs when trying to shut down a failed instance
                logger.warning(
                    f"Cancelling task {task_id} failed, runner closed communication"
                )
        if scope.cancel_called:
            logger.error("RunnerSupervisor cancel pipe blocked")
            await self._check_runner(TimeoutError("cancel pipe blocked"))

    async def _forward_events(self):
        try:
            with self._ev_recv as events:
                async for event in events:
                    if isinstance(event, RunnerTerminationError):
                        # try to get exception if possible
                        await self._check_runner(event)
                        break
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
                        self.in_progress.pop(event.task_id, None)
                        self.completed.add(event.task_id)
                    await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError):
            # this is the happy path shutdown - we don't need to spam log with it
            await self._check_runner()
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if not self.runner_process.is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))

    async def _check_runner(
        self, e: RunnerTerminationError | Exception | None = None
    ) -> None:
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info("Checking runner's status")
        if self.runner_process.is_alive():
            logger.info("Runner was found to be alive, stopping process")
            with anyio.CancelScope(shield=True):
                await self.runner_process.stop()
        rc = self.runner_process.exitcode
        logger.info(f"Runner exited with exit code {rc}")

        # If exit code is 0 then the transient errors were recoverable, meaning we don't need runner diagnostics
        if rc == 0:
            return

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                if (description := signal.strsignal(sig)) is not None:
                    cause = f"signal={sig} ({description})"
                else:
                    cause = f"signal={sig}"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause: str = f"exitcode={rc}"

        if e is not None:
            # Record how runner shut down, try exception, resort to RunnerTerminationError fallback
            if isinstance(e, Exception):
                logger.opt(exception=e).error(f"Runner terminated with {cause}")
            else:
                cause = f"{cause}\nRunner error: {e}"
                logger.error(f"Runner terminated with {cause}")
        else:
            logger.error(f"Runner terminated with {cause}")

        diagnostics = [
            d
            for d in self._runner_stdio_handler.diagnostics.diagnostics()
            if not isinstance(d, RunnerUnknown)
        ]
        for task in self.in_progress.values():
            if isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
                with anyio.CancelScope(shield=True):
                    await self._event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                model=self.shard_metadata.model_card.model_id,
                                diagnostics=diagnostics,
                                error_message=(
                                    "Runner shutdown before completing command "
                                    f"({cause})"
                                ),
                            ),
                        )
                    )

        try:
            self.status = RunnerFailed(
                error_message=f"Terminated ({cause})", diagnostics=diagnostics
            )
            with anyio.CancelScope(shield=True):
                await self._event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=self.bound_instance.bound_runner_id,
                        runner_status=self.status,
                    )
                )
        except (ClosedResourceError, BrokenResourceError):
            logger.warning(
                "Event sender already closed, unable to report runner failure"
            )
        self.shutdown()
