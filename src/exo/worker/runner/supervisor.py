import codecs
import contextlib
import signal
import sys
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Self, final

import anyio
from anyio import (
    AsyncFile,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
)
from loguru import logger

from exo.shared.constants import EXO_RUNNER_STDERR_LOG, EXO_RUNNER_STDOUT_LOG
from exo.shared.models.model_cards import ModelCard
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
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
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
    RunnerRingTransportError,
    RunnerUnknown,
    metal_out_of_memory,
)

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5
# How often the supervisor checks that its runner is alive and can still reach the others
RUNNER_WATCH_INTERVAL = 5.0
# A generating runner reports a token every step and progress every prefill chunk, so this
# long without a word means it is stuck (its instance's runners can wait on each other forever)
RUNNER_STALL_TIMEOUT = 90.0

# How long a runner may take over each step on its way to serving. Every step waits on things
# outside the runner (the instance's other runners, the disk, MLX), and a runner that will never
# get through one would hold its instance, and every request for it, forever: one that takes
# longer is stopped and failed, and the worker starts it again (with backoff, up to
# EXO_MAX_INSTANCE_RETRIES times). The bounds are far above what these steps take, to catch a
# runner that is stuck, never one that is slow.
#
# Starting the process and importing MLX and the model code: seconds, longer from a cold disk.
RUNNER_START_TIMEOUT = 300.0
# A runner is only told to connect once all of its instance's runners have started, so it waits
# for the others to be told as well (a round trip through the master) and for the sockets or RDMA
# connections to come up: seconds.
RUNNER_CONNECT_TIMEOUT = 120.0
# Loading reads the runner's share of the weights, then waits for the slowest runner of the
# instance. This allows for reading the whole model at 20 MB/s, slower than a hard disk on USB 2,
# plus RUNNER_LOAD_TIMEOUT_BASE for the tokenizer and the rest.
RUNNER_LOAD_TIMEOUT_BASE = 300.0
SLOWEST_DISK_BYTES_PER_SECOND = 20_000_000
# Warming up runs a short prompt and 50 tokens through the instance, and each token reads every
# weight, so it takes time in proportion to the model, at a rate that depends on where MLX
# computes. On a Mac's GPU: an M4 Pro decodes a 0.7 GB model at about 260 tokens a second, so 50
# tokens take about 0.3s per GB of model; this allows 5s per GB (a 100 GB model gets about 18
# minutes). On the CPU, MLX took about 4 minutes per GB of model on a 20-core Arm machine; this
# allows 1000s per GB. Both plus RUNNER_WARMUP_TIMEOUT_BASE for compiling kernels and the barriers.
RUNNER_WARMUP_TIMEOUT_BASE = 600.0
SLOWEST_METAL_WARMUP_BYTES_PER_SECOND = 200_000_000
SLOWEST_CPU_WARMUP_BYTES_PER_SECOND = 1_000_000
# On a Mac, MLX computes on the GPU through Metal (exo runs on Apple silicon only). Elsewhere a
# runner computes on the CPU, or on a GPU with MLX's CUDA build; telling which would mean importing
# MLX, so the bound allows for the CPU.
MLX_USES_METAL = sys.platform == "darwin"


def runner_load_timeout(model_card: ModelCard) -> float:
    return (
        RUNNER_LOAD_TIMEOUT_BASE
        + model_card.storage_size.in_bytes / SLOWEST_DISK_BYTES_PER_SECOND
    )


def runner_warmup_timeout(model_card: ModelCard, on_metal: bool) -> float:
    rate = (
        SLOWEST_METAL_WARMUP_BYTES_PER_SECOND
        if on_metal
        else SLOWEST_CPU_WARMUP_BYTES_PER_SECOND
    )
    return RUNNER_WARMUP_TIMEOUT_BASE + model_card.storage_size.in_bytes / rate


@final
class RunnerStoppedError(Exception):
    """Why the supervisor stopped a runner that was still running"""


@dataclass(frozen=True)
class _Step:
    """A step on the runner's way to serving that the supervisor is waiting for it to finish"""

    description: str
    finished_at: type[RunnerStatus]
    timeout: float
    deadline: float

    @classmethod
    def start(
        cls, description: str, finished_at: type[RunnerStatus], timeout: float
    ) -> "_Step":
        return cls(description, finished_at, timeout, anyio.current_time() + timeout)


@dataclass(eq=False)
class RunnerStdioHandler:
    _stdout_rx: Receiver[bytes]
    _stderr_rx: Receiver[bytes]
    _stdout_log: AsyncFile[str]
    _stderr_log: AsyncFile[str]
    diagnostics: RunnerDiagnosticCollector = field(
        default_factory=RunnerDiagnosticCollector
    )

    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)

    @classmethod
    async def create(
        cls,
        *,
        stdout_rx: Receiver[bytes],
        stderr_rx: Receiver[bytes],
        stdout_log_path: PathLike[str] = EXO_RUNNER_STDOUT_LOG,
        stderr_log_path: PathLike[str] = EXO_RUNNER_STDERR_LOG,
    ) -> Self:
        # these are append only logs used to gather data for log template mining
        #
        # TODO: in the future use [Drain3](https://github.com/logpai/Drain3)
        #       to mine these logs
        ensure_parent_directory_exists(stdout_log_path)
        ensure_parent_directory_exists(stderr_log_path)
        stdout_log = await anyio.open_file(stdout_log_path, "a")
        stderr_log = await anyio.open_file(stderr_log_path, "a")

        # instantiate and return
        self = cls(
            _stdout_rx=stdout_rx,
            _stderr_rx=stderr_rx,
            _stdout_log=stdout_log,
            _stderr_log=stderr_log,
        )
        return self

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(  # pyright: ignore[reportUnknownArgumentType]
                    self._handle_runner_output,
                    self._stdout_rx,
                    self._stdout_log,
                    lambda line: logger.info(f"Runner stdout: {line}"),  # pyright: ignore[reportUnknownLambdaType]
                    lambda _: None,  # pyright: ignore[reportUnknownLambdaType]
                )
                tg.start_soon(  # pyright: ignore[reportUnknownArgumentType]
                    self._handle_runner_output,
                    self._stderr_rx,
                    self._stderr_log,
                    lambda line: logger.warning(f"Runner stderr: {line}"),  # pyright: ignore[reportUnknownLambdaType]
                    self.diagnostics.record_line,
                )
        finally:
            with CancelScope(shield=True):
                await self._stdout_log.aclose()
                await self._stderr_log.aclose()

    async def _handle_runner_output(
        self,
        rx: Receiver[bytes],
        logfile: AsyncFile[str],
        log_line: Callable[[str], None],
        record_diagnostic_line: Callable[[str], None],
    ):
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
            log_line(line)
            record_diagnostic_line(line)

        async def handle_text(text: str):
            nonlocal pending_line

            if not text:
                return

            await logfile.write(text)
            await logfile.flush()

            # newline buffering
            pending_line += text
            lines = pending_line.split("\n")
            pending_line = lines.pop()

            for line in lines:
                await handle_line(line)

        try:
            with rx:
                async for chunk in rx:
                    await handle_text(decoder.decode(chunk, final=False))
        except (ClosedResourceError, BrokenResourceError):
            logger.warning("Runner stdio stream closed before clean EOF")
        finally:
            with CancelScope(shield=True):
                await handle_text(decoder.decode(b"", final=True))
                await logfile.flush()

                if pending_line:
                    await handle_line(pending_line)
                    pending_line = ""


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: AsyncProcess
    _runner_stdio_handler: RunnerStdioHandler
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
    # When the runner last sent anything, for noticing a runner that is stuck mid-generation
    _last_heard: float = field(default_factory=lambda: anyio.current_time(), init=False)
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )
    # The step the runner is in on its way to serving, if it's in one
    _step: _Step | None = field(default=None, init=False)

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
    ) -> Self:
        # The runner blocks in native code that holds the GIL: MLX's distributed init holds it
        # while it waits for the other ranks. Left to a background thread, the status and the
        # acknowledgement the runner sends just before would wait for the call to return, and
        # the other ranks wait for that status before they connect.
        ev_send, ev_recv = mp_channel[Event | RunnerTerminationError](
            send_immediately=True
        )
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
            stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            _runner_stdio_handler=runner_stdio_handler,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )

        return self

    async def run(self):
        try:
            async with self._tg as tg:
                self._step = _Step.start("start", RunnerIdle, RUNNER_START_TIMEOUT)
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
        """Send the task to the runner, without waiting for the runner to acknowledge it.

        The caller is the worker's planning for every runner on this node, and a runner can
        take minutes to pick up a generation task behind a long prompt, or never acknowledge a
        task at all if it's stuck. Generation tasks are tracked by id in in_progress; until the
        runner acknowledges any other task, awaiting_acknowledgement holds back the planning
        that would send it again.
        """
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
        self.pending[task.task_id] = anyio.Event()
        self.in_progress[task.task_id] = task
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            self.in_progress.pop(task.task_id, None)
            self.pending.pop(task.task_id, None)
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        match task:
            case ConnectToGroup():
                self._step = _Step.start(
                    "connect to the instance's other runners",
                    RunnerConnected,
                    RUNNER_CONNECT_TIMEOUT,
                )
            case LoadModel():
                self._step = _Step.start(
                    "load the model",
                    RunnerLoaded,
                    runner_load_timeout(self.shard_metadata.model_card),
                )
            case StartWarmup():
                self._step = _Step.start(
                    "warm up",
                    RunnerReady,
                    runner_warmup_timeout(
                        self.shard_metadata.model_card, on_metal=MLX_USES_METAL
                    ),
                )
            case _:
                pass

    @property
    def awaiting_acknowledgement(self) -> bool:
        """Whether the runner has yet to pick up a task that changes its status (any task but a
        generation task). It changes its status as it picks the task up, so until then its
        status is still the one from before the task: planning from it would send it again."""
        return any(
            not acknowledged.is_set()
            and not isinstance(
                self.in_progress.get(task_id),
                (TextGeneration, ImageGeneration, ImageEdits),
            )
            for task_id, acknowledged in self.pending.items()
        )

    async def wait_for_acknowledgement(self, task_id: TaskId) -> None:
        if (acknowledged := self.pending.get(task_id)) is not None:
            await acknowledged.wait()

    async def cancel_task(self, task_id: TaskId):
        if task_id in self.completed:
            logger.info(f"Unable to cancel {task_id} as it has been completed")
            self.cancelled.add(task_id)
            return
        self.cancelled.add(task_id)
        acknowledged = self.pending.get(task_id)
        if acknowledged is not None and not acknowledged.is_set():
            # The runner hasn't picked the task up yet, and it ignores cancellations for
            # tasks it doesn't know, so send this one once it has. (A runner that has gone
            # away releases every waiting task.)
            self._tg.start_soon(self._cancel_when_acknowledged, task_id, acknowledged)
            return
        await self._send_cancel(task_id)

    async def _cancel_when_acknowledged(
        self, task_id: TaskId, acknowledged: anyio.Event
    ) -> None:
        await acknowledged.wait()
        await self._send_cancel(task_id)

    async def _send_cancel(self, task_id: TaskId) -> None:
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
                    self._last_heard = anyio.current_time()
                    if isinstance(event, RunnerTerminationError):
                        # try to get exception if possible
                        await self._check_runner(event)
                        break
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                        if self._step is not None and isinstance(
                            self.status, self._step.finished_at
                        ):
                            self._step = None
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
                await anyio.sleep(RUNNER_WATCH_INTERVAL)
                if not self.runner_process.is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))
                elif self._lost_its_peers():
                    # MLX gave up on the connection to the other runners of this instance, but
                    # the process lives on, stuck: every request on the instance would hang.
                    # Stopping it fails its requests with an error, and the runner and its
                    # peers are recreated.
                    await self._check_runner(
                        RunnerStoppedError(
                            "Lost the connection to the instance's other runners"
                        )
                    )
                elif self._stuck():
                    # Stopping it fails its requests with an error; the runner and the
                    # instance's other runners are then recreated.
                    await self._check_runner(
                        RunnerStoppedError(
                            f"Runner made no progress for {RUNNER_STALL_TIMEOUT:.0f}s while generating"
                        )
                    )
                elif (step := self._overdue_step()) is not None:
                    await self._check_runner(
                        RunnerStoppedError(
                            f"Runner took more than {step.timeout:.0f}s to {step.description}"
                        )
                    )

    def _overdue_step(self) -> _Step | None:
        if self._step is not None and anyio.current_time() > self._step.deadline:
            return self._step
        return None

    def _stuck(self) -> bool:
        """Whether the runner is generating but has gone silent. Only the first rank of an
        instance reports tokens and prefill progress, so only its supervisor can tell."""
        return (
            self.bound_instance.bound_shard.device_rank == 0
            and isinstance(self.status, RunnerRunning)
            and any(isinstance(t, TextGeneration) for t in self.in_progress.values())
            and anyio.current_time() - self._last_heard > RUNNER_STALL_TIMEOUT
        )

    def _lost_its_peers(self) -> bool:
        return any(
            isinstance(diagnostic, RunnerRingTransportError)
            for diagnostic in self._runner_stdio_handler.diagnostics.diagnostics()
        )

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
            if isinstance(e, RunnerStoppedError):
                # Stopped for a reason of the supervisor's own: the reason is what to report
                cause = f"{cause}\nRunner error: {e}"
                logger.error(f"Runner terminated with {cause}")
            elif isinstance(e, Exception):
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
        # A GPU out of memory can also come back as the runner's own error
        if isinstance(e, RunnerTerminationError) and (
            out_of_memory := metal_out_of_memory(
                e.exception_message, (f"{e.exception_type}: {e.exception_message}",)
            )
        ):
            diagnostics.append(out_of_memory)
        # Each send yields to the event loop, where the runner's last results can still finish
        # requests (and take them out of in_progress), so walk a copy and skip those.
        for task in list(self.in_progress.values()):
            if task.task_id not in self.in_progress:
                continue
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
