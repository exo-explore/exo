import codecs
import contextlib
import signal
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Self

import anyio
from anyio import (
    AsyncFile,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
)
from exo_rs import (
    LVPublisher,
    LVSubscriber,
    TaskChunkSender,
    TaskRequest,
    TaskResponder,
)
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from exo.shared.constants import EXO_RUNNER_STDERR_LOG, EXO_RUNNER_STDOUT_LOG
from exo.shared.types.chunks import ErrorChunk, PrefillProgressChunk
from exo.shared.types.commands import (
    Command,
    DeleteInstance,
    TaskCancelled,
    TaskFinished,
)
from exo.shared.types.commands import (
    ImageEdits as ImageEditsCommand,
)
from exo.shared.types.commands import (
    ImageGeneration as ImageGenerationCommand,
)
from exo.shared.types.commands import (
    TextGeneration as TextGenerationCommand,
)
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
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
    RunnerReady,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
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
type BridgeTask = TextGeneration | ImageGeneration | ImageEdits
_BRIDGE_COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)
_BRIDGE_TASK_ADAPTER: TypeAdapter[BridgeTask] = TypeAdapter(BridgeTask)


def _task_assignment_ids(key: str) -> tuple[str, TaskId] | None:
    prefix = "task_assignments/"
    if not key.startswith(prefix):
        return None

    suffix = key.removeprefix(prefix)
    parts = suffix.split("/", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1] or "/" in parts[1]:
        return None

    return parts[0], TaskId(parts[1])


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
    bound_instance: BoundInstance
    runner_process: AsyncProcess
    _runner_stdio_handler: RunnerStdioHandler
    initialize_timeout: float
    _ev_recv: MpReceiver[Event | RunnerTerminationError]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _task_responder: TaskResponder | None
    _task_assignment_subscriber: LVSubscriber
    runner_status_publisher: LVPublisher
    _assigned_tasks: dict[TaskId, BridgeTask] = field(default_factory=dict, init=False)
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    bridge_tasks: dict[TaskId, BridgeTask] = field(default_factory=dict, init=False)
    bridge_command_tasks: dict[CommandId, TaskId] = field(
        default_factory=dict, init=False
    )
    bridge_chunk_senders: dict[CommandId, TaskChunkSender] = field(
        default_factory=dict, init=False
    )
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )

    @classmethod
    async def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        task_assignment_subscriber: LVSubscriber,
        runner_status_publisher: LVPublisher,
        task_responder: TaskResponder | None,
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
            stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
        )

        self = cls(
            bound_instance=bound_instance,
            runner_process=runner_process,
            _runner_stdio_handler=runner_stdio_handler,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
            _task_responder=task_responder,
            _task_assignment_subscriber=task_assignment_subscriber,
            runner_status_publisher=runner_status_publisher,
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
                if self._task_responder is not None:
                    tg.start_soon(self._run_task_responder, self._task_responder)
                tg.start_soon(
                    self._run_task_assignment_subscriber,
                    self._task_assignment_subscriber,
                )
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
                await self.runner_status_publisher.delete()
                await self.runner_process.stop()
                logger.info(
                    f"Runner process successfully terminated: {self.runner_process.exitcode}"
                )

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _run_task_assignment_subscriber(self, subscriber: LVSubscriber) -> None:
        instance_id = self.bound_instance.instance.instance_id
        while (received := await subscriber.recv()) is not None:
            key, payload = received
            if payload is None:
                continue
            if (ids := _task_assignment_ids(key)) is None:
                continue
            assigned_instance_id, assigned_task_id = ids
            if assigned_instance_id != instance_id:
                continue

            if payload == "":
                self._assigned_tasks.pop(assigned_task_id, None)
                continue

            try:
                task = _BRIDGE_TASK_ADAPTER.validate_json(payload)
            except ValidationError:
                logger.warning(f"Ignoring invalid task assignment from {key}")
                continue

            if task.instance_id != instance_id or task.task_id != assigned_task_id:
                logger.warning(f"Ignoring mismatched task assignment from {key}")
                continue

            self._assigned_tasks[task.task_id] = task
            await self._reconcile_assigned_tasks()

    async def _reconcile_assigned_tasks(self) -> None:
        if not isinstance(self.status, (RunnerReady, RunnerRunning)):
            return

        for task in list(self._assigned_tasks.values()):
            if task.task_id in self.in_progress or task.task_id in self.completed:
                continue
            await self.start_task(task)

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
                    match event:
                        case RunnerTerminationError():
                            await self._check_runner(event)
                            break

                        case RunnerStatusUpdated(runner_status=runner_status):
                            self.status = runner_status
                            await self.runner_status_publisher.put(
                                self.status.model_dump_json()
                            )
                            await self._event_sender.send(event)
                            await self._reconcile_assigned_tasks()

                        case TaskAcknowledged(task_id=task_id):
                            self.pending.pop(task_id).set()

                        case TaskStatusUpdated(
                            task_id=task_id, task_status=TaskStatus.Complete
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
                            self.in_progress.pop(task_id, None)
                            self.completed.add(task_id)
                            self._assigned_tasks.pop(task_id, None)
                            await self._event_sender.send(event)
                            if task_id in self.bridge_tasks:
                                await self._finish_bridge_task_id(task_id)

                        case ChunkGenerated(command_id=command_id, chunk=chunk):
                            task_id = self.bridge_command_tasks.get(command_id)
                            chunk_sender = self.bridge_chunk_senders.get(command_id)
                            if task_id is None or chunk_sender is None:
                                logger.debug(
                                    f"Dropping bridge chunk for inactive command {command_id}"
                                )
                                continue

                            await chunk_sender.send(chunk.model_dump_json())
                            if (
                                not isinstance(chunk, PrefillProgressChunk)
                                and chunk.finish_reason is not None
                            ):
                                self.bridge_chunk_senders.pop(command_id, None)

                        case _:
                            await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError):
            # this is the happy path shutdown - we don't need to spam log with it
            await self._check_runner()
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _run_task_responder(self, responder: TaskResponder) -> None:
        while True:
            received = await responder.recv()
            if received is None:
                return
            request, chunk_sender, payload = received
            if payload is None:
                request.reply_err("Task command query did not include a payload")
                continue

            try:
                command = _BRIDGE_COMMAND_ADAPTER.validate_json(payload)
                match command:
                    case TextGenerationCommand():
                        await self._submit_bridge_task(request, chunk_sender, command)
                    case ImageGenerationCommand():
                        await self._submit_bridge_task(request, chunk_sender, command)
                    case ImageEditsCommand():
                        await self._submit_bridge_task(request, chunk_sender, command)
                    case TaskCancelled(cancelled_command_id=command_id):
                        await self._cancel_bridge_task(command_id)
                        request.reply(command_id)
                    case TaskFinished(finished_command_id=command_id):
                        await self._finish_bridge_task(command_id)
                        request.reply(command_id)
                    case DeleteInstance():
                        self.shutdown()
                    case _:
                        request.reply_err(f"Unsupported bridge command: {command}")
            except Exception as exception:
                logger.opt(exception=exception).warning(
                    "Failed to admit bridge command"
                )
                request.reply_err(str(exception))

    async def _submit_bridge_task(
        self,
        request: TaskRequest,
        chunk_sender: TaskChunkSender,
        command: TextGenerationCommand | ImageGenerationCommand | ImageEditsCommand,
    ) -> None:
        task_id = TaskId()
        match command:
            case TextGenerationCommand():
                task = TextGeneration(
                    task_id=task_id,
                    command_id=command.command_id,
                    instance_id=self.bound_instance.instance.instance_id,
                    task_status=TaskStatus.Pending,
                    task_params=command.task_params,
                )
            case ImageGenerationCommand():
                task = ImageGeneration(
                    task_id=task_id,
                    command_id=command.command_id,
                    instance_id=self.bound_instance.instance.instance_id,
                    task_status=TaskStatus.Pending,
                    task_params=command.task_params,
                )
            case ImageEditsCommand():
                task = ImageEdits(
                    task_id=task_id,
                    command_id=command.command_id,
                    instance_id=self.bound_instance.instance.instance_id,
                    task_status=TaskStatus.Pending,
                    task_params=command.task_params,
                )

        self.bridge_tasks[task.task_id] = task
        self.bridge_command_tasks[command.command_id] = task.task_id
        self.bridge_chunk_senders[command.command_id] = chunk_sender
        await self._event_sender.send(TaskCreated(task_id=task.task_id, task=task))
        assert self._task_responder is not None
        await self._task_responder.assign_task(task.task_id, task.model_dump_json())
        request.reply(command.command_id)

    async def _cancel_bridge_task(self, command_id: CommandId) -> None:
        task_id = self.bridge_command_tasks.get(command_id)
        if task_id is None:
            logger.warning(f"Unable to cancel unknown bridge command {command_id}")
            return
        await self.cancel_task(task_id)
        await self._event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled)
        )
        await self._finish_bridge_task_id(task_id)

    async def _finish_bridge_task(self, command_id: CommandId) -> None:
        task_id = self.bridge_command_tasks.get(command_id)
        if task_id is None:
            logger.warning(f"Unable to finish unknown bridge command {command_id}")
            return
        await self._finish_bridge_task_id(task_id)

    async def _finish_bridge_task_id(self, task_id: TaskId) -> None:
        task = self.bridge_tasks.pop(task_id, None)
        if task is None:
            logger.warning(f"Unable to finish unknown bridge task {task_id}")
            return
        self.bridge_command_tasks.pop(task.command_id, None)
        self.bridge_chunk_senders.pop(task.command_id, None)
        if self._task_responder is not None:
            await self._task_responder.unassign_task(task_id)

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
                                model=self.bound_instance.bound_shard.model_card.model_id,
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
