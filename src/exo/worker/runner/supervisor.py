import contextlib
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from typing import Self

import anyio
import psutil
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    current_time,
    to_thread,
)
from loguru import logger

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
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.utils.task_group import TaskGroup
from exo.worker.runner.bootstrap import entrypoint

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: mp.Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _started_at: float | None = field(default=None, init=False)
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event]()
        task_sender, task_recv = mp_channel[Task]()
        cancel_sender, cancel_recv = mp_channel[TaskId]()

        runner_process = mp.Process(
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

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _cancel_sender=cancel_sender,
            _event_sender=event_sender,
        )
        logger.info(
            "Created runner supervisor "
            f"{self._runner_context()} model_id={self.shard_metadata.model_card.model_id}"
        )

        return self

    async def run(self):
        self.runner_process.start()
        self._started_at = current_time()
        logger.info(
            "Runner process started "
            f"{self._runner_context()} pid={self.runner_process.pid} "
            f"model_id={self.shard_metadata.model_card.model_id}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._watch_runner)
                tg.start_soon(self._forward_events)
        finally:
            logger.info(
                "Runner supervisor shutting down "
                f"{self._runner_context()} pid={self.runner_process.pid} "
                f"rss_mb={self._runner_rss_mb()}"
            )
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

            await to_thread.run_sync(self.runner_process.join, 5)

            if self.runner_process.is_alive():
                logger.warning(
                    "Runner process did not shutdown successfully, terminating "
                    f"{self._runner_context()} pid={self.runner_process.pid} "
                    f"rss_mb={self._runner_rss_mb()}"
                )
                self.runner_process.terminate()
                self.runner_process.join(timeout=10)

                if not self.runner_process.is_alive():
                    logger.warning(
                        "Runner terminated after first SIGTERM "
                        f"{self._runner_context()} pid={self.runner_process.pid}"
                    )

                else:
                    # Try really hard to terminate
                    for i in range(2, 11):
                        self.runner_process.terminate()
                        self.runner_process.join(timeout=2)
                        if not self.runner_process.is_alive():
                            logger.warning(
                                "Runner terminated after repeated SIGTERM "
                                f"{self._runner_context()} attempts={i} "
                                f"pid={self.runner_process.pid}"
                            )
                            break
                    # Try even harder to kill
                    else:
                        logger.critical(
                            "Runner process did not respond to SIGTERM, killing "
                            f"{self._runner_context()} pid={self.runner_process.pid} "
                            f"rss_mb={self._runner_rss_mb()}"
                        )
                        j = 0
                        while self.runner_process.is_alive():
                            j += 1
                            self.runner_process.kill()
                            self.runner_process.join(timeout=5)
                            logger.warning(
                                "Runner kill attempt completed "
                                f"{self._runner_context()} attempts={j} "
                                f"pid={self.runner_process.pid}"
                            )
            else:
                logger.info(
                    "Runner process successfully terminated "
                    f"{self._runner_context()} exitcode={self.runner_process.exitcode} "
                    f"runtime_seconds={self._runtime_seconds()}"
                )

            self.runner_process.close()

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
        logger.info(
            "Starting runner task "
            f"{self._runner_context()} task_id={task.task_id} "
            f"task_type={type(task).__name__} status={type(self.status).__name__}"
        )
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
                    if isinstance(event, RunnerStatusUpdated):
                        logger.info(
                            "Runner status update "
                            f"{self._runner_context()} "
                            f"old_status={type(self.status).__name__} "
                            f"new_status={type(event.runner_status).__name__} "
                            f"pid={self.runner_process.pid} rss_mb={self._runner_rss_mb()}"
                        )
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
        except (ClosedResourceError, BrokenResourceError) as e:
            await self._check_runner(e)
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if not self.runner_process.is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))

    async def _check_runner(self, e: Exception) -> None:
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info(
            "Checking runner status "
            f"{self._runner_context()} pid={self.runner_process.pid} "
            f"rss_mb={self._runner_rss_mb()}"
        )
        if self.runner_process.is_alive():
            logger.info(
                "Runner was found alive, attempting to join process "
                f"{self._runner_context()} pid={self.runner_process.pid}"
            )
            await to_thread.run_sync(self.runner_process.join, 5)
        rc = self.runner_process.exitcode
        logger.info(
            "Runner exited "
            f"{self._runner_context()} exitcode={rc} "
            f"runtime_seconds={self._runtime_seconds()}"
        )
        if rc == 0:
            return

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"exitcode={rc}"

        logger.opt(exception=e).error(
            f"Runner terminated with {cause} {self._runner_context()}"
        )

        for task in self.in_progress.values():
            if isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
                with anyio.CancelScope(shield=True):
                    await self._event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                model=self.shard_metadata.model_card.model_id,
                                error_message=(
                                    "Runner shutdown before completing command "
                                    f"({cause})"
                                ),
                            ),
                        )
                    )

        try:
            self.status = RunnerFailed(error_message=f"Terminated ({cause})")
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

    def _runner_context(self) -> str:
        return (
            f"instance_id={self.bound_instance.instance.instance_id} "
            f"runner_id={self.bound_instance.bound_runner_id} "
            f"node_id={self.bound_instance.bound_node_id}"
        )

    def _runner_rss_mb(self) -> float | None:
        pid = self.runner_process.pid
        if pid is None:
            return None
        try:
            return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 3)
        except psutil.Error:
            return None

    def _runtime_seconds(self) -> float | None:
        if self._started_at is None:
            return None
        return round(current_time() - self._started_at, 3)
