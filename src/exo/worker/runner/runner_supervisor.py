from __future__ import annotations

import contextlib
import multiprocessing
import signal
from dataclasses import dataclass, field
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    to_thread,
)
from loguru import logger

from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.worker.runner.bootstrap import entrypoint

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5
HEALTH_CHECK_INTERVAL_SECONDS = 1
HEARTBEAT_STALE_CHECKS = 10


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _cancel_sender: MpSender[TaskId]
    _heartbeat: Synchronized[int]
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _death_handled: bool = field(default=False, init=False)
    _last_heartbeat_value: int = field(default=0, init=False)
    _heartbeat_stale_count: int = field(default=0, init=False)

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

        heartbeat: Synchronized[int] = multiprocessing.Value("Q", 0)

        runner_process = Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                cancel_recv,
                logger,
                heartbeat,
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
            _heartbeat=heartbeat,
        )

        return self

    async def run(self):
        self.runner_process.start()
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._forward_events)
            tg.start_soon(self._health_check, tg.cancel_scope)

    def shutdown(self):
        logger.info("Runner supervisor shutting down")
        self._ev_recv.close()
        self._task_sender.close()
        self._event_sender.close()
        self._cancel_sender.send(TaskId("CANCEL_CURRENT_TASK"))
        self._cancel_sender.close()
        self.runner_process.join(1)
        if not self.runner_process.is_alive():
            logger.info("Runner process succesfully terminated")
            return

        # This is overkill but it's not technically bad, just unnecessary.
        logger.warning("Runner process didn't shutdown succesfully, terminating")
        self.runner_process.terminate()
        self.runner_process.join(1)
        if not self.runner_process.is_alive():
            return

        logger.critical("Runner process didn't respond to SIGTERM, killing")
        self.runner_process.kill()

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
        with self._ev_recv as events:
            try:
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
                if not self._death_handled:
                    self._death_handled = True
                    await self._check_runner(e)
                    for tid in self.pending:
                        self.pending[tid].set()

    async def _health_check(self, cancel_scope: CancelScope) -> None:
        """Periodically check if the runner process is alive and responsive.

        Detects two failure modes:
        1. Process death (e.g. OOM kill) without cleanly closing the event
           channel, which would leave _forward_events blocked on queue.get().
        2. Unresponsive process (e.g. frozen by OS memory pressure, deadlock)
           detected via a stale heartbeat counter.
        """
        while True:
            await anyio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

            if not self.runner_process.is_alive():
                self._handle_process_exit(cancel_scope)
                return

            # Check heartbeat counter — if it hasn't changed between
            # consecutive checks, the subprocess may be frozen.
            current = self._heartbeat.value
            if current > 0:
                if current == self._last_heartbeat_value:
                    self._heartbeat_stale_count += 1
                    if self._heartbeat_stale_count >= HEARTBEAT_STALE_CHECKS:
                        logger.error(
                            f"Health check: runner process unresponsive "
                            f"(heartbeat stale for {self._heartbeat_stale_count} checks), killing"
                        )
                        self._handle_unresponsive(cancel_scope)
                        return
                else:
                    self._heartbeat_stale_count = 0
                self._last_heartbeat_value = current

    def _handle_process_exit(self, cancel_scope: CancelScope) -> None:
        """Handle runner process that has exited."""
        if not self._death_handled:
            self._death_handled = True
            if isinstance(
                self.status, (RunnerShutdown, RunnerShuttingDown, RunnerFailed)
            ):
                logger.info("Health check: runner process exited (expected)")
            else:
                rc = self.runner_process.exitcode
                if isinstance(rc, int) and rc < 0:
                    sig = -rc
                    try:
                        cause = f"signal={sig} ({signal.strsignal(sig)})"
                    except Exception:
                        cause = f"signal={sig}"
                else:
                    cause = f"exitcode={rc}"

                logger.error(
                    f"Health check: runner process died unexpectedly ({cause})"
                )
                self._event_sender.send_nowait(
                    RunnerStatusUpdated(
                        runner_id=self.bound_instance.bound_runner_id,
                        runner_status=RunnerFailed(
                            error_message=f"Terminated ({cause})"
                        ),
                    )
                )
                self.shutdown()

            for tid in self.pending:
                self.pending[tid].set()

        cancel_scope.cancel()

    def _handle_unresponsive(self, cancel_scope: CancelScope) -> None:
        """Handle runner process that is alive but unresponsive."""
        if not self._death_handled:
            self._death_handled = True
            self._event_sender.send_nowait(
                RunnerStatusUpdated(
                    runner_id=self.bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(
                        error_message="Runner process unresponsive (heartbeat timeout)"
                    ),
                )
            )
            for tid in self.pending:
                self.pending[tid].set()
            self.shutdown()

        cancel_scope.cancel()

    def __del__(self) -> None:
        if self.runner_process.is_alive():
            logger.warning("RunnerSupervisor was not stopped cleanly.")
            with contextlib.suppress(ValueError):
                self.runner_process.kill()

    async def _check_runner(self, e: Exception) -> None:
        logger.info("Checking runner's status")
        if self.runner_process.is_alive():
            logger.info("Runner was found to be alive, attempting to join process")
            await to_thread.run_sync(self.runner_process.join, 1)
        rc = self.runner_process.exitcode
        logger.info(f"RunnerSupervisor exited with exit code {rc}")
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

        logger.opt(exception=e).error(f"Runner terminated ({cause})")

        await self._event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=f"Terminated ({cause})"),
            )
        )
        self.shutdown()
