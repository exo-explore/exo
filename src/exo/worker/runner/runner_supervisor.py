import contextlib
import multiprocessing as mp
import os
import signal
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
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


def _sigterm_handler(signum, frame):
    """
    SIGTERM handler: forcibly SIGKILL all direct child processes so that
    orphaned python3 MLX-runner processes do not survive a kickstart.
    Re-raises default SIGTERM so the supervisor itself still exits cleanly.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True, text=True, timeout=2
        )
        for pid_str in result.stdout.strip().splitlines():
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass
    # Restore default and re-raise so the process exits with SIGTERM.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Install at import time so the handler is active for the entire supervisor
# lifetime, including the finally block inside run().
signal.signal(signal.SIGTERM, _sigterm_handler)


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
    _cancel_watch_runner: anyio.CancelScope = field(
        default_factory=anyio.CancelScope, init=False
    )
    _shutdown_requested: bool = field(default=False, init=False)


    def _runner_is_alive(self) -> bool:
        try:
            return self.runner_process.is_alive()
        except ValueError:
            return False

    def _runner_exitcode(self) -> int | None:
        try:
            return self.runner_process.exitcode
        except ValueError:
            return -1


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

        return self

    # ------------------------------------------------------------------
    # Non-blocking helpers — each offloads a blocking call to a thread so
    # the asyncio event loop (and therefore the API server) stays alive.
    # ------------------------------------------------------------------

    async def _join_runner(self, timeout: float) -> None:
        """Join the runner process without blocking the event loop."""
        await to_thread.run_sync(
            lambda: self.runner_process.join(timeout), abandon_on_cancel=True
        )

    async def _terminate_runner(self) -> None:
        """Send SIGTERM to the runner without blocking the event loop."""
        await to_thread.run_sync(
            self.runner_process.terminate, abandon_on_cancel=True
        )

    async def _kill_runner(self) -> None:
        """Send SIGKILL to the runner without blocking the event loop."""
        await to_thread.run_sync(
            self.runner_process.kill, abandon_on_cancel=True
        )

    async def run(self):
        MAX_RESTARTS = 5
        restart_count = 0

        while True:
            self._shutdown_requested = False
            self.runner_process.start()
            try:
                async with self._tg as tg:
                    tg.start_soon(self._watch_runner)
                    tg.start_soon(self._forward_events)
            finally:
                logger.info("Runner supervisor shutting down" if self._shutdown_requested else "Runner process exited unexpectedly, cleaning up")
                if not self._cancel_watch_runner.cancel_called:
                    self._cancel_watch_runner.cancel()
                with contextlib.suppress(ClosedResourceError):
                    self._ev_recv.close()
                with contextlib.suppress(ClosedResourceError):
                    self._task_sender.close()
                # Only close the event sender on intentional shutdown — on crash-restart,
                # keep it open so the rest of exo keeps receiving events after reload
                if self._shutdown_requested:
                    with contextlib.suppress(ClosedResourceError):
                        self._event_sender.close()
                with contextlib.suppress(ClosedResourceError, TimeoutError, Exception):
                    with anyio.move_on_after(2.0):
                        await self._cancel_sender.send_async(CANCEL_ALL_TASKS)
                with contextlib.suppress(ClosedResourceError):
                    self._cancel_sender.close()

                await self._join_runner(5)

                if self._runner_is_alive():
                    logger.warning(
                        "Runner process didn't shutdown succesfully, terminating"
                    )
                    await self._terminate_runner()
                    await self._join_runner(10)

                    if not self._runner_is_alive():
                        logger.warning("Terminated nicely in the first attempt!")
                    else:
                        for i in range(2, 11):
                            await self._terminate_runner()
                            await self._join_runner(2)
                            if not self._runner_is_alive():
                                logger.warning(f"That took {i} attempts :)")
                                break
                        else:
                            logger.critical(
                                "Runner process didn't respond to SIGTERM, killing"
                            )
                            j = 0
                            while self._runner_is_alive():
                                j += 1
                                await self._kill_runner()
                                await self._join_runner(5)
                                logger.warning(f"That took {j} attempts :(")
                else:
                    logger.info("Runner process succesfully terminated")

                self.runner_process.close()

            if self._shutdown_requested:
                logger.info("Runner supervisor: intentional shutdown, not restarting")
                break

            restart_count += 1
            if restart_count > MAX_RESTARTS:
                logger.critical(
                    f"Runner crashed {MAX_RESTARTS} times without recovery, giving up"
                )
                break

            delay = min(2.0 * (2 ** (restart_count - 1)), 10.0)
            logger.warning(
                f"Runner crashed (attempt {restart_count}/{MAX_RESTARTS}), "
                f"restarting in {delay:.0f}s"
            )
            await anyio.sleep(delay)
            self._reset_for_restart()

    def shutdown(self):
        self._shutdown_requested = True
        self._tg.cancel_tasks()

    def _reset_for_restart(self) -> None:
        """Recreate channels and process for a runner restart after an unexpected crash."""
        ev_send, ev_recv = mp_channel[Event]()
        task_sender, task_recv = mp_channel[Task]()
        cancel_sender, cancel_recv = mp_channel[TaskId]()

        self.runner_process = mp.Process(
            target=entrypoint,
            args=(self.bound_instance, ev_send, task_recv, cancel_recv, logger),
            daemon=True,
        )
        self._ev_recv = ev_recv
        self._task_sender = task_sender
        self._cancel_sender = cancel_sender
        self._tg = TaskGroup()
        self._cancel_watch_runner = anyio.CancelScope()
        self.status = RunnerIdle()
        self.pending = {}
        self.in_progress = {}
        self.completed = set()
        self.cancelled = set()
        # _event_sender is intentionally NOT reset — it connects to the rest of exo
        # and must remain open across restarts

    def _cancel_tg(self) -> None:
        """Cancel the running task group without marking this as an intentional shutdown.
        Used by _check_runner to tear down the current run() iteration so the restart
        loop can start a fresh runner subprocess.
        """
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
        except (ClosedResourceError, BrokenResourceError) as e:
            await self._check_runner(e)
        finally:
            for tid in self.pending:
                self.pending[tid].set()

    async def _watch_runner(self) -> None:
        with self._cancel_watch_runner:
            while True:
                await anyio.sleep(5)
                if not self._runner_is_alive():
                    await self._check_runner(RuntimeError("Runner found to be dead"))

    async def _check_runner(self, e: Exception) -> None:
        if not self._cancel_watch_runner.cancel_called:
            self._cancel_watch_runner.cancel()
        logger.info("Checking runner's status")
        if self._runner_is_alive():
            logger.info("Runner was found to be alive, attempting to join process")
            await self._join_runner(5)
        rc = self._runner_exitcode()
        logger.info(f"Runner exited with exit code {rc}")
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

        logger.opt(exception=e).error(f"Runner terminated with {cause}")

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
        self._cancel_tg()
