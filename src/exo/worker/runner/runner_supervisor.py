import contextlib
import signal
from dataclasses import dataclass, field
from multiprocessing import Process
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    create_task_group,
    to_thread,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.shared.types.events import Event, RunnerStatusUpdated, TaskAcknowledged
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerStatus,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.worker.runner.bootstrap import entrypoint

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    # err_path: str
    _tg: TaskGroup | None = field(default=None, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event]()
        # A task is kind of a runner command
        task_sender, task_recv = mp_channel[Task]()

        runner_process = Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
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
            _event_sender=event_sender,
            # err_path=err_path,
        )

        return self

    async def run(self):
        self.runner_process.start()
        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._forward_events)

        self._ev_recv.close()
        self._task_sender.close()
        self._event_sender.close()
        await to_thread.run_sync(self.runner_process.join, 30)
        if not self.runner_process.is_alive():
            return

        # This is overkill but it's not technically bad, just unnecessary.
        logger.warning("Runner process didn't shutdown succesfully, terminating")
        self.runner_process.terminate()
        await to_thread.run_sync(self.runner_process.join, 5)
        if not self.runner_process.is_alive():
            return

        logger.critical("Runner process didn't respond to SIGTERM, killing")
        self.runner_process.kill()

        await to_thread.run_sync(self.runner_process.join, 5)
        if not self.runner_process.is_alive():
            return

        logger.critical(
            "Runner process didn't respond to SIGKILL. System resources may have leaked"
        )

    def shutdown(self):
        assert self._tg
        self._tg.cancel_scope.cancel()

    async def start_task(self, task: Task):
        logger.info(f"Starting task {task}")
        event = anyio.Event()
        self.pending[task.task_id] = event
        try:
            self._task_sender.send(task)
        except ClosedResourceError:
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        await event.wait()
        logger.info(f"Finished task {task}")

    async def _forward_events(self):
        with self._ev_recv as events:
            try:
                async for event in events:
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                    if isinstance(event, TaskAcknowledged):
                        self.pending.pop(event.task_id).set()
                        continue
                    await self._event_sender.send(event)
            except (ClosedResourceError, BrokenResourceError) as e:
                await self._check_runner(e)
                for tid in self.pending:
                    self.pending[tid].set()

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
