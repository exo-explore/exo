import contextlib
import signal
import sys
from dataclasses import dataclass, field
from multiprocessing import Process
from typing import Self

import anyio
import psutil
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    create_task_group,
    current_time,
    to_thread,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.shared.types.events import Event, RunnerStatusUpdated, TaskAcknowledged
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerError,
    RunnerFailed,
    RunnerStatus,
    RunnerWaitingForModel,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.worker.runner.bootstrap import entrypoint
from exo.worker.runner.utils import (
    get_weights_size,
)

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
    status: RunnerStatus = field(default_factory=RunnerWaitingForModel, init=False)
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

        """ --- not doing this for now
        with tempfile.NamedTemporaryFile(
            prefix="child_stderr_", suffix=".log", delete=False
        ) as tmp:
            err_path = tmp.name

        """
        runner_process = Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                # err_path,
                logger,
            ),
            daemon=True,
        )

        shard_metadata = bound_instance.bound_shard()

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
        self.runner_process.kill()
        await to_thread.run_sync(self.runner_process.join)

    async def start_task(self, task: Task, event: anyio.Event):
        self.pending[task.task_id] = event
        self._task_sender.send(task)

    async def _forward_events(self):
        with self._ev_recv as events:
            while True:
                try:
                    event = await events.receive_async()
                except (ClosedResourceError, BrokenResourceError, EndOfStream):
                    await self._check_runner()
                    break
                if isinstance(event, RunnerStatusUpdated):
                    self.status = event.runner_status
                if isinstance(event, TaskAcknowledged):
                    self.pending.pop(event.task_id).set()
                    continue
                await self._event_sender.send(event)

    async def shutdown(self) -> None:
        assert self._tg
        self._tg.cancel_scope.cancel()

        required_memory_bytes = get_weights_size(self.shard_metadata).in_bytes
        start_time = current_time()
        while True:
            available_memory_bytes = psutil.virtual_memory().available
            if available_memory_bytes >= required_memory_bytes:
                break
            if current_time() - start_time > 30.0:
                logger.warning("Runner memory not released after 30 seconds - exiting")
                break
            await anyio.sleep(1)

    def __del__(self) -> None:
        if self.runner_process.is_alive():
            logger.warning("RunnerSupervisor was not stopped cleanly.")
            with contextlib.suppress(ValueError):
                self.runner_process.kill()

    async def _check_runner(self) -> RunnerError | None:
        rc = self.runner_process.exitcode
        if rc == 0:
            logger.warning("Runner closed communication without terminating process")

        """ --- not doing this anymore
        try:
            with open(self.err_path, "r", errors="replace") as f:
                captured = f.read()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(self.err_path)
        """

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"exitcode={rc}"

        logger.opt(exception=sys.exception()).error(f"Runner terminated ({cause})")

        await self._event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=f"Terminated ({cause})"),
            )
        )
        await self.shutdown()
