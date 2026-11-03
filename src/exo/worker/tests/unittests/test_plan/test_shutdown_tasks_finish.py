"""Every Shutdown task the worker creates is finished, so none stays in the cluster state.

When a runner of a multi-node instance fails, each node shuts its runner down with a Shutdown
task, and the instance's runners are started again. Those tasks were never finished: a runner
that had died never got its task (it stayed Pending), one that took it up was no longer
listened to by the time it reported it done (Running), and one that didn't take it up in time
was marked TimedOut. Each runner failure left more of them in the state for good.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import anyio
import pytest
from anyio.abc import TaskStatus as StartedStatus

import exo.worker.main as worker_mod
from exo.shared.apply import apply
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerReady,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Receiver, channel, mp_channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

# A model split across this node (RUNNER_1_ID) and another (RUNNER_2_ID)
INSTANCE = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 2),
        RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, 2),
    },
)
DOWNLOADED = [
    DownloadCompleted(
        node_id=NODE_A,
        shard_metadata=INSTANCE.shard_assignments.runner_to_shard[RUNNER_1_ID],
        total=Memory(),
    )
]
PEER_FAILED = RunnerFailed(error_message="Terminated (exitcode=1)", diagnostics=[])


class _Process:
    """A runner process that does nothing by itself; the test plays the runner."""

    def __init__(self, alive: bool):
        self.alive = alive
        self.exitcode = None if alive else 1
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()

    def is_alive(self) -> bool:
        return self.alive

    async def run(self, *, task_status: StartedStatus[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        self.alive = False


@dataclass(frozen=True)
class _Runner:
    """What a test can do in the runner's place: read what the supervisor sends it, and send
    it events as the runner does."""

    supervisor: RunnerSupervisor
    tasks: MpReceiver[Task]
    events: MpSender[Event | RunnerTerminationError]


@asynccontextmanager
async def _cluster(
    process: _Process, local: RunnerStatus, peer: RunnerStatus
) -> AsyncIterator[tuple[Worker, _Runner, Receiver[Event]]]:
    """This node's worker and its runner of the instance, running, and what the node sends
    the master."""
    _, index_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )

    bound = BoundInstance(
        instance=INSTANCE, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    events_sender, events_receiver = mp_channel[Event | RunnerTerminationError]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    runner_process = cast(AsyncProcess, cast(object, process))
    supervisor = RunnerSupervisor(
        shard_metadata=bound.bound_shard,
        bound_instance=bound,
        runner_process=runner_process,
        _runner_stdio_handler=await RunnerStdioHandler.create(
            stdout_rx=process.stdout, stderr_rx=process.stderr
        ),
        _ev_recv=events_receiver,
        _task_sender=task_sender,
        _event_sender=event_sender.clone(),
        _cancel_sender=cancel_sender,
    )
    supervisor.status = local

    worker.state = State(
        instances={INSTANCE_1_ID: INSTANCE},
        runners={RUNNER_1_ID: local, RUNNER_2_ID: peer},
        downloads={NODE_A: DOWNLOADED},
    )
    worker.runners = {RUNNER_1_ID: supervisor}
    # Once its runner is shut down, the worker would start a new one: not in these tests
    no_restarts = KeyedBackoff[InstanceId](base=3600.0, cap=3600.0)
    no_restarts.record_attempt(INSTANCE_1_ID)
    worker._instance_backoff = no_restarts  # pyright: ignore[reportPrivateUsage]

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)
        tg.start_soon(worker.plan_step)
        yield worker, _Runner(supervisor, task_receiver, events_sender), event_receiver
        tg.cancel_scope.cancel()


async def _after_shutdown(worker: Worker, sent: Receiver[Event]) -> list[Event]:
    """Everything the node has sent once the worker has shut its runner down."""
    with anyio.fail_after(5):
        while RUNNER_1_ID in worker.runners:
            await anyio.sleep(0.05)
    # Long enough for the worker to be done with the shutdown
    await anyio.sleep(worker_mod.RUNNER_SHUTDOWN_GRACE + 0.5)
    return sent.collect()


def _shutdown_tasks_left(sent: list[Event]) -> list[Task]:
    """The Shutdown tasks in the cluster state once the master has applied what was sent"""
    created = [e for e in sent if isinstance(e, TaskCreated)]
    assert [type(e.task) for e in created] == [Shutdown]
    state = State()
    for idx, event in enumerate(sent):
        state = apply(state, IndexedEvent(idx=idx, event=event))
    return [task for task in state.tasks.values() if isinstance(task, Shutdown)]


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod, "RUNNER_SHUTDOWN_GRACE", 0.3)


async def test_shutting_down_a_crashed_runner_leaves_no_task_behind() -> None:
    # A model being served
    async with _cluster(_Process(alive=False), RunnerReady(), RunnerReady()) as (
        worker,
        runner,
        sent,
    ):
        # The runner dies mid-generation (e.g. a GPU timeout)
        runner.events.send(
            RunnerTerminationError.from_exception(
                RuntimeError("[METAL] Command buffer execution failed")
            )
        )
        events = await _after_shutdown(worker, sent)

    assert any(
        isinstance(e, RunnerStatusUpdated) and isinstance(e.runner_status, RunnerFailed)
        for e in events
    )
    # The task could never reach the dead runner, so it used to stay Pending
    assert _shutdown_tasks_left(events) == []


async def test_shutting_down_a_runner_that_takes_it_up_leaves_no_task_behind() -> None:
    # The instance's runner on the other node failed, so this node's runner is shut down
    async with _cluster(_Process(alive=True), RunnerIdle(), PEER_FAILED) as (
        worker,
        runner,
        sent,
    ):
        with anyio.fail_after(5):
            shutdown = await runner.tasks.receive_async()
        assert isinstance(shutdown, Shutdown)
        # As the runner does: it reports the task running, then acknowledges it. By the time
        # it would report it complete, its supervisor has stopped listening to it.
        runner.events.send(
            TaskStatusUpdated(task_id=shutdown.task_id, task_status=TaskStatus.Running)
        )
        runner.events.send(
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerShuttingDown()
            )
        )
        runner.events.send(TaskAcknowledged(task_id=shutdown.task_id))
        events = await _after_shutdown(worker, sent)

    # It used to stay Running
    assert _shutdown_tasks_left(events) == []


async def test_shutting_down_a_runner_that_never_takes_it_up_leaves_no_task_behind() -> (
    None
):
    async with _cluster(_Process(alive=True), RunnerIdle(), PEER_FAILED) as (
        worker,
        runner,
        sent,
    ):
        # The runner never reads its tasks, and is stopped once the grace period is over
        events = await _after_shutdown(worker, sent)

    # It used to stay TimedOut
    assert _shutdown_tasks_left(events) == []
    assert not runner.supervisor.runner_process.is_alive()
