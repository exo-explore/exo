"""A runner is replaced only once it is gone, and never pairs with one that is going.

A runner keeps its id when it is replaced, and its instance keeps its ports. The worker used to
start a replacement as soon as it had told the old runner to stop, while the old process was
still alive: the replacement couldn't bind the instance's port (EADDRINUSE), or, on the other
nodes, the old runner's last status stayed in the state, so the instance's other runners took it
for a live one and connected to a process that was about to be stopped. The instance then sat
for good with one runner Connected to a dead peer and the other Idle: the plan connects only
runners that are all Idle or Connecting, and loads only runners that are all Connected.

Now a runner on its way out is ShuttingDown (or Failed) to the cluster until its replacement
reports, its replacement starts only once its process has exited, a runner that has joined
others is stopped when one of them goes, and a replacement waits until none of the runners it
replaces could still be tied to its old process.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import anyio
import pytest
from anyio.abc import TaskStatus as StartedStatus

import exo.worker.main as worker_mod
import exo.worker.plan as plan_mod
import exo.worker.runner.supervisor as supervisor_mod
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ConnectToGroup,
    CreateRunner,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
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
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

# A model split across this node (RUNNER_1_ID, rank 0) and another (RUNNER_2_ID, rank 1)
INSTANCE = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 2),
        RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, 2),
    },
)
BOUND = BoundInstance(
    instance=INSTANCE, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
)
DOWNLOADED = [
    DownloadCompleted(
        node_id=NODE_A,
        shard_metadata=INSTANCE.shard_assignments.runner_to_shard[RUNNER_1_ID],
        total=Memory(),
    )
]
FAILED = RunnerFailed(error_message="Terminated (exitcode=1)", diagnostics=[])


def _plan(
    peer: RunnerStatus | None,
    local: FakeRunnerSupervisor | None = None,
    last_status: RunnerStatus | None = None,
) -> Task | None:
    """What this node's worker does next, with `local` its runner of the instance (None: it
    has none), `peer` the other runner's status in the state and `last_status` this node's."""
    all_runners: dict[plan_mod.RunnerId, RunnerStatus] = {}
    if peer is not None:
        all_runners[RUNNER_2_ID] = peer
    if local is not None:
        all_runners[RUNNER_1_ID] = local.status
    elif last_status is not None:
        all_runners[RUNNER_1_ID] = last_status
    return plan_mod.plan(
        node_id=NODE_A,
        runners={} if local is None else {RUNNER_1_ID: local},  # type: ignore[dict-item]
        global_download_status={NODE_A: DOWNLOADED},
        instances={INSTANCE_1_ID: INSTANCE},
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )


def _runner(status: RunnerStatus, stopping: bool = False) -> FakeRunnerSupervisor:
    return FakeRunnerSupervisor(bound_instance=BOUND, status=status, stopping=stopping)


def _name(status: RunnerStatus | None) -> str:
    return "absent" if status is None else type(status).__name__


# --- the plan ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "peer", [RunnerConnecting(), RunnerConnected(), RunnerReady()], ids=_name
)
def test_a_replacement_waits_while_a_peer_is_tied_to_the_runner_it_replaces(
    peer: RunnerStatus,
) -> None:
    """This node's runner failed and is gone; the other node's runner had joined it and hasn't
    been stopped yet. A replacement now would pair with that process: on real Macs, rank 1's
    replacement finished MLX's ring set-up with rank 0's old runner, still Connecting."""
    assert _plan(peer, last_status=FAILED) is None


@pytest.mark.parametrize(
    "peer", [None, RunnerIdle(), RunnerShuttingDown(), FAILED], ids=_name
)
def test_a_replacement_starts_once_no_peer_is_tied_to_the_runner_it_replaces(
    peer: RunnerStatus | None,
) -> None:
    assert isinstance(_plan(peer, last_status=FAILED), CreateRunner)


def test_an_idle_runner_waits_for_a_peer_that_is_going() -> None:
    """An idle runner is tied to no other runner: stopping it when a peer fails only throws
    away a runner that has just started (a real two-node cluster started each replacement
    twice, the first one killed at once)."""
    assert _plan(FAILED, local=_runner(RunnerIdle())) is None
    assert _plan(RunnerShuttingDown(), local=_runner(RunnerIdle())) is None


@pytest.mark.parametrize(
    "status", [RunnerConnecting(), RunnerConnected(), RunnerReady()], ids=_name
)
@pytest.mark.parametrize("peer", [FAILED, RunnerShuttingDown()], ids=_name)
def test_a_runner_that_joined_a_peer_that_is_going_is_stopped(
    status: RunnerStatus, peer: RunnerStatus
) -> None:
    assert isinstance(_plan(peer, local=_runner(status)), Shutdown)


@pytest.mark.parametrize("status", [FAILED, RunnerShuttingDown()], ids=_name)
def test_a_runner_on_its_way_out_is_left_alone_until_it_is_gone(
    status: RunnerStatus,
) -> None:
    """Neither stopped again, nor sent anything, nor replaced while it is still there."""
    assert _plan(RunnerIdle(), local=_runner(status, stopping=True)) is None
    assert _plan(FAILED, local=_runner(status, stopping=True)) is None


# --- the worker -------------------------------------------------------------------------


class _Process:
    """A runner process that ends only when the test says, as one stuck in MLX takes seconds
    to be terminated."""

    def __init__(self) -> None:
        self.exit = anyio.Event()
        self.exitcode: int | None = None
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()

    def is_alive(self) -> bool:
        return not self.exit.is_set()

    async def run(self, *, task_status: StartedStatus[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        await self.exit.wait()
        self.exitcode = -15


@dataclass(frozen=True)
class _Node:
    worker: Worker
    supervisor: RunnerSupervisor
    process: _Process
    # What the supervisor sends the runner, and what the runner can send back
    tasks: MpReceiver[Task]
    events: MpSender[Event | RunnerTerminationError]
    # What the node sends the master
    sent: Receiver[Event]
    created: list[CreateRunner]


@asynccontextmanager
async def _node(local: RunnerStatus, peer: RunnerStatus) -> AsyncIterator[_Node]:
    """This node's worker with its runner of the instance, which is `local`."""
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
    created: list[CreateRunner] = []

    async def create_supervisor(task: CreateRunner) -> None:
        # Record it instead of starting a runner process
        created.append(task)

    worker._create_supervisor = create_supervisor  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    process = _Process()
    events_sender, events_receiver = mp_channel[Event | RunnerTerminationError]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    supervisor = RunnerSupervisor(
        shard_metadata=BOUND.bound_shard,
        bound_instance=BOUND,
        runner_process=cast(AsyncProcess, cast(object, process)),
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

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._supervise, supervisor)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(worker.plan_step)
        yield _Node(
            worker,
            supervisor,
            process,
            task_receiver,
            events_sender,
            event_receiver,
            created,
        )
        tg.cancel_scope.cancel()


def _peer_restarted(worker: Worker) -> None:
    """What the master's state says once the other node's replacement has started"""
    worker.state = worker.state.model_copy(
        update={
            "runners": {RUNNER_1_ID: RunnerShuttingDown(), RUNNER_2_ID: RunnerIdle()}
        }
    )


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod, "RUNNER_SHUTDOWN_GRACE", 0.3)


async def test_a_stopped_runner_is_replaced_only_once_its_process_has_exited() -> None:
    # Serving, until the other node's runner fails
    async with _node(RunnerReady(), FAILED) as node:
        with anyio.fail_after(2):
            while not node.supervisor.stopping:
                await anyio.sleep(0.05)
        # The other node starts its replacement; this node's runner doesn't take up its
        # Shutdown, and its process takes a while to be terminated
        _peer_restarted(node.worker)
        await anyio.sleep(worker_mod.RUNNER_SHUTDOWN_GRACE + 1.0)
        assert node.created == [], (
            "replaced while the old process still holds its ports"
        )
        assert node.worker.runners[RUNNER_1_ID] is node.supervisor

        node.process.exit.set()
        with anyio.fail_after(2):
            while not node.created:
                await anyio.sleep(0.05)
        assert node.created[0].bound_instance.bound_runner_id == RUNNER_1_ID
        assert RUNNER_1_ID not in node.worker.runners


async def test_a_runner_on_its_way_out_is_shutting_down_whatever_it_says() -> None:
    """The cluster sees it ShuttingDown from the moment it is told to stop, until its
    replacement reports; what the old process still says (here: that it connected, to a
    process about to go) is not passed on. And the step it never finished is failed, not left
    Running in the state."""
    async with _node(RunnerConnecting(), FAILED) as node:
        connect = ConnectToGroup(instance_id=INSTANCE_1_ID)
        node.supervisor.in_progress[connect.task_id] = connect
        with anyio.fail_after(2):
            shutdown = await node.tasks.receive_async()
        assert isinstance(shutdown, Shutdown)
        node.events.send(
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected())
        )
        node.events.send(TaskAcknowledged(task_id=shutdown.task_id))
        node.process.exit.set()
        with anyio.fail_after(3):
            while RUNNER_1_ID in node.worker.runners:
                await anyio.sleep(0.05)
        sent = node.sent.collect()

    statuses = [
        type(e.runner_status) for e in sent if isinstance(e, RunnerStatusUpdated)
    ]
    assert statuses == [RunnerShuttingDown]
    assert [
        e.task_status
        for e in sent
        if isinstance(e, TaskStatusUpdated) and e.task_id == connect.task_id
    ] == [TaskStatus.Failed]


async def test_a_runner_stopped_on_purpose_is_not_reported_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner stopped because its peer failed, found dead (here terminated) while it is
    being stopped, didn't fail: it used to be reported Failed ("Terminated (signal=15)"),
    which the dashboard showed as the model's failure."""
    monkeypatch.setattr(supervisor_mod, "RUNNER_WATCH_INTERVAL", 0.05)
    async with _node(RunnerReady(), FAILED) as node:
        with anyio.fail_after(2):
            while not node.supervisor.stopping:
                await anyio.sleep(0.05)
        node.process.exitcode = -15
        node.process.exit.set()
        with anyio.fail_after(3):
            while RUNNER_1_ID in node.worker.runners:
                await anyio.sleep(0.05)
        sent = node.sent.collect()

    assert [
        type(e.runner_status) for e in sent if isinstance(e, RunnerStatusUpdated)
    ] == [RunnerShuttingDown]
