"""A runner is replaced only once it is gone, and never pairs with one that is going.

A runner keeps its id when it is replaced, and its instance keeps its ports. The worker used to
start a replacement as soon as it had told the old runner to stop, while the old process was
still alive, and the old runner's last status stayed in the state: the instance's other runners
took it for a live one and connected to a process that was about to be stopped (MLX's ring
pairs with any process at the address). The instance then sat for good with one runner
Connected to a dead peer and the other Idle: the plan connects only runners that are all Idle
or Connecting, and loads only runners that are all Connected.

Now:
- a runner on its way out is ShuttingDown (or Failed) to the cluster until its replacement
  reports, whatever its old process still says;
- its replacement starts only once its process has exited and this node's last report on it is
  back in the node's copy of the state (until then that copy holds the old process's statuses,
  which would be read as the new one's);
- a runner that has joined others is stopped when one of them goes, judged on every report the
  worker applies, not only on the state the plan happens to see;
- a replacement waits until none of the instance's other runners has joined anything;
- a runner whose process can't be stopped is reported failed, and never replaced.

The worker-level tests run unchanged against the worker before this (they fail there on what it
does, not on a missing name).
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

import anyio
import pytest
from anyio.abc import TaskStatus as StartedStatus

import exo.worker.main as worker_mod
import exo.worker.plan as plan_mod
import exo.worker.runner.supervisor as supervisor_mod
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    NodeDownloadProgress,
    RunnerStatusUpdated,
    StateSnapshot,
    TaskAcknowledged,
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
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Sender, channel, mp_channel
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
DOWNLOADED = DownloadCompleted(
    node_id=NODE_A,
    shard_metadata=INSTANCE.shard_assignments.runner_to_shard[RUNNER_1_ID],
    total=Memory(),
)
FAILED = RunnerFailed(error_message="Terminated (exitcode=1)", diagnostics=[])


def _plan(
    peer: RunnerStatus | None,
    local: FakeRunnerSupervisor | None = None,
    last_status: RunnerStatus | None = None,
    exits_in_flight: frozenset[RunnerId] = frozenset(),
) -> Task | None:
    """What this node's worker does next, with `local` its runner of the instance (None: it
    has none), `peer` the other runner's status in the state, and `last_status` this node's
    runner's status in the state (by default, `local`'s)."""
    all_runners: dict[RunnerId, RunnerStatus] = {}
    if peer is not None:
        all_runners[RUNNER_2_ID] = peer
    if last_status is not None:
        all_runners[RUNNER_1_ID] = last_status
    elif local is not None:
        all_runners[RUNNER_1_ID] = local.status
    runners = {} if local is None else {RUNNER_1_ID: local}
    downloads = {NODE_A: [DOWNLOADED]}
    instances = {INSTANCE_1_ID: INSTANCE}
    if exits_in_flight:
        return plan_mod.plan(
            NODE_A,
            runners,  # type: ignore[arg-type]
            downloads,
            instances,
            all_runners,
            {},
            KeyedBackoff(),
            KeyedBackoff(),
            exits_in_flight=exits_in_flight,
        )
    # Without the argument, so that these tests also run against plans that don't take it
    return plan_mod.plan(
        NODE_A,
        runners,  # type: ignore[arg-type]
        downloads,
        instances,
        all_runners,
        {},
        KeyedBackoff(),
        KeyedBackoff(),
    )


def _runner(
    status: RunnerStatus, stopping: bool = False, peer_went: bool = False
) -> FakeRunnerSupervisor:
    return FakeRunnerSupervisor(
        bound_instance=BOUND, status=status, stopping=stopping, peer_went=peer_went
    )


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


@pytest.mark.parametrize("last_status", [RunnerIdle(), RunnerConnecting()], ids=_name)
def test_a_replacement_waits_for_this_nodes_last_report_on_the_runner_it_replaces(
    last_status: RunnerStatus,
) -> None:
    """The old runner has exited, but this node's copy of the state still has what it last
    said (its report that it is going is on its way back from the master). A replacement now
    would be told to connect on the old process's Idle, and the other nodes would read it too."""
    in_flight = frozenset({RUNNER_1_ID})
    assert (
        _plan(RunnerIdle(), last_status=last_status, exits_in_flight=in_flight) is None
    )
    # Once the report is back: the replacement starts, and isn't told to connect before its
    # own Idle is in the state
    assert isinstance(_plan(RunnerIdle(), last_status=FAILED), CreateRunner)
    assert _plan(RunnerIdle(), local=_runner(RunnerIdle()), last_status=FAILED) is None


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


def test_a_runner_whose_peer_was_seen_going_is_stopped_whatever_the_state_says() -> (
    None
):
    """The worker saw the peer's report that it was going; by the time the plan looks, the
    state already has its replacement's Idle. This runner may be paired with the process that
    went, and nothing else would ever stop it: it waits to load, the replacement to connect."""
    assert isinstance(
        _plan(RunnerIdle(), local=_runner(RunnerConnected(), peer_went=True)), Shutdown
    )


@pytest.mark.parametrize("status", [FAILED, RunnerShuttingDown()], ids=_name)
def test_a_runner_on_its_way_out_is_left_alone_until_it_is_gone(
    status: RunnerStatus,
) -> None:
    """Neither stopped again, nor sent anything, nor replaced while it is still there."""
    assert _plan(RunnerIdle(), local=_runner(status, stopping=True)) is None
    assert _plan(FAILED, local=_runner(status, stopping=True)) is None


# --- the worker -------------------------------------------------------------------------


class _Process:
    """A runner process that ends only when the test says (as one stuck in MLX takes seconds to
    be terminated), with the exit code the test gives it. Like AsyncProcess, it is stopped by
    cancelling run(), which returns only once the process has exited."""

    def __init__(self) -> None:
        self.exit = anyio.Event()
        self.exitcode: int | None = None
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()

    def is_alive(self) -> bool:
        return not self.exit.is_set()

    def end(self, exitcode: int) -> None:
        self.exitcode = exitcode
        self.exit.set()

    async def run(self, *, task_status: StartedStatus[None]) -> None:
        task_status.started()
        try:
            await anyio.sleep_forever()
        finally:
            # Terminated, then killed: a process that the kernel holds never gets past this
            with anyio.CancelScope(shield=True):
                await self.exit.wait()

    async def stop(self) -> None:
        await self.exit.wait()


@dataclass(frozen=True)
class _Runner:
    """A runner the worker started, played by the test: what its supervisor sends it, and what
    it says back."""

    supervisor: RunnerSupervisor
    process: _Process
    tasks: MpReceiver[Task]
    events: MpSender[Event | RunnerTerminationError]

    async def told(self, kind: type[Task]) -> Task:
        with anyio.fail_after(10):
            while True:
                task = await self.tasks.receive_async()
                if isinstance(task, kind):
                    return task

    def say(self, *events: Event | RunnerTerminationError) -> None:
        for event in events:
            self.events.send(event)

    async def joins(self) -> None:
        """Start, be told to connect, and start connecting"""
        self.say(RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerIdle()))
        connect = await self.told(ConnectToGroup)
        self.say(
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerConnecting()
            ),
            TaskAcknowledged(task_id=connect.task_id),
        )


@dataclass
class _Node:
    """This node's worker, with a master that indexes what the node sends, and the other node's
    reports, in one order, and hands them back; or holds them back (a slow round trip)."""

    worker: Worker
    to_worker: Sender[IndexedEvent]
    runners: list[_Runner] = field(default_factory=list)
    sent: list[Event] = field(default_factory=list)
    held: list[Event] | None = None
    _index: int = 0

    async def index(self, event: Event) -> None:
        if self.held is not None:
            self.held.append(event)
            return
        await self.to_worker.send(IndexedEvent(idx=self._index, event=event))
        self._index += 1

    async def peer_is(self, status: RunnerStatus) -> None:
        """The other node reports its runner"""
        await self.index(
            RunnerStatusUpdated(runner_id=RUNNER_2_ID, runner_status=status)
        )

    def hold(self) -> None:
        self.held = []

    async def release(self) -> None:
        held, self.held = self.held or [], None
        for event in held:
            await self.index(event)

    async def started(self, count: int) -> _Runner:
        with anyio.fail_after(10):
            while len(self.runners) < count:
                await anyio.sleep(0.02)
        return self.runners[count - 1]

    async def until(self, condition: Callable[[], bool], timeout: float = 10) -> None:
        with anyio.fail_after(timeout):
            while not condition():
                await anyio.sleep(0.02)

    def statuses(self) -> list[type[RunnerStatus]]:
        """What this node said about its runner, in order"""
        return [
            type(e.runner_status)
            for e in self.sent
            if isinstance(e, RunnerStatusUpdated) and e.runner_id == RUNNER_1_ID
        ]


@asynccontextmanager
async def _node(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Node]:
    """This node's worker, holding its half of INSTANCE (the other runner is Idle); the runners
    it starts are fake processes the test plays."""
    to_worker, from_master = channel[IndexedEvent]()
    event_sender, sent = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=from_master,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )
    node = _Node(worker, to_worker)

    async def create(
        *, bound_instance: BoundInstance, event_sender: Sender[Event]
    ) -> RunnerSupervisor:
        process = _Process()
        events_sender, events_receiver = mp_channel[Event | RunnerTerminationError]()
        task_sender, task_receiver = mp_channel[Task]()
        cancel_sender, _ = mp_channel[TaskId]()
        supervisor = RunnerSupervisor(
            shard_metadata=bound_instance.bound_shard,
            bound_instance=bound_instance,
            runner_process=cast(AsyncProcess, cast(object, process)),
            _runner_stdio_handler=await RunnerStdioHandler.create(
                stdout_rx=process.stdout, stderr_rx=process.stderr
            ),
            _ev_recv=events_receiver,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _cancel_sender=cancel_sender,
        )
        node.runners.append(_Runner(supervisor, process, task_receiver, events_sender))
        return supervisor

    monkeypatch.setattr(RunnerSupervisor, "create", staticmethod(create))

    async def master() -> None:
        with sent as events:
            async for event in events:
                node.sent.append(event)
                await node.index(event)

    await node.index(InstanceCreated(instance=INSTANCE))
    await node.index(NodeDownloadProgress(download_progress=DOWNLOADED))
    await node.peer_is(RunnerIdle())
    tg = worker._tg  # pyright: ignore[reportPrivateUsage]
    async with tg:
        tg.start_soon(master)
        tg.start_soon(worker._event_applier)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(worker.plan_step)
        try:
            yield node
        finally:
            for runner in node.runners:
                runner.process.end(-9)
            tg.cancel_tasks()


@pytest.fixture(autouse=True)
def short_timers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod, "RUNNER_SHUTDOWN_GRACE", 0.3)
    monkeypatch.setattr(supervisor_mod, "RUNNER_WATCH_INTERVAL", 0.05)


async def test_a_runner_is_replaced_only_once_its_process_has_exited_and_its_last_report_is_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        # The other node's runner fails: this node's is stopped. It never takes up its
        # Shutdown (stuck in MLX's ring set-up), and its process takes a while to go.
        await node.peer_is(FAILED)
        await old.told(Shutdown)
        # The other node starts its replacement
        await node.peer_is(RunnerIdle())
        await anyio.sleep(worker_mod.RUNNER_SHUTDOWN_GRACE + 1.0)
        assert len(node.runners) == 1, "replaced while the old process is still alive"

        # The process exits; this node's last report on it is slow to come back
        node.hold()
        old.process.end(-15)
        await anyio.sleep(1.0)
        assert len(node.runners) == 1, (
            "replaced before this node's last report was back"
        )

        await node.release()
        await node.started(2)


async def test_a_runner_on_its_way_out_is_shutting_down_whatever_it_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From the moment it is told to stop, the cluster sees it ShuttingDown until its
    replacement reports. What the old process still says (here: that it connected, to a
    process about to go) is not passed on, and the step it never finished is failed rather
    than left Running in the state."""
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        await node.peer_is(FAILED)
        shutdown = await old.told(Shutdown)
        old.say(
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected()),
            TaskAcknowledged(task_id=shutdown.task_id),
        )
        old.process.end(0)
        await node.until(
            lambda: isinstance(
                node.worker.state.runners.get(RUNNER_1_ID), RunnerShuttingDown
            )
            and old.supervisor.runner_process.exitcode is not None
        )
        await anyio.sleep(0.5)
        assert RunnerConnected not in node.statuses()
        assert isinstance(
            node.worker.state.runners.get(RUNNER_1_ID), RunnerShuttingDown
        )
        assert [
            task.task_status
            for task in node.worker.state.tasks.values()
            if isinstance(task, ConnectToGroup)
        ] == [TaskStatus.Failed]


async def test_a_runner_stopped_on_purpose_is_not_reported_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner stopped because its peer failed, which had to be terminated, didn't fail: it
    used to be reported Failed ("Terminated (signal=15)"), which the dashboard showed as the
    model's failure."""
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        await node.peer_is(FAILED)
        await old.told(Shutdown)
        old.process.end(-15)
        await anyio.sleep(worker_mod.RUNNER_SHUTDOWN_GRACE + 0.5)
        assert RunnerFailed not in node.statuses()


async def test_a_runner_that_crashes_as_it_is_stopped_is_reported_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        await node.peer_is(FAILED)
        await old.told(Shutdown)
        old.process.end(1)
        await node.until(lambda: RunnerFailed in node.statuses())


async def test_a_gone_runner_is_announced_to_a_new_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner that failed and is gone waits to be replaced while the other node's runner is
    still joined to it. A new master may never have had its Failed (sent to the old one): told
    again, the other node stops its runner, and the replacement can start."""
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        await node.peer_is(RunnerConnecting())
        old.process.end(1)
        old.say(RunnerTerminationError.from_exception(RuntimeError("Metal timeout")))
        await node.until(lambda: RunnerFailed in node.statuses())
        await anyio.sleep(0.5)
        before = len(node.sent)
        await node.worker.announce_to_master()
        await anyio.sleep(0.2)
        assert [
            type(e.runner_status)
            for e in node.sent[before:]
            if isinstance(e, RunnerStatusUpdated) and e.runner_id == RUNNER_1_ID
        ] == [RunnerFailed]


async def test_a_process_that_cant_be_stopped_is_reported_and_never_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process the kernel holds (a wedged GPU driver) survives SIGKILL. Its runner is never
    replaced (the replacement would meet it, and the GPU it holds), and the user is told what
    to do instead of seeing the model stopping for good."""
    monkeypatch.setattr(supervisor_mod, "RUNNER_STOP_TIMEOUT", 0.5, raising=False)
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        await node.peer_is(FAILED)
        await old.told(Shutdown)
        await node.peer_is(RunnerIdle())
        await node.until(lambda: RunnerFailed in node.statuses())
        failure = node.worker.state.runners.get(RUNNER_1_ID)
        assert isinstance(failure, RunnerFailed)
        assert "Restart this device" in (failure.error_message or "")
        await anyio.sleep(1.0)
        assert len(node.runners) == 1, "replaced next to a process that is still there"


async def test_a_peer_seen_going_stops_a_joined_runner_even_if_replaced_since() -> None:
    """The other node's runner goes and its replacement reports Idle, both before this node's
    plan looks: the state it sees has a fresh Idle peer, and this node's Connected runner is
    paired with the process that went. The worker notes the going report as it applies it."""
    to_worker, from_master = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=from_master,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )
    process = _Process()
    supervisor = RunnerSupervisor(
        shard_metadata=BOUND.bound_shard,
        bound_instance=BOUND,
        runner_process=cast(AsyncProcess, cast(object, process)),
        _runner_stdio_handler=await RunnerStdioHandler.create(
            stdout_rx=process.stdout, stderr_rx=process.stderr
        ),
        _ev_recv=mp_channel[Event | RunnerTerminationError]()[1],
        _task_sender=mp_channel[Task]()[0],
        _event_sender=event_sender.clone(),
        _cancel_sender=mp_channel[TaskId]()[0],
    )
    supervisor.status = RunnerConnected()
    worker.runners = {RUNNER_1_ID: supervisor}
    worker.state = State(
        instances={INSTANCE_1_ID: INSTANCE},
        runners={RUNNER_1_ID: RunnerConnected(), RUNNER_2_ID: RunnerConnected()},
        downloads={NODE_A: [DOWNLOADED]},
    )
    for idx, status in enumerate((RunnerShuttingDown(), RunnerIdle())):
        await to_worker.send(
            IndexedEvent(
                idx=idx,
                event=RunnerStatusUpdated(runner_id=RUNNER_2_ID, runner_status=status),
            )
        )
    to_worker.close()
    await worker._event_applier()  # pyright: ignore[reportPrivateUsage]

    task = plan_mod.plan(
        NODE_A,
        worker.runners,
        worker.state.downloads,
        worker.state.instances,
        worker.state.runners,
        worker.state.tasks,
        KeyedBackoff(),
        KeyedBackoff(),
    )
    assert isinstance(task, Shutdown)


async def test_replacements_that_are_slow_to_start_are_left_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both runners of the instance crash mid-reply, and both nodes start replacements, which
    take a while to start (a loaded machine). A replacement that hasn't reported yet is tied to
    no one: it used to be stopped at once because the other node's runner still showed Failed,
    and stopping it showed it Failed in turn, so the two nodes stopped each other's fresh
    runners round after round until both happened to start in time."""
    async with _node(monkeypatch) as node:
        old = await node.started(1)
        await old.joins()
        old.process.end(1)
        old.say(RunnerTerminationError.from_exception(RuntimeError("Metal timeout")))
        await node.peer_is(FAILED)
        new = await node.started(2)
        # Neither replacement reports for a while
        await anyio.sleep(1.5)
        assert len(node.runners) == 2, "the fresh runner was stopped and replaced again"
        assert new.tasks.collect() == [], "the fresh runner was sent something"
        assert not new.supervisor.stopping
        # The other node's replacement starts, then this one: they connect
        await node.peer_is(RunnerIdle())
        new.say(RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerIdle()))
        assert isinstance(await new.told(ConnectToGroup), ConnectToGroup)


class _SlowSender:
    """The worker's event sender, recording what it sends. Each send yields, as the real one
    does; `during_first_send` runs at the first: whatever plan_step does between two of
    announce_to_master's sends."""

    def __init__(self, during_first_send: Callable[[], None] | None) -> None:
        self.events: list[Event] = []
        self._hook = during_first_send

    async def send(self, event: Event) -> None:
        self.events.append(event)
        if self._hook is not None:
            hook, self._hook = self._hook, None
            hook()
        await anyio.sleep(0)


def _bare_worker() -> Worker:
    _, from_master = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    return Worker(
        NODE_A,
        event_receiver=from_master,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )


async def test_a_runner_replaced_during_an_announcement_is_not_reported_going() -> None:
    """Two of this node's runners are gone (two instances restarting) when the master changes,
    and the plan replaces the second while the first is being announced. Its replacement is
    live: reported going, it would stop its peers at this master change and every later one.
    (Found in review.)"""
    worker = _bare_worker()
    worker._gone = {RUNNER_1_ID: FAILED, RUNNER_2_ID: FAILED}  # pyright: ignore[reportPrivateUsage]
    replacement = cast(
        RunnerSupervisor,
        cast(object, FakeRunnerSupervisor(bound_instance=BOUND, status=RunnerReady())),
    )

    def plan_replaces_runner_2() -> None:
        worker._gone.pop(RUNNER_2_ID, None)  # pyright: ignore[reportPrivateUsage]
        worker.runners[RUNNER_2_ID] = replacement

    sender = _SlowSender(plan_replaces_runner_2)
    worker.event_sender = cast(Sender[Event], cast(object, sender))
    await worker.announce_to_master()

    going_reports = [
        e.runner_id
        for e in sender.events
        if isinstance(e, RunnerStatusUpdated)
        and isinstance(e.runner_status, (RunnerFailed, RunnerShuttingDown))
    ]
    assert going_reports == [RUNNER_1_ID]
    assert RUNNER_2_ID not in worker._gone  # pyright: ignore[reportPrivateUsage]


async def test_gone_runners_of_an_instance_deleted_within_a_snapshot_are_forgotten() -> (
    None
):
    """An instance deleted while this node waited for a new master's state: its gone runners
    are never replaced, and aren't announced to later masters."""
    to_worker, from_master = channel[IndexedEvent | StateSnapshot]()
    worker = _bare_worker()
    worker.event_receiver = from_master
    worker._gone = {RUNNER_1_ID: FAILED}  # pyright: ignore[reportPrivateUsage]
    session = SessionId(master_node_id=NODE_B, election_clock=1)
    await to_worker.send(
        StateSnapshot(state=State(), session=session, requester=SystemId())
    )
    to_worker.close()
    await worker._event_applier()  # pyright: ignore[reportPrivateUsage]
    assert worker._gone == {}  # pyright: ignore[reportPrivateUsage]
