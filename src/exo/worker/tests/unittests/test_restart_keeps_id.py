"""A node keeps its id across restarts. If it comes back before the master misses it, the cluster
still has the instances it ran, with the status of runners that died with the restart and the
requests they were serving. The restarted node has them deleted, as if it had been missed,
rather than starting a runner in the old one's place: the instance's other runners would keep
waiting for the old one, and its requests would be run a second time."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
import pytest

from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshot,
    TaskCreated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import CreateRunner
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerStatus,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, MiscData
from exo.worker.main import Worker
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

ONE_NODE = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID},
    runner_to_shard={RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 1)},
)
TWO_NODES = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 2),
        RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, 2),
    },
)


@dataclass(frozen=True)
class _StartedRunner(FakeRunnerSupervisor):
    async def run(self) -> None:
        await anyio.sleep_forever()


@dataclass
class Running:
    worker: Worker
    events: Sender[IndexedEvent | StateSnapshot]
    sent: Receiver[Event]
    commands: Receiver[ForwarderCommand]
    started: list[RunnerId]

    def deletions(self) -> list[InstanceId]:
        return [
            c.command.instance_id
            for c in self.commands.collect()
            if isinstance(c.command, DeleteInstance)
        ]


@asynccontextmanager
async def restarted_node(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
    *loops: Callable[[Worker], Callable[[], object]],
) -> AsyncIterator[Running]:
    """A freshly started worker for NODE_A, whose cluster state is `state`."""
    started: list[RunnerId] = []

    async def create(
        bound_instance: BoundInstance, event_sender: Sender[Event]
    ) -> _StartedRunner:
        del event_sender
        started.append(bound_instance.bound_runner_id)
        return _StartedRunner(bound_instance=bound_instance, status=RunnerIdle())

    monkeypatch.setattr("exo.worker.main.RunnerSupervisor.create", create)
    event_send, event_recv = channel[Event]()
    index_send, index_recv = channel[IndexedEvent | StateSnapshot]()
    command_send, commands = channel[ForwarderCommand]()
    download_send, _downloads = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )
    worker.state = state
    async with worker._tg as tg:  # pyright: ignore[reportPrivateUsage]
        for loop in loops:
            tg.start_soon(loop(worker))  # pyright: ignore[reportArgumentType]
        yield Running(worker, index_send, event_recv, commands, started)
        tg.cancel_scope.cancel()


def planning(worker: Worker) -> Callable[[], object]:
    return worker.plan_step


def _state(instance: Instance, runners: dict[RunnerId, RunnerStatus]) -> State:
    return State(
        instances={instance.instance_id: instance}, runners=runners, downloads={}
    )


def _runners_created(sent: Receiver[Event]) -> int:
    return sum(
        1
        for e in sent.collect()
        if isinstance(e, TaskCreated) and isinstance(e.task, CreateRunner)
    )


async def test_an_instance_from_before_a_restart_is_deleted_not_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The node ran this model and restarted; the master didn't miss it
    state = _state(ONE_NODE, {RUNNER_1_ID: RunnerReady()})
    async with restarted_node(state, monkeypatch, planning) as node:
        await anyio.sleep(0.5)
        assert node.deletions() == [INSTANCE_1_ID]
        assert node.started == []
        assert _runners_created(node.sent) == 0


async def test_a_shared_instance_from_before_a_restart_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other node's runner never noticed: it is idle, waiting for requests
    state = _state(TWO_NODES, {RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()})
    async with restarted_node(state, monkeypatch, planning) as node:
        await anyio.sleep(0.5)
        assert node.deletions() == [INSTANCE_1_ID]
        assert node.started == []


async def test_the_deletion_is_asked_for_again_if_it_does_not_come(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("exo.worker.main.DELETION_REQUEST_RETRY", 0.3)
    state = _state(ONE_NODE, {RUNNER_1_ID: RunnerReady()})
    async with restarted_node(state, monkeypatch, planning) as node:
        await anyio.sleep(0.5)
        assert node.deletions() == [INSTANCE_1_ID, INSTANCE_1_ID]


async def test_an_instance_placed_while_the_node_restarted_is_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Placed on this node before its runner ever reported: no runner of a previous run
    state = _state(ONE_NODE, {})
    async with restarted_node(state, monkeypatch, planning) as node:
        await anyio.sleep(0.5)
        assert node.started == [RUNNER_1_ID]
        assert node.deletions() == []


async def test_runners_this_run_started_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(ONE_NODE, {})
    async with restarted_node(state, monkeypatch, planning) as node:
        await anyio.sleep(0.3)
        assert node.started == [RUNNER_1_ID]
        # Its runner reports, so the cluster state has it
        node.worker.state = _state(ONE_NODE, {RUNNER_1_ID: RunnerIdle()})
        await anyio.sleep(0.5)
        assert node.deletions() == []


async def test_a_runner_this_node_holds_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(ONE_NODE, {RUNNER_1_ID: RunnerReady()})

    def holding_its_runner(worker: Worker) -> Callable[[], object]:
        worker.runners[RUNNER_1_ID] = _StartedRunner(  # pyright: ignore[reportArgumentType]
            bound_instance=BoundInstance(
                instance=ONE_NODE, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
            ),
            status=RunnerReady(),
        )
        return worker.plan_step

    async with restarted_node(state, monkeypatch, holding_its_runner) as node:
        await anyio.sleep(0.5)
        assert node.deletions() == []
        assert node.started == []


async def test_a_node_the_cluster_removed_while_up_announces_itself_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info_send, info_recv = channel[GatheredInfo]()
    name = MiscData(friendly_name="studio")

    def forwarding(worker: Worker) -> Callable[[], object]:
        return lambda: worker._forward_info(info_recv)  # pyright: ignore[reportPrivateUsage]

    def applying(worker: Worker) -> Callable[[], object]:
        return worker._event_applier  # pyright: ignore[reportPrivateUsage]

    async with restarted_node(State(), monkeypatch, forwarding, applying) as node:
        # Gathered once, at start
        await info_send.send(name)
        with anyio.fail_after(5):
            await node.sent.receive()

        # The master removes another node: nothing to do
        await node.events.send(
            IndexedEvent(idx=0, event=NodeTimedOut(node_id=NodeId("other")))
        )
        await anyio.sleep(0.2)
        assert node.sent.collect() == []

        # It removes this one, which has just restarted (or was frozen) and is up
        await node.events.send(IndexedEvent(idx=1, event=NodeTimedOut(node_id=NODE_A)))
        with anyio.fail_after(5):
            announced = await node.sent.receive()
        assert isinstance(announced, NodeGatheredInfo)
        assert announced.info == name
