"""A worker keeps its runners when the master changes, and catches the new master up."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import anyio

from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InputChunkReceived,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
    TaskCreated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import DownloadModel
from exo.shared.types.worker.instances import BoundInstance, Instance
from exo.shared.types.worker.runners import RunnerIdle, RunnerReady
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MiscData,
    NodeBackends,
)
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

OLD = SessionId(master_node_id=NodeId("old-master"), election_clock=1)
NEW = SessionId(master_node_id=NodeId("new-master"), election_clock=2)
INSTANCE = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(
            MODEL_A_ID, device_rank=0, world_size=1
        )
    },
)


async def wait_until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not condition():
            await anyio.sleep(0.01)


def snapshot(session: SessionId, state: State) -> StateSnapshot:
    return StateSnapshot(session=session, requester=SystemId(), state=state)


def with_runner(
    worker: Worker, instance: Instance, status: RunnerIdle | RunnerReady
) -> None:
    worker.runners[RUNNER_1_ID] = FakeRunnerSupervisor(  # pyright: ignore[reportArgumentType]
        bound_instance=BoundInstance(
            instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
        ),
        status=status,
    )


@asynccontextmanager
async def running_worker(
    state: State, *loops: Callable[[Worker], Callable[[], object]]
) -> AsyncIterator[
    tuple[Worker, Sender[IndexedEvent | StateSnapshot], Receiver[Event]]
]:
    event_send, event_recv = channel[Event]()
    index_send, index_recv = channel[IndexedEvent | StateSnapshot]()
    command_send, _commands = channel[ForwarderCommand]()
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
    async with anyio.create_task_group() as tg:
        for loop in loops:
            tg.start_soon(loop(worker))  # pyright: ignore[reportArgumentType]
        yield worker, index_send, event_recv
        tg.cancel_scope.cancel()


def applying(worker: Worker) -> Callable[[], object]:
    return worker._event_applier  # pyright: ignore[reportPrivateUsage]


async def test_a_new_master_is_told_everything_known_about_the_node() -> None:
    info_send, info_recv = channel[GatheredInfo]()
    backends = NodeBackends(backends=[])

    def forwarding(worker: Worker) -> Callable[[], object]:
        with_runner(worker, INSTANCE, RunnerReady())
        return lambda: worker._forward_info(info_recv)  # pyright: ignore[reportPrivateUsage]

    async with running_worker(State(), forwarding) as (worker, _, sent):
        # Backends are only gathered once, when the node starts
        await info_send.send(backends)
        await info_send.send(MiscData(friendly_name="studio"))
        await info_send.send(MiscData(friendly_name="studio 2"))
        with anyio.fail_after(5):
            for _ in range(3):
                await sent.receive()

        worker.follow_new_master(NEW)
        await worker.announce_to_master()

        announced = sent.collect()
    infos = [e.info for e in announced if isinstance(e, NodeGatheredInfo)]
    assert infos == [backends, MiscData(friendly_name="studio 2")]
    statuses = [e for e in announced if isinstance(e, RunnerStatusUpdated)]
    assert [(s.runner_id, s.runner_status) for s in statuses] == [
        (RUNNER_1_ID, RunnerReady())
    ]


async def test_the_worker_waits_for_the_new_master_s_state_before_acting() -> None:
    # The runner's model isn't downloaded: once the worker acts on this state, it starts
    # downloading it
    state = State(instances={INSTANCE_1_ID: INSTANCE}, downloads={NODE_A: []})

    def planning(worker: Worker) -> Callable[[], object]:
        with_runner(worker, INSTANCE, RunnerIdle())
        return worker.plan_step

    def downloads_started(sent: Receiver[Event]) -> int:
        return sum(
            1
            for e in sent.collect()
            if isinstance(e, TaskCreated) and isinstance(e.task, DownloadModel)
        )

    async with running_worker(state, planning, applying) as (worker, events, sent):
        worker.follow_new_master(NEW)
        await anyio.sleep(0.5)
        assert downloads_started(sent) == 0

        # A snapshot from the session it left doesn't count
        await events.send(snapshot(OLD, state))
        await anyio.sleep(0.5)
        assert downloads_started(sent) == 0

        await events.send(snapshot(NEW, state))
        await anyio.sleep(0.5)
        assert downloads_started(sent) >= 1


async def test_images_of_requests_that_ended_with_the_old_master_are_dropped() -> None:
    first_of_two = InputChunkReceived(
        command_id=COMMAND_1_ID,
        chunk=InputImageChunk(
            model=MODEL_A_ID,
            command_id=COMMAND_1_ID,
            data="aGVsbG8=",
            chunk_index=0,
            total_chunks=2,
        ),
    )
    async with running_worker(State(), applying) as (worker, events, _):
        await events.send(IndexedEvent(idx=0, event=first_of_two))
        await wait_until(lambda: COMMAND_1_ID in worker.input_chunk_buffer)

        worker.follow_new_master(NEW)
        await events.send(snapshot(NEW, State()))
        await wait_until(lambda: COMMAND_1_ID not in worker.input_chunk_buffer)
    assert worker.input_chunk_counts == {}


async def test_a_worker_that_already_has_the_new_master_s_state_does_not_wait() -> None:
    async with running_worker(State(), applying) as (worker, events, _):
        await events.send(snapshot(NEW, State()))
        await wait_until(
            lambda: worker._snapshot_session == NEW  # pyright: ignore[reportPrivateUsage]
        )

        worker.follow_new_master(NEW)

        assert worker._awaiting_state_of is None  # pyright: ignore[reportPrivateUsage]
