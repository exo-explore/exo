"""The master applies its own events as it decides on them.

It used to send them to itself through the network, as nodes send theirs, and its state
changed only when they came back indexed: milliseconds later, or seconds under load. Until
then it decided on a state that lagged its own decisions. Each test here sends commands back
to back, as the API does when a user acts quickly, and the later command must see what the
earlier one did.
"""

import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import anyio
import pytest

import exo.master.main as master_main
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    Command,
    CreateInstance,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    RequestEventLog,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    StateSnapshot,
    TaskCreated,
    TaskStatusUpdated,
    TestEvent,
)
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from exo.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import Receiver, Sender, channel

MODEL = ModelId("test-model")
NODE = NodeId("node")
# The node has room for one copy of the model, not two
NODE_MEMORY = Memory.from_mb(1000)
MODEL_SIZE = Memory.from_mb(600)


def model_card() -> ModelCard:
    return ModelCard(
        model_id=MODEL,
        storage_size=MODEL_SIZE,
        n_layers=1,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def ready_instance() -> MlxRingInstance:
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=model_card(),
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={NODE: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def cluster(*instances: MlxRingInstance) -> State:
    """One node the master has just heard from, with the instances on it, ready."""
    topology = Topology()
    topology.add_node(NODE)
    return State(
        topology=topology,
        last_seen={NODE: datetime.now(tz=timezone.utc)},
        node_memory={
            NODE: MemoryUsage(
                ram_total=NODE_MEMORY,
                ram_available=NODE_MEMORY,
                swap_total=Memory(),
                swap_available=Memory(),
            )
        },
        node_backends={NODE: [Backend.MlxMetal]},
        instances={instance.instance_id: instance for instance in instances},
        runners={
            runner: RunnerReady()
            for instance in instances
            for runner in instance.shard_assignments.runner_to_shard
        },
    )


def chat_request() -> TextGeneration:
    return TextGeneration(
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        )
    )


def place_model() -> PlaceInstance:
    return PlaceInstance(
        model_card=model_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


@dataclass
class RunningMaster:
    master: Master
    session: SessionId
    commands: Sender[ForwarderCommand]
    local_events: Sender[LocalForwarderEvent]
    broadcast: Receiver[GlobalForwarderEvent]
    snapshots: Receiver[StateSnapshot]
    download_commands: Receiver[ForwarderDownloadCommand]
    sent: list[GlobalForwarderEvent]

    async def send(self, *commands: Command, origin: SystemId | None = None) -> None:
        for command in commands:
            await self.commands.send(
                ForwarderCommand(origin=origin or SystemId(), command=command)
            )

    async def sent_until(
        self, done: Callable[[list[GlobalForwarderEvent]], bool]
    ) -> list[GlobalForwarderEvent]:
        """What the master sent to every node, once it has sent enough."""
        with anyio.fail_after(5):
            while not done(self.sent):
                self.sent.append(await self.broadcast.receive())
        return self.sent

    async def all_handled(self) -> list[Event]:
        """Everything the master sent for the commands sent so far: a last command's event
        comes after all of theirs."""
        last = InstanceLinkId()
        await self.send(DeleteInstanceLink(link_id=last))
        sent = await self.sent_until(
            lambda sent: any(
                isinstance(message.event, InstanceLinkDeleted)
                and message.event.link_id == last
                for message in sent
            )
        )
        return [message.event for message in sent]


@asynccontextmanager
async def running_master(state: State) -> AsyncIterator[RunningMaster]:
    node_id = get_node_zid()
    session = SessionId(master_node_id=node_id, election_clock=0)
    command_sender, commands = channel[ForwarderCommand]()
    local_event_sender, local_events = channel[LocalForwarderEvent]()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    snapshot_sender, snapshots = channel[StateSnapshot]()
    download_command_sender, download_commands = channel[ForwarderDownloadCommand]()
    master = Master(
        node_id,
        session,
        command_receiver=commands,
        local_event_receiver=local_events,
        global_event_sender=global_event_sender,
        snapshot_sender=snapshot_sender,
        download_command_sender=download_command_sender,
        initial_state=state,
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        yield RunningMaster(
            master,
            session,
            command_sender,
            local_event_sender,
            broadcast,
            snapshots,
            download_commands,
            [],
        )
        tg.cancel_scope.cancel()


async def test_two_launches_in_a_row_do_not_share_the_same_memory() -> None:
    """The second launch sees the first one's instance, still loading, and the memory it will
    take. It used to see the memory free and put a second copy in it."""
    async with running_master(cluster()) as running:
        await running.send(place_model(), place_model())
        sent = await running.all_handled()

        assert len([event for event in sent if isinstance(event, InstanceCreated)]) == 1
        assert len(running.master.state.instances) == 1


async def test_two_given_placements_sent_together_do_not_share_the_same_memory() -> (
    None
):
    """The same for CreateInstance, which launches a placement the API made (from the
    dashboard's preview, or for POST /place_instance): the API checks it against its own state,
    which doesn't show the other launch yet, so the master checks it again."""
    first, second = ready_instance(), ready_instance()
    async with running_master(cluster()) as running:
        await running.send(
            CreateInstance(instance=first), CreateInstance(instance=second)
        )
        sent = await running.all_handled()

        assert [
            event.instance.instance_id
            for event in sent
            if isinstance(event, InstanceCreated)
        ] == [first.instance_id]
        assert list(running.master.state.instances) == [first.instance_id]


async def test_a_launch_sent_again_changes_nothing() -> None:
    """The API sends a launch again until it shows: a copy of one already applied is not
    refused for the memory the instance itself takes."""
    instance = ready_instance()
    async with running_master(cluster()) as running:
        await running.send(
            CreateInstance(instance=instance), CreateInstance(instance=instance)
        )
        sent = await running.all_handled()

        assert len([event for event in sent if isinstance(event, InstanceCreated)]) == 1
        assert list(running.master.state.instances) == [instance.instance_id]


async def test_a_command_that_fails_to_apply_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command takes effect whole or not at all. Here the deletion of an instance cancels
    its request, then fails to delete it: neither happens, nothing is sent, and its download
    cancellation isn't sent either. The master goes on with the next command."""

    def failing_on_deletion(state: State, event: IndexedEvent) -> State:
        if isinstance(event.event, InstanceDeleted):
            raise RuntimeError("a bug in apply")
        return apply(state, event)

    instance = ready_instance()
    (runner,) = instance.shard_assignments.runner_to_shard
    downloading = cluster(instance).model_copy(
        update={
            "downloads": {
                NODE: [
                    DownloadOngoing(
                        node_id=NODE,
                        shard_metadata=instance.shard_assignments.runner_to_shard[
                            runner
                        ],
                        download_progress=DownloadProgressData(
                            total=MODEL_SIZE,
                            downloaded=Memory(),
                            downloaded_this_session=Memory(),
                            completed_files=0,
                            total_files=1,
                            speed=0.0,
                            eta_ms=0,
                            files={},
                        ),
                    )
                ]
            }
        }
    )
    async with running_master(downloading) as running:
        await running.send(chat_request())
        await running.sent_until(
            lambda sent: any(isinstance(m.event, TaskCreated) for m in sent)
        )
        before = running.master.state
        monkeypatch.setattr(master_main, "apply", failing_on_deletion)

        await running.send(DeleteInstance(instance_id=instance.instance_id))
        sent = await running.all_handled()

        assert [type(event) for event in sent] == [TaskCreated, InstanceLinkDeleted]
        assert running.master.state.instances == before.instances
        assert running.master.state.tasks == before.tasks
        assert [task.task_status for task in running.master.state.tasks.values()] == [
            TaskStatus.Pending
        ]
        assert (
            running.master.state.last_event_applied_idx
            == before.last_event_applied_idx + 1
        )
        assert running.download_commands.collect() == []


@pytest.mark.parametrize("seed", range(4))
async def test_a_node_applying_what_the_master_sends_ends_with_its_state(
    seed: int,
) -> None:
    """A random mix of commands (chats, launches, stops, finishes, cancels, requests for missed
    events) and of events from two nodes, some out of order. A node that applies everything
    the master sends, in index order and each index once, ends with the master's state, and
    the indices run on with no gap; a replayed index carries the very same event."""
    rng = random.Random(seed)
    instance = ready_instance()
    initial = cluster(instance)
    async with running_master(initial) as running:
        origins = [SystemId(), SystemId()]
        next_origin_idx = [0, 0]
        held: list[LocalForwarderEvent] = []
        chats: list[TextGeneration] = []
        for _ in range(300):
            roll = rng.random()
            if roll < 0.35:
                origin = rng.randrange(2)
                event = LocalForwarderEvent(
                    origin_idx=next_origin_idx[origin],
                    origin=origins[origin],
                    session=running.session,
                    event=TestEvent(),
                )
                next_origin_idx[origin] += 1
                if rng.random() < 0.3:
                    held.append(event)
                else:
                    await running.local_events.send(event)
                    if held and rng.random() < 0.5:
                        await running.local_events.send(held.pop(0))
            elif roll < 0.55:
                chats.append(chat_request())
                await running.send(chats[-1])
            elif roll < 0.65 and chats:
                chat = rng.choice(chats)
                await running.send(
                    rng.choice(
                        [
                            TaskFinished(finished_command_id=chat.command_id),
                            TaskCancelled(cancelled_command_id=chat.command_id),
                        ]
                    )
                )
            elif roll < 0.72:
                await running.send(
                    rng.choice(
                        [place_model(), CreateInstance(instance=ready_instance())]
                    )
                )
            elif roll < 0.78:
                if instances := list(running.master.state.instances):
                    await running.send(
                        DeleteInstance(instance_id=rng.choice(instances))
                    )
            elif roll < 0.85:
                await running.send(RequestEventLog(since_idx=rng.randrange(0, 50)))
            else:
                await anyio.sleep(0)
        for event in held:
            await running.local_events.send(event)
        await running.all_handled()
        await anyio.sleep(0.1)
        running.sent.extend(running.broadcast.collect())

        received: dict[int, Event] = {}
        for message in running.sent:
            if message.origin_idx in received:
                assert received[message.origin_idx] == message.event
            else:
                received[message.origin_idx] = message.event
        assert sorted(received) == list(range(len(received)))
        follower = initial
        for idx in sorted(received):
            follower = apply(follower, IndexedEvent(idx=idx, event=received[idx]))
        assert follower.model_dump() == running.master.state.model_dump()


async def test_an_instance_deleted_twice_in_a_row_is_deleted_once() -> None:
    """A second DeleteInstance (the user stops a model twice, or a worker gives up on it as
    the user stops it) finds the instance gone."""
    instance = ready_instance()
    async with running_master(cluster(instance)) as running:
        await running.send(
            DeleteInstance(instance_id=instance.instance_id),
            DeleteInstance(instance_id=instance.instance_id),
        )
        sent = await running.all_handled()

        assert [
            event.instance_id for event in sent if isinstance(event, InstanceDeleted)
        ] == [instance.instance_id]


async def test_own_and_node_events_share_one_sequence() -> None:
    """The master's own events and the ones nodes send are numbered in one sequence, sent in
    its order, kept for replay and covered by snapshots, as if they had all come from nodes.
    A command's events are numbered together, in the order it made them."""
    instance = ready_instance()
    async with running_master(cluster(instance)) as running:
        worker = SystemId()

        async def from_node(origin_idx: int) -> None:
            await running.local_events.send(
                LocalForwarderEvent(
                    origin_idx=origin_idx,
                    origin=worker,
                    session=running.session,
                    event=TestEvent(),
                )
            )

        await from_node(0)
        await running.send(chat_request())
        await from_node(1)
        await running.send(DeleteInstance(instance_id=instance.instance_id))
        await from_node(2)
        live = list(await running.sent_until(lambda sent: len(sent) == 6))
        await anyio.sleep(0.05)
        assert running.broadcast.collect() == []

        assert [message.origin_idx for message in live] == list(range(6))
        kinds = [type(message.event) for message in live]
        assert sorted(kinds, key=lambda kind: kind.__name__) == sorted(
            [TestEvent] * 3 + [TaskCreated, TaskStatusUpdated, InstanceDeleted],
            key=lambda kind: kind.__name__,
        )
        # The deletion cancels the request first, then deletes the instance, in two indices
        # in a row
        cancelled = kinds.index(TaskStatusUpdated)
        assert kinds[cancelled + 1] is InstanceDeleted
        assert kinds.index(TaskCreated) < cancelled
        assert running.master.state.last_event_applied_idx == 5
        assert instance.instance_id not in running.master.state.instances

        # A node that missed them is sent the same events again, with the same indices
        running.sent.clear()
        await running.send(RequestEventLog(since_idx=0))
        replayed = await running.sent_until(lambda sent: len(sent) == 6)
        assert [(m.origin_idx, m.event) for m in replayed] == [
            (m.origin_idx, m.event) for m in live
        ]

        # ...or a snapshot that covers all of them
        requester = SystemId()
        await running.send(
            RequestEventLog(since_idx=0, snapshot=True), origin=requester
        )
        with anyio.fail_after(5):
            snapshot = await running.snapshots.receive()
        assert snapshot.requester == requester
        assert snapshot.state.last_event_applied_idx == 5
        assert snapshot.state is running.master.state
