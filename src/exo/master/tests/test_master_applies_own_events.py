"""The master applies its own events as it decides on them.

It used to send them to itself through the network, as nodes send theirs, and its state
changed only when they came back indexed: milliseconds later, or seconds under load. Until
then it decided on a state that lagged its own decisions. Each test here sends commands back
to back, as the API does when a user acts quickly, and the later command must see what the
earlier one did.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import anyio

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    Command,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    RequestEventLog,
    TextGeneration,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
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
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
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
    master = Master(
        node_id,
        session,
        command_receiver=commands,
        local_event_receiver=local_events,
        global_event_sender=global_event_sender,
        snapshot_sender=snapshot_sender,
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
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
