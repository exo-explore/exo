"""The master acts on a command only once, however often it arrives, even after what it did
was undone. A command that changed nothing (a launch onto a node outside the cluster) can still
take effect when a copy comes."""

from collections import deque
from datetime import datetime, timezone

import anyio
import pytest
from loguru import logger

import exo.master.main as master_main
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceCreated,
    InstanceDeleted,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, channel

MODEL = ModelId("test-model")
NODE = NodeId("node")
LOST_NODE = NodeId("lost-node")


def instance_on(node: NodeId) -> MlxRingInstance:
    card = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def master_with(
    *instances: MlxRingInstance,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    """A master with the instances, ready, on a node it has just heard from, and what it
    sends to every node."""
    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    topology = Topology()
    topology.add_node(NODE)
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=channel[ForwarderCommand]()[1],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=State(
            topology=topology,
            last_seen={NODE: datetime.now(tz=timezone.utc)},
            instances={instance.instance_id: instance for instance in instances},
            runners={
                runner: RunnerReady()
                for instance in instances
                for runner in instance.shard_assignments.runner_to_shard
            },
        ),
    )
    return master, broadcast


def fresh_master() -> Master:
    master = object.__new__(Master)
    master.state = State()
    master._processed_commands = set()  # pyright: ignore[reportPrivateUsage]
    master._processed_order = deque()  # pyright: ignore[reportPrivateUsage]
    return master


def test_a_repeated_command_is_only_processed_once() -> None:
    master = fresh_master()
    command_id = CommandId()

    assert not master._already_processed(command_id)  # pyright: ignore[reportPrivateUsage]
    master._remember_processed(command_id)  # pyright: ignore[reportPrivateUsage]
    assert master._already_processed(command_id)  # pyright: ignore[reportPrivateUsage]
    assert not master._already_processed(CommandId())  # pyright: ignore[reportPrivateUsage]


def test_only_recent_commands_are_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(master_main, "PROCESSED_COMMANDS_KEPT", 2)
    master = fresh_master()
    oldest, *rest = [CommandId() for _ in range(3)]
    for command_id in (oldest, *rest):
        master._remember_processed(command_id)  # pyright: ignore[reportPrivateUsage]

    assert len(master._processed_commands) == 2  # pyright: ignore[reportPrivateUsage]
    assert not master._already_processed(oldest)  # pyright: ignore[reportPrivateUsage]


def master_with_room() -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    """A master whose nodes, the dropped one included, have reported room for a launch."""
    master, broadcast = master_with()
    room = MemoryUsage(
        ram_total=Memory.from_gb(64),
        ram_available=Memory.from_gb(64),
        swap_total=Memory(),
        swap_available=Memory(),
    )
    master.state = master.state.model_copy(
        update={"node_memory": {NODE: room, LOST_NODE: room}}
    )
    return master, broadcast


async def next_event[T: Event](
    broadcast: Receiver[GlobalForwarderEvent], kind: type[T]
) -> T:
    """The next event of this kind the master sends to every node."""
    while True:
        event = (await broadcast.receive()).event
        if isinstance(event, kind):
            return event


async def test_a_copy_of_a_launch_after_its_instance_was_stopped_is_ignored() -> None:
    """The API sends a launch again until its own node's state shows it, under the same id,
    so a node that lags sends copies after the master placed it. One arriving after the
    instance was stopped (through another node) must not bring it back."""
    master, events = master_with_room()
    launch = CreateInstance(instance=instance_on(NODE))
    commands_sender, master.command_receiver = channel[ForwarderCommand]()
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await commands_sender.send(ForwarderCommand(origin=SystemId(), command=launch))
        with anyio.fail_after(5):
            await next_event(events, InstanceCreated)
        # Stopped since, through another node
        stop = DeleteInstance(instance_id=launch.instance.instance_id)
        await commands_sender.send(ForwarderCommand(origin=SystemId(), command=stop))
        with anyio.fail_after(5):
            await next_event(events, InstanceDeleted)
        # The lagging node's copy of the launch
        await commands_sender.send(ForwarderCommand(origin=SystemId(), command=launch))
        await anyio.sleep(0.3)
        assert not [
            sent for sent in events.collect() if isinstance(sent.event, InstanceCreated)
        ]
        tg.cancel_scope.cancel()


async def test_a_launch_on_a_node_the_master_dropped_waits_until_it_is_back() -> None:
    """As a split heals, a node's launches can reach a master that dropped the node and
    hasn't added it back yet. Created, the instance would be deleted at the master's next
    check. It isn't created, and since it changed nothing, the same launch sent again once
    the node is back is."""
    master, events = master_with_room()
    launch = CreateInstance(instance=instance_on(LOST_NODE))
    commands_sender, master.command_receiver = channel[ForwarderCommand]()
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await commands_sender.send(ForwarderCommand(origin=SystemId(), command=launch))
        await anyio.sleep(0.2)
        assert events.collect() == []

        master.state.topology.add_node(LOST_NODE)  # it is back
        await commands_sender.send(ForwarderCommand(origin=SystemId(), command=launch))
        with anyio.fail_after(5):
            created = await next_event(events, InstanceCreated)
        assert created.instance == launch.instance
        tg.cancel_scope.cancel()


async def test_a_stop_of_an_instance_already_gone_is_logged_in_one_line() -> None:
    """The API sends a stop again until its node sees the instance go, so copies can reach a
    master that deleted it already: nothing to do, and no error with a traceback."""
    master, events = master_with()
    stop = DeleteInstance(instance_id=instance_on(NODE).instance_id)
    commands_sender, master.command_receiver = channel[ForwarderCommand]()
    logged: list[str] = []
    sink = logger.add(logged.append, level="DEBUG", format="{level} {message}")
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
            await commands_sender.send(
                ForwarderCommand(origin=SystemId(), command=stop)
            )
            await anyio.sleep(0.2)
            tg.cancel_scope.cancel()
    finally:
        logger.remove(sink)

    assert events.collect() == []
    assert not [line for line in logged if "Error in command processor" in line]
    assert [line for line in logged if "already gone" in line]
