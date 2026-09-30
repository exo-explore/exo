"""The master removes nodes it hasn't heard from, and the instances they were part of, promptly."""

import time
from datetime import datetime, timedelta, timezone

import pytest

import exo.master.main as master_main
from exo.master.main import NODE_SILENCE_TIMEOUT, Master
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, InstanceDeleted, NodeTimedOut
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, channel

SILENT = NodeId("silent-node")
HEALTHY = NodeId("healthy-node")


def instance_on(*nodes: NodeId) -> MlxRingInstance:
    card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(100),
        n_layers=len(nodes),
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    runners = {node: RunnerId() for node in nodes}
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=rank,
                    world_size=len(nodes),
                    start_layer=rank,
                    end_layer=rank + 1,
                    n_layers=len(nodes),
                )
                for rank, runner in enumerate(runners.values())
            },
            node_to_runner=runners,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def master_with(
    last_seen: dict[NodeId, datetime], *instances: MlxRingInstance
) -> tuple[Master, Receiver[Event]]:
    topology = Topology()
    for node in last_seen:
        topology.add_node(node)
    master = object.__new__(Master)
    master.state = State(
        topology=topology,
        last_seen=last_seen,
        instances={instance.instance_id: instance for instance in instances},
    )
    master.event_sender, events = channel[Event]()
    master._removing_nodes = {}  # pyright: ignore[reportPrivateUsage]
    master._deleting_instances = {}  # pyright: ignore[reportPrivateUsage]
    # Listening for long enough, and checking every second
    master._listening_since = time.monotonic() - 3600  # pyright: ignore[reportPrivateUsage]
    master._last_check = None  # pyright: ignore[reportPrivateUsage]
    return master, events


def removals(events: Receiver[Event]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for event in events.collect():
        match event:
            case NodeTimedOut(node_id=node_id):
                out.append(("node", node_id))
            case InstanceDeleted(instance_id=instance_id):
                out.append(("instance", instance_id))
            case _:
                out.append(("other", type(event).__name__))
    return out


def ago(delta: timedelta) -> datetime:
    return datetime.now(tz=timezone.utc) - delta


async def check(master: Master) -> None:
    await master._remove_silent_nodes_and_broken_instances()  # pyright: ignore[reportPrivateUsage]


async def test_a_silent_node_and_its_instances_are_removed_in_the_same_pass() -> None:
    spanning, elsewhere = instance_on(SILENT, HEALTHY), instance_on(HEALTHY)
    master, events = master_with(
        {
            SILENT: ago(NODE_SILENCE_TIMEOUT + timedelta(seconds=1)),
            HEALTHY: ago(timedelta()),
        },
        spanning,
        elsewhere,
    )

    await check(master)

    assert removals(events) == [
        ("node", SILENT),
        ("instance", spanning.instance_id),
    ]


async def test_a_node_heard_from_recently_is_kept() -> None:
    master, events = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT - timedelta(seconds=2))},
        instance_on(SILENT),
    )

    await check(master)

    assert removals(events) == []


async def test_a_removal_is_not_repeated_while_it_is_being_applied() -> None:
    master, events = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT * 2)}, instance_on(SILENT)
    )

    await check(master)
    await check(master)

    assert len(removals(events)) == 2


async def test_a_removal_that_was_never_applied_is_sent_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, events = master_with({SILENT: ago(NODE_SILENCE_TIMEOUT * 2)})
    await check(master)
    assert removals(events) == [("node", SILENT)]

    monkeypatch.setattr(master_main, "REMOVAL_RESEND_INTERVAL", 0.0)
    await check(master)

    assert removals(events) == [("node", SILENT)]


async def test_nodes_are_not_blamed_for_the_master_being_stalled() -> None:
    master, events = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT + timedelta(seconds=5))}, instance_on(SILENT)
    )
    # The master's previous check was 20 s ago: it was frozen, so it heard nobody
    master._last_check = time.monotonic() - 20  # pyright: ignore[reportPrivateUsage]

    await check(master)

    assert removals(events) == []


async def test_a_node_still_silent_a_full_timeout_after_the_master_recovered_is_removed() -> (
    None
):
    master, events = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT * 3)}, instance_on(SILENT)
    )
    master._last_check = time.monotonic() - 20  # pyright: ignore[reportPrivateUsage]
    await check(master)
    assert removals(events) == []

    # A full timeout of listening later, it still hasn't reported
    master._listening_since -= NODE_SILENCE_TIMEOUT.total_seconds() + 1  # pyright: ignore[reportPrivateUsage]
    await check(master)

    assert [kind for kind, _ in removals(events)] == ["node", "instance"]
