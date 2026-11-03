"""The master removes nodes it hasn't heard from, and the instances they were part of, promptly."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

import exo.master.main as master_main
from exo.master.main import (
    NODE_SILENCE_TIMEOUT,
    PLAN_INTERVAL,
    RECONNECT_ALLOWANCE,
    Master,
)
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    InstanceDeleted,
    LocalForwarderEvent,
    NodeTimedOut,
    StateSnapshot,
)
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
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    topology = Topology()
    for node in last_seen:
        topology.add_node(node)
    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
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
            last_seen=last_seen,
            instances={instance.instance_id: instance for instance in instances},
        ),
    )
    # Listening for long enough
    master._listening_since = master_main.time.monotonic() - 3600  # pyright: ignore[reportPrivateUsage]
    return master, broadcast


def removals(broadcast: Receiver[GlobalForwarderEvent]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for message in broadcast.collect():
        match message.event:
            case NodeTimedOut(node_id=node_id):
                out.append(("node", node_id))
            case InstanceDeleted(instance_id=instance_id):
                out.append(("instance", instance_id))
            case event:
                out.append(("other", type(event).__name__))
    return out


def ago(delta: timedelta) -> datetime:
    return datetime.now(tz=timezone.utc) - delta


def check(master: Master) -> None:
    master._remove_silent_nodes_and_broken_instances()  # pyright: ignore[reportPrivateUsage]


@dataclass
class FakeClock:
    """Stands in for the time module in exo.master.main, so a test can let seconds pass."""

    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


def test_a_silent_node_and_its_instances_are_removed_in_the_same_pass() -> None:
    spanning, elsewhere = instance_on(SILENT, HEALTHY), instance_on(HEALTHY)
    master, broadcast = master_with(
        {
            SILENT: ago(NODE_SILENCE_TIMEOUT + timedelta(seconds=1)),
            HEALTHY: ago(timedelta()),
        },
        spanning,
        elsewhere,
    )

    check(master)

    assert removals(broadcast) == [
        ("node", SILENT),
        ("instance", spanning.instance_id),
    ]


def test_a_node_heard_from_recently_is_kept() -> None:
    master, broadcast = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT - timedelta(seconds=2))},
        instance_on(SILENT),
    )

    check(master)

    assert removals(broadcast) == []


def test_a_removal_is_sent_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The master's state has a removal as soon as the master makes it. It used to wait for
    the removal to come back indexed, and sent it again if that took more than 5 s."""
    clock = FakeClock()
    monkeypatch.setattr(master_main, "time", clock)
    instance = instance_on(SILENT)
    master, broadcast = master_with({SILENT: ago(NODE_SILENCE_TIMEOUT * 2)}, instance)

    # Ten seconds of checks
    for _ in range(10):
        check(master)
        clock.now += PLAN_INTERVAL

    assert removals(broadcast) == [
        ("node", SILENT),
        ("instance", instance.instance_id),
    ]
    assert SILENT not in master.state.last_seen
    assert instance.instance_id not in master.state.instances


def test_nodes_are_not_blamed_for_the_master_being_stalled() -> None:
    master, broadcast = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT + timedelta(seconds=5))}, instance_on(SILENT)
    )
    # The master's previous check was 20 s ago: it was frozen, so it heard nobody
    master._last_check = master_main.time.monotonic() - 20  # pyright: ignore[reportPrivateUsage]

    check(master)

    assert removals(broadcast) == []


def test_a_node_still_silent_a_full_timeout_after_the_master_recovered_is_removed() -> (
    None
):
    master, broadcast = master_with(
        {SILENT: ago(NODE_SILENCE_TIMEOUT * 3)}, instance_on(SILENT)
    )
    master._last_check = master_main.time.monotonic() - 20  # pyright: ignore[reportPrivateUsage]
    check(master)
    assert removals(broadcast) == []

    # Time to reconnect and a full timeout of listening later, it still hasn't reported
    master._listening_since -= (  # pyright: ignore[reportPrivateUsage]
        (RECONNECT_ALLOWANCE + NODE_SILENCE_TIMEOUT).total_seconds() + 1
    )
    check(master)

    assert [kind for kind, _ in removals(broadcast)] == ["node", "instance"]
