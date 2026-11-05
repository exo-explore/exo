"""The master removes a node, and the instances it was part of, when its membership says the node
is Gone (Suspect for the grace period) and its quorum read allows it: with a majority or exactly
half of the cluster reachable, not with less. When a node was last heard from (last_seen) no
longer matters: it is only shown."""

from datetime import datetime, timedelta, timezone

import pytest

from exo.master.main import Master
from exo.routing.membership import (
    LossReason,
    MemberState,
    Quorum,
    reachable_topology,
)
from exo.routing.tests.fake_membership import FakeMembership
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

MASTER = NodeId("master-node")
SILENT = NodeId("silent-node")
HEALTHY = NodeId("healthy-node")
PEER_A = NodeId("peer-a")
PEER_B = NodeId("peer-b")
PEER_C = NodeId("peer-c")


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
    last_seen: dict[NodeId, datetime],
    *instances: MlxRingInstance,
    membership: FakeMembership | None = None,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    """A master whose state lists the nodes in `last_seen`, and the events it sends. Its
    membership, unless given, sees every one of them Alive, with a majority of the cluster
    reachable."""
    topology = Topology()
    for node in last_seen:
        topology.add_node(node)
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    master = Master(
        MASTER,
        SessionId(master_node_id=MASTER, election_clock=0),
        command_receiver=channel[ForwarderCommand]()[1],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        membership=membership or FakeMembership(MASTER, *last_seen),
        initial_state=State(
            topology=topology,
            last_seen=last_seen,
            instances={instance.instance_id: instance for instance in instances},
        ),
    )
    return master, broadcast


def membership(
    quorum: Quorum,
    alive: tuple[NodeId, ...] = (),
    gone: tuple[NodeId, ...] = (),
    suspect: tuple[NodeId, ...] = (),
    reason: LossReason = "Disconnected",
) -> FakeMembership:
    """The master's membership view: these nodes Alive, Gone and Suspect, and its quorum read"""
    view = FakeMembership(MASTER, MASTER, *alive, quorum=quorum)
    states: list[tuple[tuple[NodeId, ...], MemberState]] = [
        (gone, "Gone"),
        (suspect, "Suspect"),
    ]
    for nodes, state in states:
        for node in nodes:
            view.set(node, state, reason)
    return view


def removals(broadcast: Receiver[GlobalForwarderEvent]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for message in broadcast.collect():
        match message.event:
            case NodeTimedOut(node_id=node_id):
                out.append(("node", node_id))
            case InstanceDeleted(instance_id=instance_id, reason="NodeLost"):
                out.append(("instance", instance_id))
            case event:
                out.append(("other", type(event).__name__))
    return out


def ago(delta: timedelta) -> datetime:
    return datetime.now(tz=timezone.utc) - delta


def listed(*nodes: NodeId) -> dict[NodeId, datetime]:
    """The nodes the master's state lists, last heard from a minute ago (which doesn't matter)"""
    return {node: ago(timedelta(minutes=1)) for node in nodes}


def check(master: Master) -> None:
    master._remove_silent_nodes_and_broken_instances()  # pyright: ignore[reportPrivateUsage]


# ---- Gone, and the quorum read ----


async def test_a_gone_node_and_its_instances_are_removed_in_the_same_pass() -> None:
    spanning, elsewhere = instance_on(SILENT, HEALTHY), instance_on(HEALTHY)
    master, events = master_with(
        listed(MASTER, SILENT, HEALTHY),
        spanning,
        elsewhere,
        membership=membership("Majority", alive=(HEALTHY,), gone=(SILENT,)),
    )

    check(master)

    assert removals(events) == [
        ("node", SILENT),
        ("instance", spanning.instance_id),
    ]


@pytest.mark.parametrize("reason", ["Disconnected", "Wedged"])
async def test_a_suspect_node_is_kept_however_long_ago_it_was_heard_from(
    reason: LossReason,
) -> None:
    """Unreachable now (its connection lost, or exo on it wedged), but it may be back within
    the grace period. Its requests end at once anyway: every API checks reachability."""
    master, events = master_with(
        {MASTER: ago(timedelta()), SILENT: ago(timedelta(minutes=10))},
        instance_on(SILENT),
        membership=membership("Majority", suspect=(SILENT,), reason=reason),
    )

    check(master)

    assert removals(events) == []


async def test_a_wedged_node_is_removed_once_it_is_gone() -> None:
    instance = instance_on(SILENT)
    master, events = master_with(
        listed(MASTER, SILENT),
        instance,
        membership=membership("Majority", gone=(SILENT,), reason="Wedged"),
    )

    check(master)

    assert removals(events) == [("node", SILENT), ("instance", instance.instance_id)]


async def test_a_node_that_hasnt_reported_for_long_is_kept_while_it_is_alive() -> None:
    """last_seen is only shown now: the membership decides"""
    master, events = master_with(
        {MASTER: ago(timedelta()), SILENT: ago(timedelta(minutes=10))},
        instance_on(SILENT),
    )

    check(master)

    assert removals(events) == []


async def test_a_pair_keeps_a_gone_node_and_its_instances_until_the_split_hold() -> (
    None
):
    """A 2-node cluster whose other node has been Gone for longer than the grace, but not for
    the split hold: a Mac whose lid closed for a minute or two comes back to its models.
    Removing the node, or ending its instances, at Gone lost the model of a follower asleep
    for 35 s on real Macs: it was back 5 s after it was Gone (discovery holds off for 12 s
    after a stall). Its requests fail fast meanwhile (every API checks reachability)."""
    instance = instance_on(MASTER, PEER_A)
    master, events = master_with(
        listed(MASTER, PEER_A),
        instance,
        membership=membership("Half", gone=(PEER_A,)),
    )

    for _ in range(3):
        check(master)

    assert removals(events) == []


async def test_an_even_split_keeps_the_other_half_and_its_instances() -> None:
    instance = instance_on(PEER_B, PEER_C)
    master, events = master_with(
        listed(MASTER, PEER_A, PEER_B, PEER_C),
        instance,
        membership=membership("Half", alive=(PEER_A,), gone=(PEER_B, PEER_C)),
    )

    check(master)

    assert removals(events) == []


async def test_a_pair_removes_the_gone_node_and_ends_its_instances_after_the_split_hold() -> (
    None
):
    """After FORGET_AFTER (the split hold) Gone, the node no longer counts: the read is a
    majority of the nodes this one reaches, and the node is removed and its instances end"""
    instance = instance_on(MASTER, PEER_A)
    view = membership("Half", gone=(PEER_A,))
    master, events = master_with(listed(MASTER, PEER_A), instance, membership=view)
    check(master)
    assert removals(events) == []

    view.quorum = "Majority"
    check(master)

    assert removals(events) == [("node", PEER_A), ("instance", instance.instance_id)]


async def test_a_master_in_a_minority_removes_nothing_and_sends_nothing() -> None:
    """A Mac cut off from the rest of a 3-node cluster becomes its own master. Its peers are
    Gone in its view, but it reaches less than half of the cluster: it is most likely the one
    cut off, and the others still run the models it can't see. No NodeTimedOut and no
    InstanceDeleted, for the nodes or for an instance on them."""
    spanning = instance_on(PEER_A, PEER_B)
    master, events = master_with(
        listed(MASTER, PEER_A, PEER_B),
        spanning,
        membership=membership("Minority", gone=(PEER_A, PEER_B)),
    )

    for _ in range(3):
        check(master)

    assert removals(events) == []


async def test_a_minority_removes_them_once_the_membership_counts_only_what_it_reaches() -> (
    None
):
    """Once the gone nodes have been Gone for FORGET_AFTER (the split hold, 5 min), the
    membership stops counting them, so its quorum read is a majority of the nodes this one
    reaches, and the gone nodes are removed."""
    spanning = instance_on(PEER_A, PEER_B)
    view = membership("Minority", gone=(PEER_A, PEER_B))
    master, events = master_with(
        listed(MASTER, PEER_A, PEER_B), spanning, membership=view
    )
    check(master)
    assert removals(events) == []

    view.quorum = "Majority"
    check(master)

    assert sorted(removals(events)) == sorted(
        [("node", PEER_A), ("node", PEER_B), ("instance", spanning.instance_id)]
    )


async def test_gone_nodes_the_state_doesnt_list_and_this_node_are_never_removed() -> (
    None
):
    view = membership("Majority", gone=(PEER_A,))
    view.set(MASTER, "Gone", "Disconnected")
    master, events = master_with(listed(MASTER), membership=view)

    check(master)

    assert removals(events) == []


# ---- Made once ----


async def test_a_removal_is_made_once() -> None:
    """Each removal is in the state as soon as it is made: the next pass finds nothing to do"""
    master, events = master_with(
        listed(MASTER, SILENT),
        instance_on(SILENT),
        membership=membership("Majority", gone=(SILENT,)),
    )

    check(master)
    check(master)

    assert len(removals(events)) == 2


# ---- This node's own stalls ----


async def test_nodes_are_not_blamed_for_the_masters_machine_having_slept() -> None:
    """On macOS the monotonic clock stops while the machine sleeps. The master's own
    stall check read only that clock, so a master whose Mac slept for a minute saw no stall
    while every follower's last report was a minute old, and removed them all at once. The
    membership reads a clock that counts sleep: after the sleep this node is Resuming, and the
    peers it lost meanwhile are Suspect, not timed toward Gone until a grace period after it.

    On real Macs a master waking from a 50-120 s sleep also removed itself ("117s without
    hearing from it": its own reports were as old) and deleted its own instance: it never
    removes itself now, whatever its view says."""
    view = membership("Minority", suspect=(SILENT,))
    view.set(MASTER, "Resuming")
    master, events = master_with(
        {MASTER: ago(timedelta(seconds=117)), SILENT: ago(timedelta(seconds=117))},
        instance_on(SILENT),
        instance_on(MASTER),
        membership=view,
    )

    check(master)

    assert removals(events) == [], (
        "a woken master removed nodes it couldn't hear while asleep"
    )


# ---- Placement ----


async def test_a_model_is_not_placed_on_a_node_it_cant_reach() -> None:
    view = membership("Majority", alive=(PEER_A, PEER_B), gone=(PEER_C,))
    master, _ = master_with(listed(MASTER, PEER_A, PEER_B, PEER_C), membership=view)
    await view.lose(PEER_B)

    reachable = reachable_topology(master.membership, master.state.topology)

    assert set(reachable.list_nodes()) == {MASTER, PEER_A}
    assert set(master.state.topology.list_nodes()) == {MASTER, PEER_A, PEER_B, PEER_C}
