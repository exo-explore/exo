"""The Python side of membership: what the networking layer reports reaches the rest of the node
as each node's latest state, the nodes that leave Alive, and the quorum read."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import cast

import anyio
import exo_rs
import pytest
from anyio import create_task_group, fail_after
from exo_rs import FromSwarm, LossReason, MembershipSnapshot, NetworkingHandle, Quorum

import exo.routing.topics as topics
from exo.routing.membership import (
    GRACE,
    SPLIT_HOLD,
    WEDGE_TIMEOUT,
    Member,
    Membership,
    MembershipChange,
    MembershipView,
    MemberState,
    MemberView,
    Reachability,
)
from exo.routing.membership import (
    LossReason as LossReasonName,
)
from exo.routing.router import Router
from exo.routing.tests.fake_networking import FakeNetworking, no_snapshot
from exo.routing.tests.fake_reachability import FakeReachability
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceDeleted,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshot,
)
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import MiscData

ME = NodeId("ME")
PEER = NodeId("PEER")
OTHER = NodeId("OTHER")
SINCE = 1_800_000_000.0


def member(
    node: NodeId,
    state: MemberState,
    reason: LossReasonName | None = None,
    since: float = SINCE,
) -> Member:
    return Member(
        node_id=node,
        state=state,
        reason=reason,
        since=datetime.fromtimestamp(since, tz=UTC),
    )


def in_view(node: Member, counted: bool = True) -> MemberView:
    return MemberView(
        node_id=node.node_id,
        state=node.state,
        reason=node.reason,
        since=node.since,
        counted=counted,
    )


@asynccontextmanager
async def running_membership(
    snapshot: Callable[[], MembershipSnapshot] = no_snapshot,
) -> AsyncIterator[tuple[Membership, Sender[Member]]]:
    sender, receiver = channel[Member]()
    _, cluster_events = channel[IndexedEvent | StateSnapshot]()
    membership = Membership(
        ME,
        member_receiver=receiver,
        networking=FakeNetworking(snapshot),
        cluster_events=cluster_events,
    )
    async with create_task_group() as tg:
        tg.start_soon(membership.run)
        yield membership, sender
        sender.close()


async def settle() -> None:
    """Lets the membership task take what was sent"""
    for _ in range(10):
        await anyio.sleep(0)


# ---- From the networking layer ----


@pytest.mark.parametrize(
    ("state", "reason", "expected_state", "expected_reason"),
    [
        (exo_rs.MemberState.Alive, None, "Alive", None),
        (exo_rs.MemberState.Resuming, None, "Resuming", None),
        (
            exo_rs.MemberState.Suspect,
            LossReason.Disconnected,
            "Suspect",
            "Disconnected",
        ),
        (exo_rs.MemberState.Suspect, LossReason.Wedged, "Suspect", "Wedged"),
        (exo_rs.MemberState.Gone, LossReason.Left, "Gone", "Left"),
    ],
)
def test_a_change_from_the_networking_layer_keeps_the_node_its_state_and_why(
    state: exo_rs.MemberState,
    reason: LossReason | None,
    expected_state: MemberState,
    expected_reason: LossReasonName | None,
) -> None:
    update = FromSwarm.Membership("1f2e", state, reason, SINCE)
    assert Member.from_swarm(update) == member(
        NodeId("1f2e"), expected_state, expected_reason
    )


class ReplayingHandle:
    """A networking handle that reports the given events, then nothing"""

    def __init__(self, events: list[FromSwarm]) -> None:
        self._events = events

    async def recv(self) -> FromSwarm:
        if self._events:
            return self._events.pop(0)
        await anyio.sleep_forever()
        raise AssertionError("unreachable")


async def test_the_router_hands_each_membership_change_to_the_membership_topic() -> (
    None
):
    handle = ReplayingHandle(
        [
            FromSwarm.Membership("a1", exo_rs.MemberState.Alive, None, SINCE),
            FromSwarm.Membership(
                "b2", exo_rs.MemberState.Suspect, LossReason.Disconnected, SINCE
            ),
        ]
    )
    router = Router(cast(NetworkingHandle, cast(object, handle)))
    await router.register_topic(topics.MEMBERSHIP)
    received = router.receiver(topics.MEMBERSHIP)
    got: list[Member] = []
    async with create_task_group() as tg:
        tg.start_soon(router._networking_recv)  # pyright: ignore[reportPrivateUsage]
        with fail_after(2):
            got = [await received.receive(), await received.receive()]
        tg.cancel_scope.cancel()
    assert tuple(got) == (
        member(NodeId("a1"), "Alive"),
        member(NodeId("b2"), "Suspect", "Disconnected"),
    )


def test_membership_is_never_sent_to_other_nodes() -> None:
    assert topics.MEMBERSHIP.publish_policy is topics.PublishPolicy.Never


# ---- The latest state of each node ----


async def test_a_node_is_reachable_only_while_it_is_alive() -> None:
    async with running_membership() as (membership, events):
        assert not membership.is_reachable(PEER), "never heard of"
        await events.send(member(PEER, "Alive"))
        await settle()
        assert membership.is_reachable(PEER)
        await events.send(member(PEER, "Suspect", "Disconnected"))
        await settle()
        assert not membership.is_reachable(PEER)
        await events.send(member(PEER, "Gone", "Disconnected"))
        await settle()
        assert not membership.is_reachable(PEER)
        await events.send(member(PEER, "Alive"))
        await settle()
        assert membership.is_reachable(PEER)
        assert membership.state_of(PEER) == member(PEER, "Alive")


async def test_this_node_is_always_reachable_to_itself() -> None:
    async with running_membership() as (membership, events):
        assert membership.is_reachable(ME)
        await events.send(member(ME, "Resuming"))
        await settle()
        assert membership.is_reachable(ME)


async def test_watchers_hear_of_each_node_that_leaves_alive_and_nothing_else() -> None:
    async with running_membership() as (membership, events):
        first = membership.watch()
        second = membership.watch()
        for event in [
            member(ME, "Alive"),
            member(PEER, "Alive"),
            member(OTHER, "Alive"),
            member(PEER, "Suspect", "Disconnected"),
            # Gone after Suspect: it already left Alive
            member(PEER, "Gone", "Disconnected"),
            # It left: Gone straight from Alive
            member(OTHER, "Gone", "Left"),
            member(ME, "Resuming"),
            member(ME, "Alive"),
            member(PEER, "Alive"),
            member(PEER, "Suspect", "Wedged"),
        ]:
            await events.send(event)
        await settle()
        assert first.collect() == [PEER, OTHER, PEER]
        assert second.collect() == [PEER, OTHER, PEER]


async def test_a_watcher_that_stops_listening_is_forgotten() -> None:
    async with running_membership() as (membership, events):
        gone = membership.watch()
        listening = membership.watch()
        gone.close()
        await events.send(member(PEER, "Alive"))
        await events.send(member(PEER, "Suspect", "Disconnected"))
        await settle()
        assert listening.collect() == [PEER]


async def test_changes_carry_the_state_before() -> None:
    async with running_membership() as (membership, events):
        changes = membership.changes()
        await events.send(member(PEER, "Alive"))
        await events.send(member(PEER, "Suspect", "Disconnected"))
        await settle()
        assert changes.collect() == [
            MembershipChange(node_id=PEER, previous=None, current="Alive", reason=None),
            MembershipChange(
                node_id=PEER,
                previous="Alive",
                current="Suspect",
                reason="Disconnected",
            ),
        ]


# ---- The quorum read and the view the API shows ----


def snapshot_of(
    members: list[exo_rs.Member], quorum: Quorum, reachable: int, last_known: int
) -> Callable[[], MembershipSnapshot]:
    return lambda: MembershipSnapshot(members, quorum, reachable, last_known)


async def test_the_view_is_the_networking_layers_snapshot_now() -> None:
    snapshot = snapshot_of(
        [
            exo_rs.Member("ME", exo_rs.MemberState.Resuming, None, SINCE),
            exo_rs.Member(
                "PEER", exo_rs.MemberState.Gone, LossReason.Disconnected, SINCE + 1
            ),
            # Gone for a while: in the view, not counted
            exo_rs.Member(
                "OTHER",
                exo_rs.MemberState.Gone,
                LossReason.Disconnected,
                SINCE + 2,
                counted=False,
            ),
        ],
        Quorum.Minority,
        0,
        2,
    )
    async with running_membership(snapshot) as (membership, _):
        assert membership.view() == MembershipView(
            node_id=ME,
            self_state="Resuming",
            quorum="Minority",
            reachable=0,
            last_known=2,
            members=[
                in_view(member(ME, "Resuming")),
                in_view(member(PEER, "Gone", "Disconnected", since=SINCE + 1)),
                in_view(
                    member(OTHER, "Gone", "Disconnected", since=SINCE + 2),
                    counted=False,
                ),
            ],
        )


@pytest.mark.parametrize(
    ("quorum", "expected"),
    [
        (Quorum.Majority, "Majority"),
        (Quorum.Half, "Half"),
        (Quorum.Minority, "Minority"),
    ],
)
def test_each_quorum_read_reaches_python(quorum: Quorum, expected: str) -> None:
    own = exo_rs.Member("ME", exo_rs.MemberState.Alive, None, SINCE)
    view = MembershipView.from_snapshot(ME, MembershipSnapshot([own], quorum, 1, 2))
    assert view.quorum == expected


# ---- Reachability ----


async def wait_for_unreachable(reachability: Reachability, node: NodeId) -> NodeId:
    """What a reader of `Reachability` does: hears when a node becomes unreachable"""
    watched: Receiver[NodeId] = reachability.watch()
    with fail_after(2):
        lost = await watched.receive()
    assert not reachability.is_reachable(lost)
    return lost


async def test_membership_and_its_fake_both_tell_readers_of_reachability() -> None:
    async with running_membership() as (membership, events):
        await events.send(member(PEER, "Alive"))
        await settle()
        results: list[NodeId] = []
        async with create_task_group() as tg:

            async def watch() -> None:
                results.append(await wait_for_unreachable(membership, PEER))

            tg.start_soon(watch)
            await settle()
            await events.send(member(PEER, "Suspect", "Disconnected"))
        assert results == [PEER]

    fake = FakeReachability(ME)
    assert not fake.is_reachable(PEER), "never heard of, as with the real one"
    fake.reach(PEER)
    assert fake.is_reachable(PEER)
    fake_results: list[NodeId] = []
    async with create_task_group() as tg:

        async def watch_fake() -> None:
            fake_results.append(await wait_for_unreachable(fake, PEER))

        tg.start_soon(watch_fake)
        await settle()
        await fake.lose(PEER)
    assert fake_results == [PEER]
    assert fake.is_reachable(ME)
    fake.reach(PEER)
    assert fake.is_reachable(PEER)


# ---- The timers and the quorum read, for exo's own code ----


def test_exo_reads_the_membership_timers_from_their_one_table() -> None:
    assert timedelta(seconds=30) == GRACE
    assert timedelta(seconds=30) == WEDGE_TIMEOUT
    assert timedelta(minutes=5) == SPLIT_HOLD


async def test_membership_gives_the_quorum_read_now() -> None:
    own = exo_rs.Member("ME", exo_rs.MemberState.Alive, None, SINCE)
    snapshot = snapshot_of([own], Quorum.Half, 1, 2)
    async with running_membership(snapshot) as (membership, _):
        assert membership.quorum() == "Half"


# ---- What the cluster's events tell the networking layer ----


async def test_a_node_the_coordinator_removed_stops_counting_in_the_quorum() -> None:
    """A node that restarts comes back under a new id: its old one must stop counting, or a
    healthy pair reads Half, then Minority. The coordinator removes it (NodeTimedOut, which is
    also how a quitting node's departure arrives), and every node then forgets it."""
    networking = FakeNetworking()
    members, member_receiver = channel[Member]()
    events, event_receiver = channel[IndexedEvent | StateSnapshot]()
    membership = Membership(
        ME,
        member_receiver=member_receiver,
        networking=networking,
        cluster_events=event_receiver,
    )
    async with create_task_group() as tg:
        tg.start_soon(membership.run)
        await events.send(IndexedEvent(idx=0, event=NodeTimedOut(node_id=PEER)))
        await events.send(
            IndexedEvent(
                idx=1, event=InstanceDeleted(instance_id=InstanceId("an-instance"))
            )
        )
        await events.send(
            StateSnapshot(
                session=SessionId(master_node_id=OTHER, election_clock=1),
                requester=SystemId(),
                state=State(),
            )
        )
        await events.send(IndexedEvent(idx=2, event=NodeTimedOut(node_id=OTHER)))
        await settle()
        members.close()
        events.close()
    assert networking.forgotten == [PEER, OTHER]


async def told_by_the_cluster(
    *items: IndexedEvent | StateSnapshot,
) -> FakeNetworking:
    """What a running Membership tells the networking layer for these cluster events"""
    networking = FakeNetworking()
    members, member_receiver = channel[Member]()
    events, event_receiver = channel[IndexedEvent | StateSnapshot]()
    membership = Membership(
        ME,
        member_receiver=member_receiver,
        networking=networking,
        cluster_events=event_receiver,
    )
    async with create_task_group() as tg:
        tg.start_soon(membership.run)
        for item in items:
            await events.send(item)
        await settle()
        members.close()
        events.close()
    return networking


async def test_a_report_shows_that_exo_on_the_node_runs_and_that_the_cluster_lists_it() -> (
    None
):
    report = NodeGatheredInfo(
        node_id=PEER,
        when=str(datetime.now(tz=UTC)),
        info=MiscData(friendly_name="a Mac"),
    )
    networking = await told_by_the_cluster(
        IndexedEvent(idx=0, event=report),
        IndexedEvent(idx=1, event=InstanceDeleted(instance_id=InstanceId("one"))),
    )
    assert (networking.heard, networking.listed_nodes) == ([PEER], [PEER])
    assert networking.forgotten == []


async def test_a_snapshot_lists_every_node_in_the_cluster_state() -> None:
    topology = Topology()
    topology.add_node(PEER)
    state = State(
        topology=topology, last_seen={OTHER: datetime.fromtimestamp(SINCE, tz=UTC)}
    )
    networking = await told_by_the_cluster(
        StateSnapshot(
            session=SessionId(master_node_id=OTHER, election_clock=1),
            requester=SystemId(),
            state=state,
        )
    )
    assert sorted(networking.listed_nodes) == sorted([OTHER, PEER])
    assert networking.heard == []


async def test_a_change_says_why_the_node_wasnt_alive_before() -> None:
    async with running_membership() as (membership, events):
        changes = membership.changes()
        await events.send(member(PEER, "Alive"))
        await events.send(member(PEER, "Suspect", "Wedged"))
        await events.send(member(PEER, "Alive"))
        await settle()
        reasons = [
            (change.reason, change.previous_reason) for change in changes.collect()
        ]
        assert reasons == [(None, None), ("Wedged", None), (None, "Wedged")]
