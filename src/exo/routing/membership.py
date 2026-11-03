"""This node's view of which nodes are in the cluster.

The networking layer owns membership: from zenoh's liveliness tokens, its connections and a
probe of nodes whose connection closed, it decides whether each node is Alive, Suspect or
Gone (with the reason), and it owns every liveness timer (see rust/networking/src/membership.rs).
This module keeps the latest state it reported for the rest of the node to read locally. It
isn't part of the replicated state: two nodes may disagree for a moment.

It also tells the networking layer what only exo knows, from the cluster's events as this node
applies them: that the coordinator removed a node (NodeTimedOut, which is also how a quitting
node's departure arrives), so that the quorum read stops counting it.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Final, Literal, Protocol, final

import exo_rs
from anyio import BrokenResourceError, ClosedResourceError, create_task_group
from loguru import logger

from exo.routing.event_router import NodeEventReceiver
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent, NodeTimedOut
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import FrozenModel

MemberState = Literal["Alive", "Suspect", "Gone", "Resuming"]
"""Suspect: unreachable now, and Gone if it stays so for the grace period. Resuming is only ever
this node's own state, for a while after it was stopped (frozen, or the machine asleep)."""

LossReason = Literal["Disconnected", "Left", "Wedged"]
"""Why a node isn't Alive: its transport was lost, it said it was leaving, or its transport is
up but exo on it stopped reporting."""

Quorum = Literal["Majority", "Half", "Minority"]
"""How many of the nodes it knows this node reaches: more than half, exactly half (two nodes with
one away, or any even split), or less than half."""

# The membership's timers that exo acts on, from the one table that owns them
# (rust/networking/src/membership.rs, where each is derived)
GRACE: Final = timedelta(seconds=exo_rs.GRACE_SECONDS)
"""How long a node is Suspect before it is Gone"""
WEDGE_TIMEOUT: Final = timedelta(seconds=exo_rs.WEDGE_TIMEOUT_SECONDS)
"""How long a node whose transport is up may go without a report before it is Suspect(Wedged)"""
SPLIT_HOLD: Final = timedelta(seconds=exo_rs.SPLIT_HOLD_SECONDS)
"""How long the side of a split without a majority holds before it takes over the lost nodes'
work"""
STALL_THRESHOLD: Final = timedelta(seconds=exo_rs.STALL_THRESHOLD_SECONDS)
"""How late a 1 s tick must be to mean this node itself was stopped (frozen, starved or asleep)"""


class Member(FrozenModel):
    """A node and the state it has been in since `since`, as this node sees it. Each change the
    networking layer reports is one of these, on a local topic (never sent to other nodes)."""

    node_id: NodeId
    state: MemberState
    reason: LossReason | None
    since: datetime

    @classmethod
    def from_swarm(cls, update: exo_rs.FromSwarm.Membership) -> "Member":
        return cls(
            node_id=NodeId(update.node),
            state=_state(update.state),
            reason=_reason(update.reason),
            since=_datetime(update.since),
        )


class MemberView(Member):
    """A node in this node's view, and whether it counts in the quorum read: one Gone for a while
    (5 min), or that left, doesn't, though it stays in the view until the coordinator removes it"""

    counted: bool

    @classmethod
    def from_snapshot(cls, member: exo_rs.Member) -> "MemberView":
        return cls(
            node_id=NodeId(member.node),
            state=_state(member.state),
            reason=_reason(member.reason),
            since=_datetime(member.since),
            counted=member.counted,
        )


class MembershipChange(FrozenModel):
    """A node moving from one state to another, as this node sees it"""

    node_id: NodeId
    # None when this node hadn't heard of it before
    previous: MemberState | None
    current: MemberState
    reason: LossReason | None

    @property
    def became_alive(self) -> bool:
        """It joined, rejoined, or is back from being unreachable (not this node resuming)"""
        return self.current == "Alive" and self.previous in (None, "Suspect", "Gone")

    @property
    def left_alive(self) -> bool:
        """It became unreachable, or is gone without having been suspect first (it left)"""
        return self.previous == "Alive" and self.current in ("Suspect", "Gone")


class MembershipView(FrozenModel):
    """This node's view of the cluster, as GET /membership answers it"""

    node_id: NodeId
    self_state: MemberState
    quorum: Quorum
    # The Alive peers, plus this node unless it is Resuming
    reachable: int
    # This node and every peer it counts
    last_known: int
    # This node first, then its peers
    members: Sequence[MemberView]

    @classmethod
    def from_snapshot(
        cls, node_id: NodeId, snapshot: exo_rs.MembershipSnapshot
    ) -> "MembershipView":
        members = [MemberView.from_snapshot(member) for member in snapshot.members]
        own = next((member for member in members if member.node_id == node_id), None)
        return cls(
            node_id=node_id,
            self_state=own.state if own is not None else "Alive",
            quorum=_quorum(snapshot.quorum),
            reachable=snapshot.reachable,
            last_known=snapshot.last_known,
            members=members,
        )


class Reachability(Protocol):
    """Whether this node can reach another one now, as its membership sees it"""

    def is_reachable(self, node: NodeId) -> bool:
        """Whether `node` is Alive, or is this node"""
        ...

    def watch(self) -> Receiver[NodeId]:
        """Every node that leaves Alive (for Suspect or Gone), as it happens"""
        ...


class NetworkingMembership(Protocol):
    """The networking layer's membership, as exo reads it and tells it what only exo knows (the
    router, over exo_rs.NetworkingHandle)"""

    def members(self) -> exo_rs.MembershipSnapshot: ...

    def forget(self, node: NodeId) -> None:
        """The coordinator removed the node: it stops counting in the quorum (unless this node
        still reaches it)"""
        ...


@final
class Membership(Reachability):
    """The latest state of each node, as this node's networking reported it"""

    def __init__(
        self,
        node_id: NodeId,
        *,
        member_receiver: Receiver[Member],
        networking: NetworkingMembership,
        cluster_events: NodeEventReceiver,
    ):
        self.node_id = node_id
        self._member_receiver = member_receiver
        self._networking = networking
        self._cluster_events = cluster_events
        self._latest: dict[NodeId, Member] = {}
        self._watchers: list[Sender[NodeId]] = []
        self._change_senders: list[Sender[MembershipChange]] = []

    async def run(self) -> None:
        try:
            async with create_task_group() as tg:
                tg.start_soon(self._follow_cluster)
                with self._member_receiver as members:
                    async for member in members:
                        await self._update(member)
                tg.cancel_scope.cancel()
        finally:
            # Those reading what it hands out see their stream end with it
            for sender in [*self._watchers, *self._change_senders]:
                sender.close()

    def is_reachable(self, node: NodeId) -> bool:
        if node == self.node_id:
            return True
        latest = self._latest.get(node)
        return latest is not None and latest.state == "Alive"

    def watch(self) -> Receiver[NodeId]:
        sender, receiver = channel[NodeId]()
        self._watchers.append(sender)
        return receiver

    def changes(self) -> Receiver[MembershipChange]:
        """Every change of a node's state, with the state before it"""
        sender, receiver = channel[MembershipChange]()
        self._change_senders.append(sender)
        return receiver

    def state_of(self, node: NodeId) -> Member | None:
        """The node's latest state; None if this node never heard of it"""
        return self._latest.get(node)

    def view(self) -> MembershipView:
        """Every node this node knows of, and the quorum read, from the networking layer now"""
        return MembershipView.from_snapshot(self.node_id, self._networking.members())

    def quorum(self) -> Quorum:
        """How much of the cluster this node reaches now"""
        return self.view().quorum

    async def _follow_cluster(self) -> None:
        """Tells the networking layer which nodes the coordinator removed, as this node applies
        the cluster's events. Every node does, so every node's quorum read leaves them out: a
        node that restarts comes back under a new id, and its old one must stop counting."""
        with self._cluster_events as events:
            async for item in events:
                match item:
                    case IndexedEvent(event=NodeTimedOut(node_id=node_id)):
                        self._networking.forget(node_id)
                    case _:
                        pass

    async def _update(self, member: Member) -> None:
        previous = self._latest.get(member.node_id)
        self._latest[member.node_id] = member
        change = MembershipChange(
            node_id=member.node_id,
            previous=previous.state if previous is not None else None,
            current=member.state,
            reason=member.reason,
        )
        logger.debug(f"Membership: {change}")
        if change.left_alive:
            self._watchers = await _send_to_all(self._watchers, member.node_id)
        self._change_senders = await _send_to_all(self._change_senders, change)


async def _send_to_all[T](senders: list[Sender[T]], item: T) -> list[Sender[T]]:
    """Sends `item` to each sender, and returns those whose receiver is still open"""
    open_senders: list[Sender[T]] = []
    for sender in senders:
        try:
            await sender.send(item)
        except (ClosedResourceError, BrokenResourceError):
            continue
        open_senders.append(sender)
    return open_senders


def _state(state: exo_rs.MemberState) -> MemberState:
    match state:
        case exo_rs.MemberState.Alive:
            return "Alive"
        case exo_rs.MemberState.Suspect:
            return "Suspect"
        case exo_rs.MemberState.Gone:
            return "Gone"
        case exo_rs.MemberState.Resuming:
            return "Resuming"


def _reason(reason: exo_rs.LossReason | None) -> LossReason | None:
    match reason:
        case None:
            return None
        case exo_rs.LossReason.Disconnected:
            return "Disconnected"
        case exo_rs.LossReason.Left:
            return "Left"
        case exo_rs.LossReason.Wedged:
            return "Wedged"


def _quorum(quorum: exo_rs.Quorum) -> Quorum:
    match quorum:
        case exo_rs.Quorum.Majority:
            return "Majority"
        case exo_rs.Quorum.Half:
            return "Half"
        case exo_rs.Quorum.Minority:
            return "Minority"


def _datetime(seconds_since_epoch: float) -> datetime:
    return datetime.fromtimestamp(seconds_since_epoch, tz=UTC)
