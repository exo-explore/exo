from datetime import UTC, datetime
from typing import final

from exo.routing.membership import (
    LossReason,
    MembershipReader,
    MembershipView,
    MemberState,
    MemberView,
    Quorum,
)
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel


@final
class FakeMembership(MembershipReader):
    """A membership for tests of the code that reads it. The nodes it is given are Alive until a
    test says otherwise; like the real one, it can't reach a node it was never told of. The
    quorum read is what the test sets."""

    def __init__(
        self, node_id: NodeId, *nodes: NodeId, quorum: Quorum = "Majority"
    ) -> None:
        self.node_id = node_id
        self.quorum: Quorum = quorum
        self.self_state: MemberState = "Alive"
        self._members: dict[NodeId, MemberView] = {}
        self._watchers: list[Sender[NodeId]] = []
        for node in nodes:
            self.set(node, "Alive")

    def set(
        self, node: NodeId, state: MemberState, reason: LossReason | None = None
    ) -> None:
        """The node's state, without telling watchers"""
        self._members[node] = MemberView(
            node_id=node,
            state=state,
            reason=reason,
            since=datetime.now(tz=UTC),
            counted=True,
        )

    async def lose(
        self,
        node: NodeId,
        state: MemberState = "Suspect",
        reason: LossReason = "Disconnected",
    ) -> None:
        """The node leaves Alive, and watchers hear of it"""
        was_reachable = self.is_reachable(node)
        self.set(node, state, reason)
        if was_reachable:
            for watcher in self._watchers:
                await watcher.send(node)

    def regain(self, node: NodeId) -> None:
        """The node is Alive again"""
        self.set(node, "Alive")

    def is_reachable(self, node: NodeId) -> bool:
        member = self._members.get(node)
        return node == self.node_id or (member is not None and member.state == "Alive")

    def watch(self) -> Receiver[NodeId]:
        sender, receiver = channel[NodeId]()
        self._watchers.append(sender)
        return receiver

    def view(self) -> MembershipView:
        peers = [
            member
            for member in self._members.values()
            if member.node_id != self.node_id
        ]
        return MembershipView(
            node_id=self.node_id,
            self_state=self.self_state,
            quorum=self.quorum,
            reachable=1 + sum(1 for member in peers if member.state == "Alive"),
            last_known=1 + len(peers),
            members=[
                MemberView(
                    node_id=self.node_id,
                    state="Alive",
                    reason=None,
                    since=datetime.now(tz=UTC),
                    counted=True,
                ),
                *peers,
            ],
        )
