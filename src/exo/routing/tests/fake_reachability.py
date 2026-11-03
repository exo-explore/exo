from typing import final

from exo.routing.membership import Reachability
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel


@final
class FakeReachability(Reachability):
    """A `Reachability` for tests of the code that reads it. As with the real one, a node is
    reachable only while a test says so (and this node always is)."""

    def __init__(self, node_id: NodeId, reachable: set[NodeId] | None = None):
        self.node_id = node_id
        self._reachable: set[NodeId] = set(reachable or ())
        self._watchers: list[Sender[NodeId]] = []

    def is_reachable(self, node: NodeId) -> bool:
        return node == self.node_id or node in self._reachable

    def watch(self) -> Receiver[NodeId]:
        sender, receiver = channel[NodeId]()
        self._watchers.append(sender)
        return receiver

    def reach(self, node: NodeId) -> None:
        """The node is Alive"""
        self._reachable.add(node)

    async def lose(self, node: NodeId) -> None:
        """The node leaves Alive"""
        if node not in self._reachable:
            return
        self._reachable.discard(node)
        for watcher in self._watchers:
            await watcher.send(node)
