from typing import final

from exo.routing.membership import Reachability
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel


@final
class FakeReachability(Reachability):
    """A `Reachability` for tests of the code that reads it: nodes are reachable until a test
    says otherwise"""

    def __init__(self, node_id: NodeId):
        self.node_id = node_id
        self._unreachable: set[NodeId] = set()
        self._watchers: list[Sender[NodeId]] = []

    def is_reachable(self, node: NodeId) -> bool:
        return node == self.node_id or node not in self._unreachable

    def watch(self) -> Receiver[NodeId]:
        sender, receiver = channel[NodeId]()
        self._watchers.append(sender)
        return receiver

    async def lose(self, node: NodeId) -> None:
        """The node leaves Alive"""
        if node in self._unreachable:
            return
        self._unreachable.add(node)
        for watcher in self._watchers:
            await watcher.send(node)

    def regain(self, node: NodeId) -> None:
        """The node is Alive again"""
        self._unreachable.discard(node)
