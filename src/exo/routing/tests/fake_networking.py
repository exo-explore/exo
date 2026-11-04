from collections.abc import Callable
from typing import final

from exo_rs import MembershipSnapshot

from exo.routing.membership import NetworkingMembership
from exo.shared.types.common import NodeId


def no_snapshot() -> MembershipSnapshot:
    raise AssertionError("not asked for in this test")


@final
class FakeNetworking(NetworkingMembership):
    """The networking layer's membership, for tests of the Python side: it answers with the
    given snapshot and records what it is told"""

    def __init__(
        self, snapshot: Callable[[], MembershipSnapshot] = no_snapshot
    ) -> None:
        self._snapshot = snapshot
        self.forgotten: list[NodeId] = []

    def members(self) -> MembershipSnapshot:
        return self._snapshot()

    def forget(self, node: NodeId) -> None:
        self.forgotten.append(node)
