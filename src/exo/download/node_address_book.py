from __future__ import annotations

from dataclasses import dataclass, field

from exo.shared.types.common import NodeId


@dataclass
class NodeAddressBook:
    """
    Tracks the most recently observed IPv4 address for each node.

    This is populated from libp2p connection updates (via :class:`ConnectionMessage`)
    and is used for local-network services (e.g. model distribution).
    """

    ipv4_by_node_id: dict[NodeId, str] = field(default_factory=dict)

    def set_ipv4(self, node_id: NodeId, ipv4: str) -> None:
        self.ipv4_by_node_id[node_id] = ipv4

    def remove(self, node_id: NodeId) -> None:
        self.ipv4_by_node_id.pop(node_id, None)

    def get_ipv4(self, node_id: NodeId) -> str | None:
        return self.ipv4_by_node_id.get(node_id)

