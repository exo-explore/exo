"""Peer state provider for discovering which peers have which models.

Reads from the shared State object (populated via gossipsub events) to
determine which peer nodes have completed or are in the process of
downloading a given model. Resolves peer IP addresses from the topology.
"""

from dataclasses import dataclass
from typing import Callable, Literal

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.topology import SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
)


@dataclass(frozen=True)
class PeerInfo:
    """A peer that has (or is downloading) a model."""

    node_id: NodeId
    ip: str
    status: Literal["complete", "ongoing"]


class PeerStateProvider:
    """Provides information about which peers have which models.

    Reads from the Worker's shared State to find peers and resolve their
    network addresses from the topology graph.
    """

    def __init__(
        self,
        node_id: NodeId,
        state_accessor: Callable[[], State],
        peer_download_port: int,
    ) -> None:
        self.node_id = node_id
        self._state_accessor = state_accessor
        self.peer_download_port = peer_download_port

    def get_peers_for_model(self, model_id: str) -> list[PeerInfo]:
        """Find peers that have a specific model (complete or in-progress).

        Returns peers sorted by completeness (completed first, then ongoing).
        Excludes self.
        """
        state = self._state_accessor()
        peers: list[PeerInfo] = []

        # Check download status across all nodes
        for peer_node_id, download_list in state.downloads.items():
            if peer_node_id == self.node_id:
                continue

            for dl in download_list:
                dl_model_id = dl.shard_metadata.model_card.model_id
                if dl_model_id.normalize() != model_id:
                    continue

                if isinstance(dl, DownloadCompleted):
                    status: Literal["complete", "ongoing"] = "complete"
                elif isinstance(dl, DownloadOngoing):
                    status = "ongoing"
                else:
                    continue

                # Resolve IP from topology
                ip = self._resolve_peer_ip(peer_node_id, state)
                if ip:
                    peers.append(PeerInfo(node_id=peer_node_id, ip=ip, status=status))

        # Sort: completed peers first
        peers.sort(key=lambda p: 0 if p.status == "complete" else 1)
        return peers

    def _resolve_peer_ip(self, peer_node_id: NodeId, state: State) -> str | None:
        """Resolve a peer's IP address from the topology graph."""
        try:
            for conn in state.topology.out_edges(self.node_id):
                if conn.sink == peer_node_id and isinstance(
                    conn.edge, SocketConnection
                ):
                    return conn.edge.sink_multiaddr.ip_address
        except Exception as e:
            logger.debug(f"Could not resolve IP for peer {peer_node_id}: {e}")
        return None
