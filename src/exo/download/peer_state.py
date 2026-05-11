"""Pure functions for discovering which peers have which models.

These functions are called by the Worker (which owns the State) to compute
peer availability at command-emit time. The results are embedded in the
StartDownload command so the download coordinator stays decoupled from
Worker state.
"""

from loguru import logger

from exo.shared.types.commands import PeerEndpoint
from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.topology import RDMAConnection, SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
)


def discover_peers_for_model(
    node_id: NodeId,
    state: State,
    model_id_normalized: str,
    peer_download_port: int,
) -> list[PeerEndpoint]:
    """Find peers that have a specific model (complete or in-progress).

    Called by the Worker when emitting a StartDownload command. Returns
    peers sorted by priority: RDMA/Thunderbolt connections first, then
    completed downloads before ongoing ones.

    Args:
        node_id: This node's ID (excluded from results).
        state: The global State object (owned by Worker).
        model_id_normalized: Normalized model ID (e.g. "org--model").
        peer_download_port: Port where peers run their PeerFileServer.

    Returns:
        List of PeerEndpoint sorted by connection quality and completeness.
    """
    peers: list[PeerEndpoint] = []

    for peer_node_id, download_list in state.downloads.items():
        if peer_node_id == node_id:
            continue

        for dl in download_list:
            dl_model_id = dl.shard_metadata.model_card.model_id
            if dl_model_id.normalize() != model_id_normalized:
                continue

            if isinstance(dl, DownloadCompleted):
                status = "complete"
            elif isinstance(dl, DownloadOngoing):
                status = "ongoing"
            else:
                continue

            # Resolve IP and connection type from topology
            endpoint = _resolve_peer_endpoint(
                node_id, peer_node_id, state, peer_download_port, status
            )
            if endpoint:
                peers.append(endpoint)

    # Sort by priority:
    # 1. RDMA/Thunderbolt connections first (lower latency, higher bandwidth)
    # 2. Completed downloads before ongoing ones
    peers.sort(
        key=lambda p: (
            0 if p.connection_type == "rdma" else 1,
            0 if p.status == "complete" else 1,
        )
    )
    return peers


def _resolve_peer_endpoint(
    node_id: NodeId,
    peer_node_id: NodeId,
    state: State,
    peer_download_port: int,
    status: str,
) -> PeerEndpoint | None:
    """Resolve a peer's IP address and connection type from the topology.

    Iteration order over ``out_edges`` is not guaranteed to surface RDMA
    edges before socket edges, so we scan the full edge set once: any
    RDMA edge wins (we use the peer's socket address for the actual TCP
    connect since RDMA edges don't carry routable IPs), and only when no
    RDMA edge exists do we fall back to the socket endpoint. Returning
    on the first non-RDMA hit would otherwise mislabel peers as
    ``socket`` whenever the socket edge happens to be visited first.
    """
    try:
        edges = [
            conn
            for conn in state.topology.out_edges(node_id)
            if conn.sink == peer_node_id
        ]
        has_rdma = any(isinstance(conn.edge, RDMAConnection) for conn in edges)
        socket_ip = next(
            (
                conn.edge.sink_multiaddr.ip_address
                for conn in edges
                if isinstance(conn.edge, SocketConnection)
            ),
            None,
        )
        if has_rdma and socket_ip:
            return PeerEndpoint(
                node_id=peer_node_id,
                ip=socket_ip,
                port=peer_download_port,
                status=status,
                connection_type="rdma",
            )
        if socket_ip:
            return PeerEndpoint(
                node_id=peer_node_id,
                ip=socket_ip,
                port=peer_download_port,
                status=status,
                connection_type="socket",
            )
    except Exception as e:
        logger.debug(f"Could not resolve endpoint for peer {peer_node_id}: {e}")
    return None
