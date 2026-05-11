"""Regression tests for ``exo.download.peer_state``.

These exercise the topology-iteration ordering that decides whether a peer
is reachable over RDMA or merely via socket. The original implementation
returned on the first edge whose type happened to be visited first, which
mislabelled peers when ``out_edges`` yielded the socket edge before the
RDMA edge. We now scan all edges and prefer RDMA whenever any RDMA edge
exists for that peer.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import cast

from exo.download.peer_state import discover_peers_for_model
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
)
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

LOCAL = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
PEER = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()


def _make_shard() -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=MODEL_ID,
            storage_size=Memory.from_mb(100),
            n_layers=4,
            hidden_size=64,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )


def _build_topology(edges: Iterable[Connection]) -> Topology:
    topology = Topology()
    topology.add_node(LOCAL)
    topology.add_node(PEER)
    for conn in edges:
        topology.add_connection(conn)
    return topology


def _state_with_completed_peer(topology: Topology) -> State:
    completed = DownloadCompleted(
        node_id=PEER,
        shard_metadata=_make_shard(),
        total=Memory.from_mb(100),
        model_directory=str(Path("/fake/models/test-org--test-model")),
    )
    return State(
        downloads={PEER: [cast(DownloadProgress, completed)]},
        topology=topology,
    )


def _socket_edge() -> Connection:
    return Connection(
        source=LOCAL,
        sink=PEER,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/4001")
        ),
    )


def _rdma_edge() -> Connection:
    return Connection(
        source=LOCAL,
        sink=PEER,
        edge=RDMAConnection(source_rdma_iface="bridge0", sink_rdma_iface="bridge0"),
    )


def test_peer_marked_rdma_when_socket_edge_inserted_first() -> None:
    """If both an RDMA edge and a socket edge exist for the same peer, the
    peer must be reported as RDMA *regardless of insertion order*. The
    original implementation returned on the first edge it saw, so a socket
    edge inserted before the RDMA edge silently downgraded a real RDMA peer
    to ``socket`` and broke the "RDMA first" ordering used by the peer
    downloader.
    """
    topology = _build_topology([_socket_edge(), _rdma_edge()])
    state = _state_with_completed_peer(topology)

    peers = discover_peers_for_model(LOCAL, state, NORMALIZED, peer_download_port=52416)

    assert len(peers) == 1
    assert peers[0].connection_type == "rdma"
    assert peers[0].ip == "10.0.0.2"


def test_peer_marked_rdma_when_rdma_edge_inserted_first() -> None:
    topology = _build_topology([_rdma_edge(), _socket_edge()])
    state = _state_with_completed_peer(topology)

    peers = discover_peers_for_model(LOCAL, state, NORMALIZED, peer_download_port=52416)

    assert len(peers) == 1
    assert peers[0].connection_type == "rdma"


def test_peer_marked_socket_when_no_rdma_edge_exists() -> None:
    topology = _build_topology([_socket_edge()])
    state = _state_with_completed_peer(topology)

    peers = discover_peers_for_model(LOCAL, state, NORMALIZED, peer_download_port=52416)

    assert len(peers) == 1
    assert peers[0].connection_type == "socket"
    assert peers[0].ip == "10.0.0.2"


def test_peer_skipped_when_only_rdma_edge_has_no_socket_companion() -> None:
    """An RDMA-only peer cannot be contacted over the peer-download HTTP
    server, so we must omit it rather than fabricate a missing IP.
    """
    topology = _build_topology([_rdma_edge()])
    state = _state_with_completed_peer(topology)

    peers = discover_peers_for_model(LOCAL, state, NORMALIZED, peer_download_port=52416)

    assert peers == []
