from datetime import datetime, timezone

from exo.shared.apply import apply_node_gathered_info
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo
from exo.shared.types.profiling import (
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
)
from exo.shared.types.state import State
from exo.shared.types.thunderbolt import ThunderboltConnection, ThunderboltIdentifier
from exo.shared.types.topology import RDMAConnection
from exo.utils.info_gatherer.info_gatherer import (
    MacThunderboltConnections,
    RdmaCtlStatus,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_state_with_thunderbolt_idents(
    *node_ids_and_uuids: tuple[NodeId, str, str],
    rdma_ctl: dict[NodeId, NodeRdmaCtlStatus] | None = None,
) -> State:
    """Build a State with Thunderbolt identifiers per node so the apply MacThunderboltConnections
    case can resolve uuid -> (node, iface)."""
    node_thunderbolt = {
        nid: NodeThunderboltInfo(
            interfaces=[ThunderboltIdentifier(rdma_interface=iface, domain_uuid=uuid)]
        )
        for nid, uuid, iface in node_ids_and_uuids
    }
    return State(
        node_thunderbolt=node_thunderbolt,
        node_rdma_ctl=rdma_ctl or {},
    )


def _has_rdma_edge(topology: Topology, source: NodeId, sink: NodeId) -> bool:
    return any(
        isinstance(edge, RDMAConnection)
        for edge in topology.get_all_connections_between(source, sink)
    )


def test_mac_thunderbolt_connections_emits_rdma_when_both_endpoints_enabled():
    node_a = NodeId()
    node_b = NodeId()
    state = _make_state_with_thunderbolt_idents(
        (node_a, "uuid-a", "rdma_en1"),
        (node_b, "uuid-b", "rdma_en1"),
        rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    event = NodeGatheredInfo(
        node_id=node_a,
        when=_now(),
        info=MacThunderboltConnections(
            conns=[ThunderboltConnection(source_uuid="uuid-a", sink_uuid="uuid-b")]
        ),
    )

    new_state = apply_node_gathered_info(event, state)

    assert _has_rdma_edge(new_state.topology, node_a, node_b)


def test_mac_thunderbolt_connections_skips_rdma_when_source_rdma_ctl_disabled():
    node_a = NodeId()
    node_b = NodeId()
    state = _make_state_with_thunderbolt_idents(
        (node_a, "uuid-a", "rdma_en1"),
        (node_b, "uuid-b", "rdma_en1"),
        rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=False),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    event = NodeGatheredInfo(
        node_id=node_a,
        when=_now(),
        info=MacThunderboltConnections(
            conns=[ThunderboltConnection(source_uuid="uuid-a", sink_uuid="uuid-b")]
        ),
    )

    new_state = apply_node_gathered_info(event, state)

    assert not _has_rdma_edge(new_state.topology, node_a, node_b)


def test_mac_thunderbolt_connections_skips_rdma_when_sink_rdma_ctl_disabled():
    node_a = NodeId()
    node_b = NodeId()
    state = _make_state_with_thunderbolt_idents(
        (node_a, "uuid-a", "rdma_en1"),
        (node_b, "uuid-b", "rdma_en1"),
        rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=False),
        },
    )

    event = NodeGatheredInfo(
        node_id=node_a,
        when=_now(),
        info=MacThunderboltConnections(
            conns=[ThunderboltConnection(source_uuid="uuid-a", sink_uuid="uuid-b")]
        ),
    )

    new_state = apply_node_gathered_info(event, state)

    assert not _has_rdma_edge(new_state.topology, node_a, node_b)


def test_mac_thunderbolt_connections_skips_rdma_when_rdma_ctl_status_missing():
    """Missing rdma_ctl status defaults to not-enabled — node is RDMA-incapable."""
    node_a = NodeId()
    node_b = NodeId()
    state = _make_state_with_thunderbolt_idents(
        (node_a, "uuid-a", "rdma_en1"),
        (node_b, "uuid-b", "rdma_en1"),
        rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            # node_b intentionally absent
        },
    )

    event = NodeGatheredInfo(
        node_id=node_a,
        when=_now(),
        info=MacThunderboltConnections(
            conns=[ThunderboltConnection(source_uuid="uuid-a", sink_uuid="uuid-b")]
        ),
    )

    new_state = apply_node_gathered_info(event, state)

    assert not _has_rdma_edge(new_state.topology, node_a, node_b)


def test_rdma_ctl_status_disabled_purges_existing_rdma_edges():
    """When a node reports rdma_ctl disabled, all RDMA edges touching it must be removed."""
    node_a = NodeId()
    node_b = NodeId()

    # Start with both nodes RDMA-enabled and existing RDMA edges in the topology.
    state = _make_state_with_thunderbolt_idents(
        (node_a, "uuid-a", "rdma_en1"),
        (node_b, "uuid-b", "rdma_en1"),
        rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_a,
            when=_now(),
            info=MacThunderboltConnections(
                conns=[ThunderboltConnection(source_uuid="uuid-a", sink_uuid="uuid-b")]
            ),
        ),
        state,
    )
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_b,
            when=_now(),
            info=MacThunderboltConnections(
                conns=[ThunderboltConnection(source_uuid="uuid-b", sink_uuid="uuid-a")]
            ),
        ),
        state,
    )
    assert _has_rdma_edge(state.topology, node_a, node_b)
    assert _has_rdma_edge(state.topology, node_b, node_a)

    # Now node_a flips to rdma_ctl disabled — both directions of RDMA edge must drop.
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_a, when=_now(), info=RdmaCtlStatus(enabled=False)
        ),
        state,
    )

    assert not _has_rdma_edge(state.topology, node_a, node_b)
    assert not _has_rdma_edge(state.topology, node_b, node_a)
    assert state.node_rdma_ctl[node_a].enabled is False


def test_topology_remove_all_rdma_connections_touching_keeps_socket_edges():
    """Purging RDMA edges for a disabled node must not affect non-RDMA edges."""
    from exo.shared.types.multiaddr import Multiaddr
    from exo.shared.types.topology import Connection, SocketConnection

    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=RDMAConnection(
                source_rdma_iface="rdma_en1", sink_rdma_iface="rdma_en1"
            ),
        )
    )
    socket_edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )
    topology.add_connection(Connection(source=node_a, sink=node_b, edge=socket_edge))

    topology.remove_all_rdma_connections_touching(node_a)

    assert not _has_rdma_edge(topology, node_a, node_b)
    # Socket edge survives.
    assert any(
        isinstance(edge, SocketConnection)
        for edge in topology.get_all_connections_between(node_a, node_b)
    )
