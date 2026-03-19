"""
Tests for topology-mutating apply functions.
Verifies COW: original state topology must be unchanged after apply.
"""
from exo.shared.apply import apply_node_timed_out, apply_topology_edge_created, apply_topology_edge_deleted
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut, TopologyEdgeCreated, TopologyEdgeDeleted
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection


def _conn(src: str, dst: str) -> Connection:
    return Connection(
        source=NodeId(src),
        sink=NodeId(dst),
        edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/52415")),
    )


def _state_with_nodes(*node_ids: str) -> State:
    topo = Topology()
    for nid in node_ids:
        topo.add_node(NodeId(nid))
    return State(topology=topo)


def test_apply_node_timed_out_removes_node():
    state = _state_with_nodes("a", "b")
    new_state = apply_node_timed_out(NodeTimedOut(node_id=NodeId("a")), state)
    assert NodeId("a") not in list(new_state.topology.list_nodes())
    assert NodeId("b") in list(new_state.topology.list_nodes())


def test_apply_node_timed_out_cow():
    """Original state topology is unchanged after apply (COW)."""
    state = _state_with_nodes("a", "b")
    apply_node_timed_out(NodeTimedOut(node_id=NodeId("a")), state)
    # original must still have both nodes
    assert NodeId("a") in list(state.topology.list_nodes())
    assert NodeId("b") in list(state.topology.list_nodes())


def test_apply_topology_edge_created():
    state = _state_with_nodes("x", "y")
    conn = _conn("x", "y")
    new_state = apply_topology_edge_created(TopologyEdgeCreated(conn=conn), state)
    edges = list(new_state.topology.list_connections())
    assert any(e.source == NodeId("x") and e.sink == NodeId("y") for e in edges)


def test_apply_topology_edge_created_cow():
    state = _state_with_nodes("x", "y")
    conn = _conn("x", "y")
    apply_topology_edge_created(TopologyEdgeCreated(conn=conn), state)
    # original topology must have no edges
    assert list(state.topology.list_connections()) == []


def test_apply_topology_edge_deleted():
    state = _state_with_nodes("p", "q")
    conn = _conn("p", "q")
    state_with_edge = apply_topology_edge_created(TopologyEdgeCreated(conn=conn), state)
    state_after_delete = apply_topology_edge_deleted(TopologyEdgeDeleted(conn=conn), state_with_edge)
    assert list(state_after_delete.topology.list_connections()) == []
