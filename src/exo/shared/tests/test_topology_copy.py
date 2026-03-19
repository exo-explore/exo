from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.topology import Connection, SocketConnection


def _socket_conn(src: str, dst: str) -> Connection:
    edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/52415"),
    )
    return Connection(source=NodeId(src), sink=NodeId(dst), edge=edge)


def test_copy_is_independent():
    """Mutating a copy must not affect the original."""
    t = Topology()
    t.add_node(NodeId("a"))
    t.add_node(NodeId("b"))
    t.add_connection(_socket_conn("a", "b"))

    copy = t.copy()
    copy.add_node(NodeId("c"))
    copy.remove_node(NodeId("a"))

    # original unchanged
    assert NodeId("a") in list(t.list_nodes())
    assert NodeId("c") not in list(t.list_nodes())
    # copy has the mutations
    assert NodeId("c") in list(copy.list_nodes())
    assert NodeId("a") not in list(copy.list_nodes())


def test_copy_preserves_edges():
    """Copied topology must have the same edges as the original."""
    t = Topology()
    t.add_connection(_socket_conn("x", "y"))

    copy = t.copy()
    edges = list(copy.list_connections())
    assert len(edges) == 1
    assert edges[0].source == NodeId("x")
    assert edges[0].sink == NodeId("y")


def test_copy_empty():
    """Copying an empty topology must not raise."""
    t = Topology()
    copy = t.copy()
    assert list(copy.list_nodes()) == []
