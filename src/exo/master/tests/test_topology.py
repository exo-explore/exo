import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import SocketConnection


@pytest.fixture
def topology() -> Topology:
    return Topology()


@pytest.fixture
def connection() -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/1235"),
    )


@pytest.fixture
def node_profile() -> NodePerformanceProfile:
    memory_profile = MemoryUsage.from_bytes(
        ram_total=1000, ram_available=1000, swap_total=1000, swap_available=1000
    )
    system_profile = SystemPerformanceProfile()
    return NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=memory_profile,
        network_interfaces=[],
        system=system_profile,
    )


def test_add_node(topology: Topology):
    # arrange
    node_id = NodeId()

    # act
    topology.add_node(node_id)

    # assert
    assert topology.node_is_leaf(node_id)


def test_add_connection(topology: Topology, connection: SocketConnection):
    # arrange
    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(node_a, node_b, connection)

    # act
    data = list(conn for _, _, conn in topology.list_connections())

    # assert
    assert data == [connection]

    assert topology.node_is_leaf(node_a)
    assert topology.node_is_leaf(node_b)


def test_remove_connection_still_connected(
    topology: Topology, connection: SocketConnection
):
    # arrange
    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(node_a, node_b, connection)

    # act
    topology.remove_connection(node_a, node_b, connection)

    # assert
    assert list(topology.get_all_connections_between(node_a, node_b)) == []


def test_remove_node_still_connected(topology: Topology, connection: SocketConnection):
    # arrange
    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(node_a, node_b, connection)
    assert list(topology.out_edges(node_a)) == [(node_b, connection)]

    # act
    topology.remove_node(node_b)

    # assert
    assert list(topology.out_edges(node_a)) == []


def test_list_nodes(topology: Topology, connection: SocketConnection):
    # arrange
    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(node_a, node_b, connection)
    assert list(topology.out_edges(node_a)) == [(node_b, connection)]

    # act
    nodes = list(topology.list_nodes())

    # assert
    assert len(nodes) == 2
    assert all(isinstance(node, NodeId) for node in nodes)
    assert {node for node in nodes} == {node_a, node_b}
