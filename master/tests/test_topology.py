import pytest

from shared.topology import Topology
from shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from shared.types.topology import Connection, ConnectionProfile, Node, NodeId


@pytest.fixture
def topology() -> Topology:
    return Topology()

@pytest.fixture
def connection() -> Connection:
    return Connection(source_node_id=NodeId(), sink_node_id=NodeId(), source_multiaddr="/ip4/127.0.0.1/tcp/1234", sink_multiaddr="/ip4/127.0.0.1/tcp/1235", connection_profile=ConnectionProfile(throughput=1000, latency=1000, jitter=1000))

@pytest.fixture
def node_profile() -> NodePerformanceProfile:
    memory_profile = MemoryPerformanceProfile(ram_total=1000, ram_available=1000, swap_total=1000, swap_available=1000)
    system_profile = SystemPerformanceProfile(flops_fp16=1000)
    return NodePerformanceProfile(model_id="test", chip_id="test", memory=memory_profile, network_interfaces=[], system=system_profile)

@pytest.fixture
def connection_profile() -> ConnectionProfile:
    return ConnectionProfile(throughput=1000, latency=1000, jitter=1000)

def test_add_node(topology: Topology, node_profile: NodePerformanceProfile):
    # arrange
    node_id = NodeId()

    # act
    topology.add_node(Node(node_id=node_id, node_profile=node_profile), node_id=node_id)

    # assert
    data = topology.get_node_profile(node_id)
    assert data == node_profile


def test_add_connection(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    # act
    data = topology.get_connection_profile(connection)

    # assert
    assert data == connection.connection_profile 

def test_update_node_profile(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    new_node_profile = NodePerformanceProfile(model_id="test", chip_id="test", memory=MemoryPerformanceProfile(ram_total=1000, ram_available=1000, swap_total=1000, swap_available=1000), network_interfaces=[], system=SystemPerformanceProfile(flops_fp16=1000))

    # act
    topology.update_node_profile(connection.source_node_id, node_profile=new_node_profile)

    # assert
    data = topology.get_node_profile(connection.source_node_id)
    assert data == new_node_profile

def test_update_connection_profile(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    new_connection_profile = ConnectionProfile(throughput=2000, latency=2000, jitter=2000)
    connection = Connection(source_node_id=connection.source_node_id, sink_node_id=connection.sink_node_id, source_multiaddr=connection.source_multiaddr, sink_multiaddr=connection.sink_multiaddr, connection_profile=new_connection_profile)

    # act
    topology.update_connection_profile(connection)

    # assert
    data = topology.get_connection_profile(connection)
    assert data == new_connection_profile

def test_remove_connection_still_connected(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    # act
    topology.remove_connection(connection)

    # assert
    with pytest.raises(IndexError):
        topology.get_connection_profile(connection)
    
def test_remove_connection_bridge(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    """Create a bridge scenario: master -> node_a -> node_b
    and remove the bridge connection (master -> node_a)"""
    # arrange
    master_id = NodeId()
    node_a_id = NodeId()
    node_b_id = NodeId()
    
    topology.add_node(Node(node_id=master_id, node_profile=node_profile), node_id=master_id)
    topology.add_node(Node(node_id=node_a_id, node_profile=node_profile), node_id=node_a_id)
    topology.add_node(Node(node_id=node_b_id, node_profile=node_profile), node_id=node_b_id)
    
    connection_master_to_a = Connection(
        source_node_id=master_id,
        sink_node_id=node_a_id,
        source_multiaddr="/ip4/127.0.0.1/tcp/1234",
        sink_multiaddr="/ip4/127.0.0.1/tcp/1235",
        connection_profile=ConnectionProfile(throughput=1000, latency=1000, jitter=1000)
    )
    
    connection_a_to_b = Connection(
        source_node_id=node_a_id,
        sink_node_id=node_b_id,
        source_multiaddr="/ip4/127.0.0.1/tcp/1236",
        sink_multiaddr="/ip4/127.0.0.1/tcp/1237",
        connection_profile=ConnectionProfile(throughput=1000, latency=1000, jitter=1000)
    )
    
    topology.add_connection(connection_master_to_a)
    topology.add_connection(connection_a_to_b)
    
    assert len(list(topology.list_nodes())) == 3
    
    topology.remove_connection(connection_master_to_a)
    
    remaining_nodes = list(topology.list_nodes())
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0].node_id == master_id
    
    with pytest.raises(KeyError):
        topology.get_node_profile(node_a_id)
    
    with pytest.raises(KeyError):
        topology.get_node_profile(node_b_id)


def test_remove_node_still_connected(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    # act
    topology.remove_node(connection.source_node_id)

    # assert
    with pytest.raises(KeyError):
        topology.get_node_profile(connection.source_node_id)


def test_list_nodes(topology: Topology, node_profile: NodePerformanceProfile, connection: Connection):
    # arrange
    topology.add_node(Node(node_id=connection.source_node_id, node_profile=node_profile), node_id=connection.source_node_id)
    topology.add_node(Node(node_id=connection.sink_node_id, node_profile=node_profile), node_id=connection.sink_node_id)
    topology.add_connection(connection)

    # act
    nodes = list(topology.list_nodes())

    # assert
    assert len(nodes) == 2
    assert all(isinstance(node, Node) for node in nodes)
    assert {node.node_id for node in nodes} == {connection.source_node_id, connection.sink_node_id}
