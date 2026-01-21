import pytest

from exo.master.placement_utils import (
    allocate_layers_proportionally,
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_mlx_jaccl_coordinators,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.shards import Sharding


def test_filter_cycles_by_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_bytes(1))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n for n in filtered_cycles[0]) == {node1_id, node2_id}


def test_filter_cycles_by_insufficient_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles = filter_cycles_by_memory(
        topology.get_cycles(), node_memory, Memory.from_kb(2001)
    )

    # assert
    assert len(filtered_cycles) == 0


def test_filter_multiple_cycles_by_memory():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    node_a_mem = create_node_memory(500 * 1024)
    node_b_mem = create_node_memory(500 * 1024)
    node_c_mem = create_node_memory(1000 * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = topology.get_cycles()

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_kb(1500))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n for n in filtered_cycles[0]) == {
        node_a_id,
        node_b_id,
        node_c_id,
    }


def test_get_smallest_cycles():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]  # ignore singletons

    # act
    smallest_cycles = get_smallest_cycles(cycles)

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n for n in smallest_cycles[0]) == {node_a_id, node_b_id}


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
        # Edge case: one node has ~90% of memory - should not over-allocate.
        # Each node must have enough memory for at least 1 layer (50 KB = 1000/20).
        ((900, 50, 50), 20, (18, 1, 1)),
    ],
)
def test_get_shard_assignments(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # create connections (A -> B -> C -> A forms a 3-cycle, plus B -> A also exists)
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(4)
    )

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    node_a_mem = create_node_memory(available_memory[0] * 1024)
    node_b_mem = create_node_memory(available_memory[1] * 1024)
    node_c_mem = create_node_memory(available_memory[2] * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=total_layers,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )

    cycles = topology.get_cycles()

    # pick the 3-node cycle deterministically (cycle ordering can vary)
    selected_cycle = next(cycle for cycle in cycles if len(cycle) == 3)

    # act
    shard_assignments = get_shard_assignments(
        model_card, selected_cycle, Sharding.Pipeline, node_memory=node_memory
    )

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]

    assert (
        shard_assignments.runner_to_shard[runner_id_a].end_layer
        - shard_assignments.runner_to_shard[runner_id_a].start_layer
        == expected_layers[0]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_b].end_layer
        - shard_assignments.runner_to_shard[runner_id_b].start_layer
        == expected_layers[1]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_c].end_layer
        - shard_assignments.runner_to_shard[runner_id_c].start_layer
        == expected_layers[2]
    )


def test_get_hosts_from_subgraph():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )

    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)

    # act
    hosts = get_hosts_from_subgraph(topology)

    # assert
    assert len(hosts) == 3
    expected_hosts = [
        Host(ip="169.254.0.1", port=1234),
        Host(ip="169.254.0.2", port=1234),
        Host(ip="169.254.0.3", port=1234),
    ]
    for expected_host in expected_hosts:
        assert expected_host in hosts


def test_get_mlx_jaccl_coordinators():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(5)
    )
    conn_a_c = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(6)
    )

    network_a = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.5"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.2"),
        ]
    )
    network_b = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.4"),
        ]
    )
    network_c = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.3"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.6"),
        ]
    )
    node_network = {
        node_a_id: network_a,
        node_b_id: network_b,
        node_c_id: network_c,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_a_c)

    # act
    coordinators = get_mlx_jaccl_coordinators(
        node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    # assert
    assert len(coordinators) == 3
    assert node_a_id in coordinators
    assert node_b_id in coordinators
    assert node_c_id in coordinators

    # All coordinators should have IP:PORT format
    for node_id, coordinator in coordinators.items():
        assert ":" in coordinator, (
            f"Coordinator for {node_id} should have ':' separator"
        )

    # Verify port is correct
    for node_id, coordinator in coordinators.items():
        assert coordinator.endswith(":5000"), (
            f"Coordinator for {node_id} should use port 5000"
        )

    # Rank 0 (node_a) treats this as the listen socket so should listen on all IPs
    assert coordinators[node_a_id].startswith("0.0.0.0:"), (
        "Rank 0 node should use 0.0.0.0 as coordinator listen address"
    )

    # Non-rank-0 nodes should use the specific IP from their connection to rank 0
    # node_b uses the IP from conn_b_a (node_b -> node_a)
    assert isinstance(conn_b_a.edge, SocketConnection)
    assert (
        coordinators[node_b_id] == f"{conn_b_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_b should use the IP from conn_b_a"

    # node_c uses the IP from conn_c_a (node_c -> node_a)
    assert isinstance(conn_c_a.edge, SocketConnection)
    assert coordinators[node_c_id] == (
        f"{conn_c_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_c should use the IP from conn_c_a"


class TestAllocateLayersProportionally:
    def test_empty_node_list_raises(self):
        with pytest.raises(ValueError, match="empty node list"):
            allocate_layers_proportionally(total_layers=10, memory_fractions=[])

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=0, memory_fractions=[0.5, 0.5])

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=-1, memory_fractions=[0.5, 0.5])

    def test_fewer_layers_than_nodes_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(
                total_layers=2, memory_fractions=[0.33, 0.33, 0.34]
            )

    def test_equal_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.25, 0.25]
        )
        assert result == [3, 3, 3, 3]
        assert sum(result) == 12

    def test_proportional_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.50]
        )
        assert result == [3, 3, 6]
        assert sum(result) == 12

    def test_extreme_imbalance_ensures_minimum(self):
        result = allocate_layers_proportionally(
            total_layers=20, memory_fractions=[0.975, 0.0125, 0.0125]
        )
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 20
        # Small nodes get minimum 1 layer
        assert result == [18, 1, 1]

    def test_single_node_gets_all_layers(self):
        result = allocate_layers_proportionally(total_layers=10, memory_fractions=[1.0])
        assert result == [10]

    def test_minimum_viable_allocation(self):
        result = allocate_layers_proportionally(
            total_layers=3, memory_fractions=[0.33, 0.33, 0.34]
        )
        assert result == [1, 1, 1]
        assert sum(result) == 3


def test_get_shard_assignments_insufficient_memory_raises():
    """Test that ValueError is raised when a node has insufficient memory for its layers."""
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    # Node C has only 10 KB but would need 50 KB for 1 layer (1000 KB / 20 layers)
    node_a_mem = create_node_memory(900 * 1024)
    node_b_mem = create_node_memory(50 * 1024)
    node_c_mem = create_node_memory(10 * 1024)  # Insufficient memory

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)

    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=20,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments(
            model_card, selected_cycle, Sharding.Pipeline, node_memory
        )
