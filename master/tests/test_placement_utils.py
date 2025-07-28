from ipaddress import IPv4Address
from typing import Callable

import pytest

from master.utils.placement_utils import (
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_shard_assignments,
    get_smallest_cycles,
)
from shared.topology import Topology
from shared.types.common import Host, NodeId
from shared.types.models import ModelMetadata
from shared.types.topology import Connection, Node


@pytest.fixture
def topology() -> Topology:
    topology = Topology()
    return topology


def test_filter_cycles_by_memory(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId], Connection]):
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()

    node1 = create_node(1000*1024, node1_id)
    node2 = create_node(1000*1024, node2_id)
    
    topology.add_node(node1)
    topology.add_node(node2)
    
    connection1 = create_connection(node1_id, node2_id)
    connection2 = create_connection(node2_id, node1_id)
    
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    
    cycles = topology.get_cycles()
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, 1)

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n.node_id for n in filtered_cycles[0]) == {node1_id, node2_id}


def test_filter_cycles_by_insufficient_memory(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId], Connection]):
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()

    node1 = create_node(1000*1024, node1_id)
    node2 = create_node(1000*1024, node2_id)

    topology.add_node(node1)
    topology.add_node(node2)

    connection1 = create_connection(node1_id, node2_id)
    connection2 = create_connection(node2_id, node1_id)
    
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    
    # act
    filtered_cycles = filter_cycles_by_memory(topology.get_cycles(), 2001*1024)

    # assert
    assert len(filtered_cycles) == 0


def test_filter_multiple_cycles_by_memory(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId], Connection]):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    
    node_a = create_node(500*1024, node_a_id)
    node_b = create_node(500*1024, node_b_id)
    node_c = create_node(1000*1024, node_c_id)
    
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
    
    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))
    
    topology.add_connection(create_connection(node_a_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_b_id))
    
    cycles = topology.get_cycles()
    
    # act
    filtered_cycles = filter_cycles_by_memory(cycles, 1500*1024)
    
    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n.node_id for n in filtered_cycles[0]) == {node_a_id, node_b_id, node_c_id}

def test_get_smallest_cycles(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId], Connection]):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    
    node_a = create_node(500*1024, node_a_id)
    node_b = create_node(500*1024, node_b_id)
    node_c = create_node(1000*1024, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_a_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))

    # act
    smallest_cycles = get_smallest_cycles(topology.get_cycles())

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n.node_id for n in smallest_cycles[0]) == {node_a_id, node_b_id}

@pytest.mark.parametrize("available_memory,total_layers,expected_layers", [
    ((500, 500, 1000), 12, (3, 3, 6)),
    ((500, 500, 500), 12, (4, 4, 4)),
    ((312, 518, 1024), 12, (2, 3, 7))
])
def test_get_shard_assignments(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId], Connection], available_memory: tuple[int, int, int], total_layers: int, expected_layers: tuple[int, int, int]):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    
    node_a = create_node(available_memory[0]*1024, node_a_id)
    node_b = create_node(available_memory[1]*1024, node_b_id)
    node_c = create_node(available_memory[2]*1024, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_a_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))
    
    model_meta = ModelMetadata(
        model_id="test-model",
        pretty_name="Test Model",
        n_layers=total_layers,
        storage_size_kilobytes=1000
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]
    
    # act
    shard_assignments = get_shard_assignments(model_meta, selected_cycle)

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]
    assert shard_assignments.runner_to_shard[runner_id_c].end_layer - shard_assignments.runner_to_shard[runner_id_c].start_layer == expected_layers[2]
    assert shard_assignments.runner_to_shard[runner_id_a].end_layer - shard_assignments.runner_to_shard[runner_id_a].start_layer == expected_layers[0]
    assert shard_assignments.runner_to_shard[runner_id_b].end_layer - shard_assignments.runner_to_shard[runner_id_b].start_layer == expected_layers[1]


def test_get_hosts_from_subgraph(topology: Topology, create_node: Callable[[int, NodeId | None], Node], create_connection: Callable[[NodeId, NodeId, int | None], Connection]):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    
    node_a = create_node(500, node_a_id)
    node_b = create_node(500, node_b_id)
    node_c = create_node(1000, node_c_id)
    
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
        
    topology.add_connection(create_connection(node_a_id, node_b_id, 5001))
    topology.add_connection(create_connection(node_b_id, node_c_id, 5002))
    topology.add_connection(create_connection(node_c_id, node_a_id, 5003))
    topology.add_connection(create_connection(node_b_id, node_a_id, 5004))

    # act
    hosts = get_hosts_from_subgraph(topology)

    # assert
    assert len(hosts) == 3
    expected_hosts = [
        Host(ip=IPv4Address("127.0.0.1"), port=5001),
        Host(ip=IPv4Address("127.0.0.1"), port=5002),
        Host(ip=IPv4Address("127.0.0.1"), port=5003),
    ]
    for expected_host in expected_hosts:
        assert expected_host in hosts
