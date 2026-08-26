import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    allocate_layers_by_throughput,
    estimate_memory_bandwidth_gigabytes_per_second,
    find_ip_prioritised,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


def test_allocate_layers_by_throughput_fills_fastest_first() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=12,
        node_throughputs=[819.0, 273.0, 800.0],
        max_layers_per_node=[4, 4, 8],
    )
    assert allocations == [4, 1, 7]
    assert sum(allocations) == 12


def test_allocate_layers_by_throughput_keeps_one_layer_per_node() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=3,
        node_throughputs=[1000.0, 1.0, 1.0],
        max_layers_per_node=[3, 3, 3],
    )
    assert allocations == [1, 1, 1]


def test_allocate_layers_by_throughput_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _ = allocate_layers_by_throughput(
            total_layers=10,
            node_throughputs=[100.0, 100.0],
            max_layers_per_node=[4, 4],
        )


def test_estimate_memory_bandwidth_matches_known_chips() -> None:
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Apple M3 Ultra")
        )
        == 819.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GB10")
        )
        == 273.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GeForce RTX 3090")
        )
        == 936.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Unknown Chip")
        )
        is None
    )


def _fully_connected_three_node_topology() -> tuple[Topology, NodeId, NodeId, NodeId]:
    node_a, node_b, node_c = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for node_id in (node_a, node_b, node_c):
        topology.add_node(node_id)
    pairs = [
        (node_a, node_b),
        (node_b, node_c),
        (node_c, node_a),
        (node_b, node_a),
        (node_c, node_b),
        (node_a, node_c),
    ]
    for index, (source, sink) in enumerate(pairs):
        topology.add_connection(
            Connection(
                source=source, sink=sink, edge=create_socket_connection(index + 1)
            )
        )
    return topology, node_a, node_b, node_c


def test_pipeline_placement_uses_bandwidth_when_identities_known() -> None:
    topology, node_a, node_b, node_c = _fully_connected_three_node_topology()
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(1500),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    command = PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=3,
    )
    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(1000),
    }
    node_network = {node_id: create_node_network() for node_id in node_memory}
    node_backends = {node_id: [Backend.MlxMetal] for node_id in node_memory}
    node_identities = {
        node_a: NodeIdentity(chip_id="Apple M3 Ultra"),
        node_b: NodeIdentity(chip_id="NVIDIA GB10"),
        node_c: NodeIdentity(chip_id="Apple M2 Ultra"),
    }

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))

    def layer_count(node_id: NodeId) -> int:
        runner_id = instance.shard_assignments.node_to_runner[node_id]
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        return shard.end_layer - shard.start_layer

    # Fastest nodes are filled to their memory caps first; the slow GB10
    # keeps the single layer every pipeline stage requires.
    assert layer_count(node_a) == 4
    assert layer_count(node_b) == 1
    assert layer_count(node_c) == 7


def _two_node_topology_with_two_links(
    thunderbolt_ip: str, ethernet_ip: str
) -> tuple[Topology, NodeId, NodeId]:
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    for ip in (thunderbolt_ip, ethernet_ip):
        last_octet = int(ip.rsplit(".", 1)[1])
        topology.add_connection(
            Connection(
                source=node_a,
                sink=node_b,
                edge=create_socket_connection(last_octet),
            )
        )
    return topology, node_a, node_b


def test_find_ip_prioritised_prefers_measured_link_speed_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                    link_speed_megabits=200_000,
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # 200 GbE with a measured speed beats thunderbolt's nominal 40 Gb/s.
    assert selected_ip == ethernet_ip


def test_find_ip_prioritised_falls_back_to_nominal_speeds_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # Without measured speeds the previous type preference is preserved.
    assert selected_ip == thunderbolt_ip
