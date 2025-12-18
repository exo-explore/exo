from typing import Callable

import pytest
from loguru import logger

from exo.master.placement import (
    get_transition_events,
    place_instance,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.profiling import NetworkInterfaceInfo, NodePerformanceProfile
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def topology() -> Topology:
    return Topology()


@pytest.fixture
def instance() -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts=[],
    )


@pytest.fixture
def model_meta() -> ModelMetadata:
    return ModelMetadata(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        pretty_name="Test Model",
        n_layers=10,
    )


def place_instance_command(model_meta: ModelMetadata) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_meta=model_meta,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
    ],
)
def test_get_instance_placements_create_instance(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    topology: Topology,
    model_meta: ModelMetadata,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # arrange
    model_meta.n_layers = total_layers
    model_meta.storage_size.in_bytes = sum(
        available_memory
    )  # make it exactly fit across all nodes

    cic = place_instance_command(model_meta)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    topology.add_node(create_node(available_memory[0], node_id_a))
    topology.add_node(create_node(available_memory[1], node_id_b))
    topology.add_node(create_node(available_memory[2], node_id_c))
    topology.add_connection(create_connection(node_id_a, node_id_b))
    topology.add_connection(create_connection(node_id_b, node_id_c))
    topology.add_connection(create_connection(node_id_c, node_id_a))

    # act
    placements = place_instance(cic, topology, {})

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_meta.model_id

    runner_id_a = instance.shard_assignments.node_to_runner[node_id_a]
    runner_id_b = instance.shard_assignments.node_to_runner[node_id_b]
    runner_id_c = instance.shard_assignments.node_to_runner[node_id_c]

    shard_a = instance.shard_assignments.runner_to_shard[runner_id_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_id_b]
    shard_c = instance.shard_assignments.runner_to_shard[runner_id_c]

    assert shard_a.end_layer - shard_a.start_layer == expected_layers[0]
    assert shard_b.end_layer - shard_b.start_layer == expected_layers[1]
    assert shard_c.end_layer - shard_c.start_layer == expected_layers[2]

    shards = [shard_a, shard_b, shard_c]
    shards_sorted = sorted(shards, key=lambda s: s.start_layer)
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


def test_get_instance_placements_one_node_exact_fit(
    create_node: Callable[[int, NodeId | None], NodeInfo],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1000 * 1024, node_id))
    cic = place_instance_command(
        ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = place_instance(cic, topology, {})

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_get_instance_placements_one_node_fits_with_extra_memory(
    create_node: Callable[[int, NodeId | None], NodeInfo],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1001 * 1024, node_id))
    cic = place_instance_command(
        ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = place_instance(cic, topology, {})

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_get_instance_placements_one_node_not_fit(
    create_node: Callable[[int, NodeId | None], NodeInfo],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1000 * 1024, node_id))
    cic = place_instance_command(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1001),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        place_instance(cic, topology, {})


def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_placement_prioritizes_leaf_cycle_with_less_memory(
    topology: Topology,
    model_meta: ModelMetadata,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # Arrange two 3-node cycles. The A-B-C cycle has a leaf node (only one outgoing
    # neighbor per node). The D-E-F cycle has extra outgoing edges making its nodes
    # non-leaves. Ensure both cycles have sufficient total memory, with the A-B-C
    # cycle having LESS total memory than D-E-F. The algorithm should still choose
    # the cycle that contains a leaf node.

    # Model requires more than any single node but fits within a 3-node cycle
    model_meta.storage_size.in_bytes = 1500
    model_meta.n_layers = 12

    # Create node ids
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()
    node_id_e = NodeId()
    node_id_f = NodeId()

    # Extra sink nodes to make D/E/F non-leaf via additional outgoing edges
    node_id_x = NodeId()
    node_id_y = NodeId()
    node_id_z = NodeId()

    # A-B-C cycle total memory = 1600 (< D-E-F total)
    topology.add_node(create_node(400, node_id_a))
    topology.add_node(create_node(400, node_id_b))
    topology.add_node(create_node(800, node_id_c))

    # D-E-F cycle total memory = 1800 (> A-B-C total)
    topology.add_node(create_node(600, node_id_d))
    topology.add_node(create_node(600, node_id_e))
    topology.add_node(create_node(600, node_id_f))

    # Extra nodes with tiny memory so they can't form singleton placements
    topology.add_node(create_node(10, node_id_x))
    topology.add_node(create_node(10, node_id_y))
    topology.add_node(create_node(10, node_id_z))

    # Build directed cycles
    topology.add_connection(create_connection(node_id_a, node_id_b))
    topology.add_connection(create_connection(node_id_b, node_id_c))
    topology.add_connection(create_connection(node_id_c, node_id_a))

    topology.add_connection(create_connection(node_id_d, node_id_e))
    topology.add_connection(create_connection(node_id_e, node_id_f))
    topology.add_connection(create_connection(node_id_f, node_id_d))

    # Add extra outgoing edges from D/E/F so none of them are leaves
    topology.add_connection(create_connection(node_id_d, node_id_x))
    topology.add_connection(create_connection(node_id_e, node_id_y))
    topology.add_connection(create_connection(node_id_f, node_id_z))

    cic = place_instance_command(
        model_meta=model_meta,
    )

    # Act
    placements = place_instance(cic, topology, {})

    # Assert the chosen cycle is A-B-C (contains at least one leaf node), even though
    # D-E-F has more total memory.
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    expected_leaf_cycle_nodes = {node_id_a, node_id_b, node_id_c}
    non_leaf_cycle_nodes = {node_id_d, node_id_e, node_id_f}

    assert expected_leaf_cycle_nodes.issubset(assigned_nodes)
    assert assigned_nodes.isdisjoint(non_leaf_cycle_nodes)


def test_tensor_rdma_backend_connectivity_matrix(
    topology: Topology,
    model_meta: ModelMetadata,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    model_meta.n_layers = 12
    model_meta.storage_size.in_bytes = 1500

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()

    node_a = create_node(500, node_id_a)
    node_b = create_node(500, node_id_b)
    node_c = create_node(500, node_id_c)

    ethernet_interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="192.168.1.100",
    )

    assert node_a.node_profile is not None
    assert node_b.node_profile is not None
    assert node_c.node_profile is not None

    conn_a_b = create_connection(node_id_a, node_id_b)
    conn_b_c = create_connection(node_id_b, node_id_c)
    conn_c_a = create_connection(node_id_c, node_id_a)

    conn_b_a = create_connection(node_id_b, node_id_a)
    conn_c_b = create_connection(node_id_c, node_id_b)
    conn_a_c = create_connection(node_id_a, node_id_c)

    assert conn_a_b.send_back_multiaddr is not None
    assert conn_b_c.send_back_multiaddr is not None
    assert conn_c_a.send_back_multiaddr is not None

    assert conn_b_a.send_back_multiaddr is not None
    assert conn_c_b.send_back_multiaddr is not None
    assert conn_a_c.send_back_multiaddr is not None

    node_a.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_a.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_c_a.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_b_a.send_back_multiaddr.ip_address,
            ),
            ethernet_interface,
        ],
        system=node_a.node_profile.system,
    )
    node_b.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_b.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_c_b.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_a_b.send_back_multiaddr.ip_address,
            ),
            ethernet_interface,
        ],
        system=node_b.node_profile.system,
    )
    node_c.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_c.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_a_c.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_b_c.send_back_multiaddr.ip_address,
            ),
            ethernet_interface,
        ],
        system=node_c.node_profile.system,
    )

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_a_c)

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_meta=model_meta,
        min_nodes=1,
    )

    placements = place_instance(cic, topology, {})

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assert isinstance(instance, MlxJacclInstance)

    assert instance.ibv_devices is not None
    assert instance.ibv_coordinators is not None

    matrix = instance.ibv_devices
    assert len(matrix) == 3

    for i in range(3):
        assert matrix[i][i] is None

    assigned_nodes = list(instance.shard_assignments.node_to_runner.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(assigned_nodes)}

    idx_a = node_to_idx[node_id_a]
    idx_b = node_to_idx[node_id_b]
    idx_c = node_to_idx[node_id_c]

    logger.info(matrix)

    assert matrix[idx_a][idx_b] == "rdma_en4"
    assert matrix[idx_b][idx_c] == "rdma_en3"
    assert matrix[idx_c][idx_a] == "rdma_en3"

    # Verify coordinators are set for all nodes
    assert len(instance.ibv_coordinators) == 3
    for node_id in assigned_nodes:
        assert node_id in instance.ibv_coordinators
        coordinator = instance.ibv_coordinators[node_id]
        assert ":" in coordinator
        # Rank 0 node should use 0.0.0.0, others should use connection-specific IPs
        if node_id == assigned_nodes[0]:
            assert coordinator.startswith("0.0.0.0:")
        else:
            # Non-rank-0 nodes should have valid IP addresses (can be link-local)
            ip_part = coordinator.split(":")[0]
            # Just verify it's a valid IP format
            assert len(ip_part.split(".")) == 4
