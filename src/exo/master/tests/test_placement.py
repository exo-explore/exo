import pytest
from loguru import logger

from exo.master.placement import (
    get_transition_events,
    place_instance,
)
from exo.master.tests.conftest import (
    create_connection,
    create_node_profile,
    create_rdma_connection,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NetworkInterfaceInfo
from exo.shared.types.topology import SocketConnection
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
    model_meta: ModelMetadata,
):
    # arrange
    model_meta.n_layers = total_layers
    model_meta.storage_size.in_bytes = sum(
        available_memory
    )  # make it exactly fit across all nodes
    topology = Topology()

    cic = place_instance_command(model_meta)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    profiles = {
        node_id_a: create_node_profile(available_memory[0]),
        node_id_b: create_node_profile(available_memory[1]),
        node_id_c: create_node_profile(available_memory[2]),
    }
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(node_id_a, node_id_b, create_connection(1))
    topology.add_connection(node_id_b, node_id_c, create_connection(2))
    topology.add_connection(node_id_c, node_id_a, create_connection(3))

    # act
    placements = place_instance(cic, topology, {}, profiles)

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


def test_get_instance_placements_one_node_exact_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    profiles = {node_id: create_node_profile(1000 * 1024)}
    cic = place_instance_command(
        ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = place_instance(cic, topology, {}, profiles)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_get_instance_placements_one_node_fits_with_extra_memory() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    profiles = {node_id: create_node_profile(1001 * 1024)}
    cic = place_instance_command(
        ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = place_instance(cic, topology, {}, profiles)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_get_instance_placements_one_node_not_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    profiles = {node_id: create_node_profile(1000 * 1024)}
    cic = place_instance_command(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1001),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        place_instance(cic, topology, {}, profiles)


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
    model_meta: ModelMetadata,
):
    # arrange
    topology = Topology()

    model_meta.storage_size = Memory.from_bytes(1000)

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()

    profiles = {
        node_id_a: create_node_profile(500),
        node_id_b: create_node_profile(600),
        node_id_c: create_node_profile(600),
        node_id_d: create_node_profile(500),
    }

    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_node(node_id_d)

    # Daisy chain topology
    topology.add_connection(node_id_a, node_id_b, create_connection(1))
    topology.add_connection(node_id_b, node_id_a, create_connection(1))
    topology.add_connection(node_id_b, node_id_c, create_connection(1))
    topology.add_connection(node_id_c, node_id_b, create_connection(1))
    topology.add_connection(node_id_c, node_id_d, create_connection(1))
    topology.add_connection(node_id_d, node_id_c, create_connection(1))

    logger.info(list(topology.list_connections()))

    cic = place_instance_command(
        model_meta=model_meta,
    )

    # act
    placements = place_instance(cic, topology, {}, profiles)

    # assert
    assert len(placements) == 1
    instance = list(placements.values())[0]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == set((node_id_a, node_id_b)) or assigned_nodes == set(
        (node_id_c, node_id_d)
    )


def test_tensor_rdma_backend_connectivity_matrix(
    model_meta: ModelMetadata,
):
    topology = Topology()
    model_meta.n_layers = 12
    model_meta.storage_size.in_bytes = 1500

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    profiles = {
        node_a: create_node_profile(500),
        node_b: create_node_profile(500),
        node_c: create_node_profile(500),
    }

    ethernet_interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="192.168.1.100",
    )
    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/192.168.1.{100}/tcp/{8000}")
    )

    profiles[node_a].network_interfaces = [ethernet_interface]
    profiles[node_b].network_interfaces = [ethernet_interface]
    profiles[node_c].network_interfaces = [ethernet_interface]

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
    topology.add_connection(node_a, node_b, create_rdma_connection(3))
    topology.add_connection(node_b, node_c, create_rdma_connection(4))
    topology.add_connection(node_c, node_a, create_rdma_connection(5))
    topology.add_connection(node_b, node_a, create_rdma_connection(3))
    topology.add_connection(node_c, node_b, create_rdma_connection(4))
    topology.add_connection(node_a, node_c, create_rdma_connection(5))

    topology.add_connection(node_a, node_b, ethernet_conn)
    topology.add_connection(node_b, node_c, ethernet_conn)
    topology.add_connection(node_c, node_a, ethernet_conn)
    topology.add_connection(node_a, node_c, ethernet_conn)
    topology.add_connection(node_b, node_a, ethernet_conn)
    topology.add_connection(node_c, node_b, ethernet_conn)

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_meta=model_meta,
        min_nodes=1,
    )

    placements = place_instance(cic, topology, {}, profiles)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assert isinstance(instance, MlxJacclInstance)

    assert instance.jaccl_devices is not None
    assert instance.jaccl_coordinators is not None

    matrix = instance.jaccl_devices
    assert len(matrix) == 3

    for i in range(3):
        assert matrix[i][i] is None

    assigned_nodes = list(instance.shard_assignments.node_to_runner.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(assigned_nodes)}

    idx_a = node_to_idx[node_a]
    idx_b = node_to_idx[node_b]
    idx_c = node_to_idx[node_c]

    logger.info(matrix)

    assert matrix[idx_a][idx_b] == "rdma_en3"
    assert matrix[idx_b][idx_c] == "rdma_en4"
    assert matrix[idx_c][idx_a] == "rdma_en5"

    # Verify coordinators are set for all nodes
    assert len(instance.jaccl_coordinators) == 3
    for node_id in assigned_nodes:
        assert node_id in instance.jaccl_coordinators
        coordinator = instance.jaccl_coordinators[node_id]
        assert ":" in coordinator
        # Rank 0 node should use 0.0.0.0, others should use connection-specific IPs
        if node_id == assigned_nodes[0]:
            assert coordinator.startswith("0.0.0.0:")
        else:
            # Non-rank-0 nodes should have valid IP addresses (can be link-local)
            ip_part = coordinator.split(":")[0]
            # Just verify it's a valid IP format
            assert len(ip_part.split(".")) == 4
