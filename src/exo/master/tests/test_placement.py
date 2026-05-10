import pytest

from exo.master.placement import (
    _is_routable_jaccl_ipv4,  # pyright: ignore[reportPrivateUsage]
    get_transition_events,
    place_instance,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
)
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import (
    AsymmetricTensorShardMetadata,
    PipelineShardMetadata,
    Sharding,
)


def create_jaccl_node_network(
    thunderbolt_ip_address: str,
    ethernet_ip_address: str = "192.168.1.10",
) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(
                name="en1",
                ip_address=thunderbolt_ip_address,
                interface_type="thunderbolt",
            ),
            NetworkInterfaceInfo(
                name="en9",
                ip_address=ethernet_ip_address,
                interface_type="ethernet",
            ),
        ]
    )


@pytest.fixture
def instance() -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


@pytest.fixture
def model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def place_instance_command(model_card: ModelCard) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def create_node_memory_with_total(*, available: int, total: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=total,
        ram_available=available,
        swap_total=0,
        swap_available=0,
    )


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 468, 1092), 12, (2, 3, 7)),
    ],
)
def test_get_instance_placements_create_instance(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    model_card: ModelCard,
):
    # arrange
    model_card = model_card.model_copy(
        update={
            "n_layers": total_layers,
            "storage_size": Memory.from_bytes(
                sum(available_memory)
            ),  # make it exactly fit across all nodes
        }
    )
    topology = Topology()

    cic = place_instance_command(model_card)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_id_a, sink=node_id_b, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_id_b, sink=node_id_c, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_id_c, sink=node_id_a, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_id_c, sink=node_id_b, edge=create_socket_connection(4)
    )
    conn_a_c = Connection(
        source=node_id_a, sink=node_id_c, edge=create_socket_connection(5)
    )
    conn_b_a = Connection(
        source=node_id_b, sink=node_id_a, edge=create_socket_connection(6)
    )

    node_memory = {
        node_id_a: create_node_memory(available_memory[0]),
        node_id_b: create_node_memory(available_memory[1]),
        node_id_c: create_node_memory(available_memory[2]),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
    }
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_a_c)
    topology.add_connection(conn_b_a)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_card.model_id

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
    node_memory = {node_id: create_node_memory(1000 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

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
    node_memory = {node_id: create_node_memory(1001 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

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
    node_memory = {node_id: create_node_memory(1000 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1001),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        place_instance(cic, topology, {}, node_memory, node_network)


def test_filtered_single_node_placement_can_use_total_memory_capacity() -> None:
    topology = Topology()
    selected_node = NodeId()
    other_node = NodeId()
    topology.add_node(selected_node)
    topology.add_node(other_node)
    topology.add_connection(
        Connection(
            source=selected_node, sink=other_node, edge=create_socket_connection(1)
        )
    )
    topology.add_connection(
        Connection(
            source=other_node, sink=selected_node, edge=create_socket_connection(2)
        )
    )
    node_memory = {
        selected_node: create_node_memory_with_total(available=1000, total=2000),
        other_node: create_node_memory_with_total(available=2000, total=2000),
    }
    node_network = {
        selected_node: create_node_network(),
        other_node: create_node_network(),
    }
    command = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_bytes(1500),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        allowed_nodes={selected_node},
        allow_single_node_total_memory=True,
    )

    instance = next(iter(placements.values()))
    assert list(instance.shard_assignments.node_to_runner) == [selected_node]


def test_filtered_single_node_placement_still_rejects_over_capacity_node() -> None:
    topology = Topology()
    selected_node = NodeId()
    other_node = NodeId()
    topology.add_node(selected_node)
    topology.add_node(other_node)
    topology.add_connection(
        Connection(
            source=selected_node, sink=other_node, edge=create_socket_connection(1)
        )
    )
    topology.add_connection(
        Connection(
            source=other_node, sink=selected_node, edge=create_socket_connection(2)
        )
    )
    node_memory = {
        selected_node: create_node_memory_with_total(available=1000, total=1200),
        other_node: create_node_memory_with_total(available=2000, total=2000),
    }
    node_network = {
        selected_node: create_node_network(),
        other_node: create_node_network(),
    }
    command = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_bytes(1500),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            allowed_nodes={selected_node},
            allow_single_node_total_memory=True,
        )


def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_placement_uses_leaf_nodes_as_tie_breaker(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()

    model_card = model_card.model_copy(update={"storage_size": Memory.from_bytes(1000)})

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()

    node_memory = {
        node_id_a: create_node_memory(500),
        node_id_b: create_node_memory(500),
        node_id_c: create_node_memory(500),
        node_id_d: create_node_memory(500),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
        node_id_d: create_node_network(),
    }

    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_node(node_id_d)

    # Daisy chain topology (directed)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_a, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_c, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_d, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_d, sink=node_id_c, edge=create_socket_connection(1))
    )

    cic = place_instance_command(model_card=model_card)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance = list(placements.values())[0]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == set((node_id_a, node_id_b)) or assigned_nodes == set(
        (
            node_id_c,
            node_id_d,
        )
    )


def test_tensor_rdma_backend_connectivity_matrix(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "n_layers": 12,
            "storage_size": Memory.from_bytes(1500),
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(500),
    }

    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.0.1"),
        node_b: create_jaccl_node_network("192.168.0.2"),
        node_c: create_jaccl_node_network("192.168.0.5"),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    # RDMA connections (directed)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_c, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_b, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=create_rdma_connection(5))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_a, edge=create_rdma_connection(5))
    )

    # Ethernet connections (directed)
    topology.add_connection(Connection(source=node_a, sink=node_b, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_a, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_b, edge=ethernet_conn))

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    node_rdma_ctl = {
        node_a: NodeRdmaCtlStatus(enabled=True),
        node_b: NodeRdmaCtlStatus(enabled=True),
        node_c: NodeRdmaCtlStatus(enabled=True),
    }

    # act
    placements = place_instance(
        cic,
        topology,
        {},
        node_memory,
        node_network,
        node_rdma_ctl=node_rdma_ctl,
    )

    # assert
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
            ip_part = coordinator.split(":")[0]
            assert len(ip_part.split(".")) == 4


def test_qwen3_5_tensor_auto_upgrade_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = Topology()
    large_node = NodeId()
    small_node = NodeId()
    topology.add_node(large_node)
    topology.add_node(small_node)
    topology.add_connection(
        Connection(source=large_node, sink=small_node, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=small_node, sink=large_node, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=large_node, sink=small_node, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=small_node, sink=large_node, edge=create_socket_connection(2))
    )

    model_card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-72B-8bit"),
        storage_size=Memory.from_bytes(130_648_036_320),
        n_layers=48,
        hidden_size=3072,
        num_key_value_heads=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        base_model="Qwen3.5 72B",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )
    node_rdma_ctl = {
        large_node: NodeRdmaCtlStatus(enabled=True),
        small_node: NodeRdmaCtlStatus(enabled=True),
    }

    placements_without_opt_in = place_instance(
        command,
        topology,
        {},
        {
            large_node: create_node_memory(128_000_000_000),
            small_node: create_node_memory(48_000_000_000),
        },
        {
            large_node: create_jaccl_node_network("192.168.0.1"),
            small_node: create_jaccl_node_network("192.168.0.2"),
        },
        node_rdma_ctl=node_rdma_ctl,
    )
    instance_without_opt_in = next(iter(placements_without_opt_in.values()))
    large_runner_without_opt_in = (
        instance_without_opt_in.shard_assignments.node_to_runner[large_node]
    )
    large_shard_without_opt_in = (
        instance_without_opt_in.shard_assignments.runner_to_shard[
            large_runner_without_opt_in
        ]
    )
    assert not isinstance(large_shard_without_opt_in, AsymmetricTensorShardMetadata)

    monkeypatch.setenv("EXO_ENABLE_ASYMMETRIC_TP_AUTO_UPGRADE", "1")

    placements = place_instance(
        command,
        topology,
        {},
        {
            large_node: create_node_memory(128_000_000_000),
            small_node: create_node_memory(48_000_000_000),
        },
        {
            large_node: create_jaccl_node_network("192.168.0.1"),
            small_node: create_jaccl_node_network("192.168.0.2"),
        },
        node_rdma_ctl=node_rdma_ctl,
    )

    instance = next(iter(placements.values()))
    large_runner = instance.shard_assignments.node_to_runner[large_node]
    small_runner = instance.shard_assignments.node_to_runner[small_node]
    large_shard = instance.shard_assignments.runner_to_shard[large_runner]
    small_shard = instance.shard_assignments.runner_to_shard[small_runner]

    assert isinstance(large_shard, AsymmetricTensorShardMetadata)
    assert isinstance(small_shard, AsymmetricTensorShardMetadata)
    assert large_shard.device_rank == 0
    assert small_shard.device_rank == 1
    assert large_shard.ratio == small_shard.ratio == 0.75


def test_qwen3_5_tensor_auto_upgrade_ignores_non_two_node_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = Topology()
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_c, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_a, edge=create_socket_connection(3))
    )

    model_card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-72B-8bit"),
        storage_size=Memory.from_bytes(140_000_000_000),
        n_layers=48,
        hidden_size=3072,
        num_key_value_heads=6,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        base_model="Qwen3.5 72B",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=3,
    )

    monkeypatch.setenv("EXO_ENABLE_ASYMMETRIC_TP_AUTO_UPGRADE", "1")

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_id_a: create_node_memory(128_000_000_000),
            node_id_b: create_node_memory(128_000_000_000),
            node_id_c: create_node_memory(48_000_000_000),
        },
        {
            node_id_a: create_node_network(),
            node_id_b: create_node_network(),
            node_id_c: create_node_network(),
        },
    )

    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 3
    assert all(
        not isinstance(shard, AsymmetricTensorShardMetadata)
        for shard in instance.shard_assignments.runner_to_shard.values()
    )


def test_asymmetric_tensor_rejects_unreachable_largest_rank_zero() -> None:
    topology = Topology()
    large_node = NodeId()
    small_node = NodeId()
    topology.add_node(large_node)
    topology.add_node(small_node)
    topology.add_connection(
        Connection(source=large_node, sink=small_node, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=small_node, sink=large_node, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=large_node, sink=small_node, edge=create_socket_connection(3))
    )

    model_card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-72B-8bit"),
        storage_size=Memory.from_bytes(130_648_036_320),
        n_layers=48,
        hidden_size=3072,
        num_key_value_heads=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        base_model="Qwen3.5 72B",
    )
    command = PlaceInstance(
        sharding=Sharding.AsymmetricTensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="rank-0 node socket-reachable"):
        place_instance(
            command,
            topology,
            {},
            {
                large_node: create_node_memory(128_000_000_000),
                small_node: create_node_memory(48_000_000_000),
            },
            {
                large_node: create_node_network(),
                small_node: create_node_network(),
            },
        )


def test_asymmetric_tensor_rejects_qwen3_5_with_unsplittable_kv_heads() -> None:
    topology = Topology()
    large_node = NodeId()
    small_node = NodeId()
    topology.add_node(large_node)
    topology.add_node(small_node)
    topology.add_connection(
        Connection(source=large_node, sink=small_node, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=small_node, sink=large_node, edge=create_socket_connection(2))
    )

    model_card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-122B-A10B-8bit"),
        storage_size=Memory.from_bytes(130_648_036_320),
        n_layers=48,
        hidden_size=3072,
        num_key_value_heads=2,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        base_model="Qwen3.5 122B A10B",
    )
    command = PlaceInstance(
        sharding=Sharding.AsymmetricTensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="No valid asymmetric ratio"):
        place_instance(
            command,
            topology,
            {},
            {
                large_node: create_node_memory(128_000_000_000),
                small_node: create_node_memory(48_000_000_000),
            },
            {
                large_node: create_node_network(),
                small_node: create_node_network(),
            },
        )


def test_asymmetric_tensor_rejects_unsupported_model_family(
    model_card: ModelCard,
) -> None:
    topology = Topology()
    node_id_a = NodeId()
    node_id_b = NodeId()
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_a, edge=create_socket_connection(2))
    )
    command = PlaceInstance(
        sharding=Sharding.AsymmetricTensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="Supported: Qwen3.5"):
        place_instance(
            command,
            topology,
            {},
            {
                node_id_a: create_node_memory(2_000_000),
                node_id_b: create_node_memory(2_000_000),
            },
            {
                node_id_a: create_node_network(),
                node_id_b: create_node_network(),
            },
        )


def test_ring_placement_uses_advertised_lan_ips_for_rdma_only_topology(
    model_card: ModelCard,
) -> None:
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
        }
    )

    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en9", ip_address="192.168.1.10", interface_type="ethernet"
                )
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en9", ip_address="192.168.1.11", interface_type="ethernet"
                )
            ]
        ),
    }

    command = place_instance_command(model_card)
    command = command.model_copy(update={"min_nodes": 2})

    placements = place_instance(command, topology, {}, node_memory, node_network)

    instance = list(placements.values())[0]
    assert isinstance(instance, MlxRingInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2
    assert any(host.ip == "192.168.1.11" for host in instance.hosts_by_node[node_a])
    assert any(host.ip == "192.168.1.10" for host in instance.hosts_by_node[node_b])


def test_jaccl_placement_uses_advertised_lan_ip_for_rdma_coordinator(
    model_card: ModelCard,
) -> None:
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_jaccl_node_network("192.168.0.1", "192.168.1.10"),
        node_b: create_jaccl_node_network("192.168.0.2", "192.168.1.11"),
    }
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )
    node_rdma_ctl = {
        node_a: NodeRdmaCtlStatus(enabled=True),
        node_b: NodeRdmaCtlStatus(enabled=True),
    }

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_rdma_ctl=node_rdma_ctl,
    )

    instance = list(placements.values())[0]
    assert isinstance(instance, MlxJacclInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2
    assert any(
        coordinator.startswith("192.168.1.")
        for coordinator in instance.jaccl_coordinators.values()
    )


def test_jaccl_placement_skips_thunderbolt_preflight_for_single_node_fallback(
    model_card: ModelCard,
) -> None:
    """A ``MlxJaccl`` request with ``min_nodes=1`` on a singleton cycle
    must downgrade to ``MlxRing`` instead of failing the JACCL
    Thunderbolt IPv4 preflight.

    The preflight enforces a multi-node JACCL contract -- every target
    rank must advertise a routable Thunderbolt IPv4 address so the
    JACCL coordinator can dial each peer. A singleton cycle has no
    peers to dial: the placement code immediately downgrades to
    Pipeline / Ring at line ``len(selected_cycle) == 1``. Running the
    preflight before the downgrade means a single-node placement
    request on a host without TB IPv4 (e.g. a developer laptop on
    Wi-Fi) would raise instead of falling back to Ring, breaking
    operator-facing placement previews and any API callers that probe
    JACCL with ``min_nodes=1``.

    Cluster shape: a single node with only a Wi-Fi interface (no TB
    IPv4). Pre-fix this raised the ``bb rdma repair all`` ValueError;
    post-fix the placement returns a single ``MlxRingInstance``.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(800),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    solo_node = NodeId()
    topology.add_node(solo_node)

    node_network = {
        solo_node: NodeNetworkInfo(
            interfaces=[
                # No Thunderbolt and no maybe_ethernet bridge -- only
                # Wi-Fi. Pre-fix this passed all the upstream checks
                # (no peers, so no RDMA edges to demand) and then hit
                # the preflight, which rejected it.
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {solo_node: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={solo_node: NodeRdmaCtlStatus(enabled=True)},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    # The downgrade-to-ring branch fires because the cycle has length
    # 1; the JACCL preflight is skipped because the request can no
    # longer be a JACCL placement at this point.
    assert isinstance(instance, MlxRingInstance)


def test_jaccl_placement_accepts_maybe_ethernet_thunderbolt_bridge(
    model_card: ModelCard,
) -> None:
    """JACCL preflight accepts ``maybe_ethernet`` interfaces with
    routable IPv4 addresses, not only literal ``"thunderbolt"``.

    On every cluster machine we ship, the Thunderbolt bridge sits on
    ``en2`` / ``en3`` / ``en4``, and ``system_info._get_interface_types_from_networksetup``
    reclassifies any ``en*`` adapter that isn't ``en0`` / ``en1`` to
    ``"maybe_ethernet"`` regardless of what ``networksetup`` reports
    the hardware port as. Restricting the preflight to
    ``interface_type == "thunderbolt"`` rejected (correctly repaired)
    Thunderbolt bridges as missing, causing false placement failures
    in real deployments. The upstream RDMA-cycle requirement keeps a
    real LAN ethernet from sneaking past: libp2p only forms RDMA
    edges over Thunderbolt on Apple Silicon, so a node reaching this
    branch with ``maybe_ethernet`` must already have a TB hardware
    link.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                # Real-world setup: Thunderbolt bridge at en3 with a
                # routable IPv4. ``system_info`` reclassifies en2+ to
                # ``maybe_ethernet`` even when ``networksetup`` reports
                # the hardware port as Thunderbolt.
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.10",
                    interface_type="maybe_ethernet",
                )
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.11",
                    interface_type="maybe_ethernet",
                )
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)


def test_jaccl_placement_requires_repaired_thunderbolt_ipv4_paths(
    model_card: ModelCard,
) -> None:
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en1",
                    ip_address="169.254.1.10",
                    interface_type="thunderbolt",
                )
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en1",
                    ip_address="169.254.1.11",
                    interface_type="thunderbolt",
                )
            ]
        ),
    }
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair all"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )


def test_jaccl_placement_falls_back_to_eligible_cycle_when_another_cycle_has_invalid_path(
    model_card: ModelCard,
) -> None:
    """Mixed clusters where one node still lacks a valid Thunderbolt
    IPv4 path must not block placement: as long as at least one
    candidate RDMA cycle of the smallest size has every node on a
    routable JACCL TB IPv4, placement should pick that cycle and
    succeed.

    Pre-fix the preflight ran AFTER ``selected_cycle = max(...)`` had
    already chosen a cycle, so a higher-memory or higher-download cycle
    that happened to contain one unrepaired node would propagate to the
    post-selection check and raise -- even when another size-2 cycle
    of equal class was perfectly valid. This test stages exactly that
    shape.

    Cluster shape::

        node_a (good TB) <-> node_b (good TB)            <- valid cycle
        node_c (good TB) <-> node_d (169.254-only)       <- invalid cycle

    Both cycles are RDMA-connected size 2. Without the candidate-time
    filter, scoring by ``(download_score, total_memory, has_leaf)``
    could pick either pair; we want to guarantee that the invalid pair
    is *never* selected when a valid one exists. We deliberately bias
    the invalid pair to look more attractive to the scorer (more total
    memory) so the test fails on the legacy code path even when the
    selection happens to be deterministic.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    good_a = NodeId()
    good_b = NodeId()
    bad_c = NodeId()
    bad_d = NodeId()

    for node in (good_a, good_b, bad_c, bad_d):
        topology.add_node(node)

    # Two independent RDMA pairs (cycles of size 2):
    topology.add_connection(
        Connection(source=good_a, sink=good_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=good_b, sink=good_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=bad_c, sink=bad_d, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=bad_d, sink=bad_c, edge=create_rdma_connection(4))
    )

    node_network = {
        good_a: create_jaccl_node_network("192.168.10.1", "10.0.0.1"),
        good_b: create_jaccl_node_network("192.168.10.2", "10.0.0.2"),
        bad_c: create_jaccl_node_network("192.168.10.3", "10.0.0.3"),
        bad_d: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en1",
                    ip_address="169.254.1.99",
                    interface_type="thunderbolt",
                )
            ]
        ),
    }

    # Bias the broken cycle to look more attractive to the scorer
    # (higher total memory). The legacy code path picked by score and
    # then raised on the post-selection preflight; the fix moves the
    # filter upstream so the broken cycle is never selected.
    node_memory = {
        good_a: create_node_memory(1000),
        good_b: create_node_memory(1000),
        bad_c: create_node_memory(2000),
        bad_d: create_node_memory(2000),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_rdma_ctl={
            good_a: NodeRdmaCtlStatus(enabled=True),
            good_b: NodeRdmaCtlStatus(enabled=True),
            bad_c: NodeRdmaCtlStatus(enabled=True),
            bad_d: NodeRdmaCtlStatus(enabled=True),
        },
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    selected_node_ids = set(instance.shard_assignments.node_to_runner.keys())
    # Must pick the all-good pair, not the pair that contains the
    # node with the unrepaired 169.254-only Thunderbolt path.
    assert selected_node_ids == {good_a, good_b}


def test_jaccl_placement_prefers_eligible_cycle_among_multiple_size_2_cycles(
    model_card: ModelCard,
) -> None:
    """Even when *every* size-2 cycle in the smallest-cycles set is
    RDMA-connected, the JACCL Thunderbolt IPv4 preflight must
    short-circuit any cycle whose nodes don't all advertise routable
    JACCL paths. Pre-fix this only happened after selection.

    Cluster shape: two RDMA pairs that share a node. Pair (a,b) has
    valid TB IPv4 on both ends; pair (a,c) is broken on the c side.
    The scorer picks by (download_score, total_memory, has_leaf), and
    we tilt c's memory higher so the pair (a,c) would beat (a,b)
    without the upstream filter.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    for node in (node_a, node_b, node_c):
        topology.add_node(node)

    # Two overlapping RDMA pairs.
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_a, edge=create_rdma_connection(4))
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.10.1", "10.0.0.1"),
        node_b: create_jaccl_node_network("192.168.10.2", "10.0.0.2"),
        node_c: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en1",
                    # Broken node: only 169.254 link-local, no routable
                    # peer path.
                    ip_address="169.254.5.5",
                    interface_type="thunderbolt",
                )
            ]
        ),
    }

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
        # ``node_c`` is fatter so its cycle would otherwise win on
        # total-memory tiebreak.
        node_c: create_node_memory(5000),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
            node_c: NodeRdmaCtlStatus(enabled=True),
        },
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    selected_node_ids = set(instance.shard_assignments.node_to_runner.keys())
    assert selected_node_ids == {node_a, node_b}


def test_placement_prefers_socket_reachable_rank_zero(
    model_card: ModelCard,
) -> None:
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
        }
    )

    listener = NodeId()
    peer = NodeId()

    topology.add_node(listener)
    topology.add_node(peer)
    topology.add_connection(
        Connection(source=listener, sink=peer, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=peer, sink=listener, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=peer, sink=listener, edge=create_socket_connection(10))
    )

    node_memory = {
        listener: create_node_memory(1000),
        peer: create_node_memory(1000),
    }
    node_network = {
        listener: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en9", ip_address="192.168.1.10", interface_type="ethernet"
                )
            ]
        ),
        peer: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en9", ip_address="192.168.1.11", interface_type="ethernet"
                )
            ]
        ),
    }

    command = place_instance_command(model_card)
    command = command.model_copy(update={"min_nodes": 2})

    placements = place_instance(command, topology, {}, node_memory, node_network)

    instance = list(placements.values())[0]
    runner_id = instance.shard_assignments.node_to_runner[listener]
    shard = instance.shard_assignments.runner_to_shard[runner_id]
    assert shard.device_rank == 0


def _build_three_node_rdma_topology() -> tuple[
    Topology, NodeId, NodeId, NodeId, dict[NodeId, NodeNetworkInfo]
]:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    ethernet_interface = NetworkInterfaceInfo(name="en0", ip_address="10.0.0.1")
    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )
    node_network = {
        node_a: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_b: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_c: NodeNetworkInfo(interfaces=[ethernet_interface]),
    }

    for node_id in (node_a, node_b, node_c):
        topology.add_node(node_id)

    for source, sink, iface in (
        (node_a, node_b, 3),
        (node_b, node_a, 3),
        (node_b, node_c, 4),
        (node_c, node_b, 4),
        (node_a, node_c, 5),
        (node_c, node_a, 5),
    ):
        topology.add_connection(
            Connection(source=source, sink=sink, edge=create_rdma_connection(iface))
        )

    for source, sink in (
        (node_a, node_b),
        (node_b, node_c),
        (node_c, node_a),
        (node_a, node_c),
        (node_b, node_a),
        (node_c, node_b),
    ):
        topology.add_connection(
            Connection(source=source, sink=sink, edge=ethernet_conn)
        )

    return topology, node_a, node_b, node_c, node_network


def test_place_mlx_jaccl_rejects_when_a_node_has_rdma_ctl_disabled(
    model_card: ModelCard,
) -> None:
    model_card = model_card.model_copy(
        update={"n_layers": 12, "storage_size": Memory.from_bytes(1500)}
    )
    topology, node_a, node_b, node_c, node_network = _build_three_node_rdma_topology()
    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(500),
    }
    node_rdma_ctl = {
        node_a: NodeRdmaCtlStatus(enabled=True),
        node_b: NodeRdmaCtlStatus(enabled=True),
        node_c: NodeRdmaCtlStatus(enabled=False),
    }
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=3,
    )

    with pytest.raises(
        ValueError, match="Requested RDMA \\(MlxJaccl\\) but no RDMA-connected cycles"
    ):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_rdma_ctl=node_rdma_ctl,
        )


def test_place_mlx_jaccl_rejects_when_node_rdma_ctl_missing(
    model_card: ModelCard,
) -> None:
    model_card = model_card.model_copy(
        update={"n_layers": 12, "storage_size": Memory.from_bytes(1500)}
    )
    topology, node_a, node_b, node_c, node_network = _build_three_node_rdma_topology()
    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(500),
    }
    node_rdma_ctl = {
        node_a: NodeRdmaCtlStatus(enabled=True),
        node_b: NodeRdmaCtlStatus(enabled=True),
    }
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=3,
    )

    with pytest.raises(
        ValueError, match="Requested RDMA \\(MlxJaccl\\) but no RDMA-connected cycles"
    ):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_rdma_ctl=node_rdma_ctl,
        )


def _make_task(
    instance_id: InstanceId,
    status: TaskStatus = TaskStatus.Running,
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        task_status=status,
        instance_id=instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def test_get_transition_events_delete_instance_cancels_running_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Running)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – cancellation event should come before the deletion event
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)
    assert events[1].instance_id == instance_id


def test_get_transition_events_delete_instance_cancels_pending_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Pending)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)


def test_get_transition_events_delete_instance_ignores_completed_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    tasks = {
        t.task_id: t
        for t in [
            _make_task(instance_id, TaskStatus.Complete),
            _make_task(instance_id, TaskStatus.Failed),
            _make_task(instance_id, TaskStatus.TimedOut),
            _make_task(instance_id, TaskStatus.Cancelled),
        ]
    }

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only the InstanceDeleted event, no cancellations
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


def test_get_transition_events_delete_instance_cancels_only_matching_tasks(
    instance: Instance,
):
    # arrange
    instance_id_a = InstanceId()
    instance_id_b = InstanceId()
    current_instances: dict[InstanceId, Instance] = {
        instance_id_a: instance,
        instance_id_b: instance,
    }
    # only delete instance A, keep instance B
    target_instances: dict[InstanceId, Instance] = {instance_id_b: instance}

    task_a = _make_task(instance_id_a, TaskStatus.Running)
    task_b = _make_task(instance_id_b, TaskStatus.Running)
    tasks = {task_a.task_id: task_a, task_b.task_id: task_b}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only task_a should be cancelled
    cancel_events = [e for e in events if isinstance(e, TaskStatusUpdated)]
    delete_events = [e for e in events if isinstance(e, InstanceDeleted)]
    assert len(cancel_events) == 1
    assert cancel_events[0].task_id == task_a.task_id
    assert cancel_events[0].task_status == TaskStatus.Cancelled
    assert len(delete_events) == 1
    assert delete_events[0].instance_id == instance_id_a


def _make_shard_metadata(model_card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


def test_placement_prefers_cycle_with_downloaded_model(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with the model already downloaded."""
    topology = Topology()

    model_card = model_card.model_copy(update={"storage_size": Memory.from_bytes(500)})

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    # No connections between them — two single-node cycles

    shard_meta = _make_shard_metadata(model_card)

    # node_b has the model fully downloaded, node_a does not
    download_status = {
        node_b: [
            DownloadCompleted(
                node_id=node_b,
                shard_metadata=shard_meta,
                total=model_card.storage_size,
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_prefers_cycle_with_higher_download_progress(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with more download progress."""
    topology = Topology()

    model_card = model_card.model_copy(update={"storage_size": Memory.from_bytes(1000)})

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_a: 30% downloaded, node_b: 80% downloaded
    download_status = {
        node_a: [
            DownloadOngoing(
                node_id=node_a,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(300),
                    downloaded_this_session=Memory.from_bytes(300),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
        node_b: [
            DownloadOngoing(
                node_id=node_b,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(800),
                    downloaded_this_session=Memory.from_bytes(800),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_does_not_prefer_cycle_with_failed_download(
    model_card: ModelCard,
) -> None:
    """A failed download should count as 0% — not preferred over a node with no download history."""
    topology = Topology()

    model_card = model_card.model_copy(update={"storage_size": Memory.from_bytes(500)})

    node_a = NodeId()
    node_b = NodeId()

    # node_a has slightly more RAM so it would win on the RAM tiebreaker
    node_memory = {
        node_a: create_node_memory(1001),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_b has a failed download — should not be preferred
    download_status = {
        node_b: [
            DownloadFailed(
                node_id=node_b,
                shard_metadata=shard_meta,
                error_message="connection reset",
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    # node_a should win on RAM tiebreaker since failed download scores 0.0
    assert assigned_nodes == {node_a}


# ----------------------------------------------------------------------
# _is_routable_jaccl_ipv4 - octet validation
# ----------------------------------------------------------------------


def test_is_routable_jaccl_ipv4_accepts_valid_thunderbolt_ranges() -> None:
    """Common Thunderbolt-bridge IPv4 ranges we deploy on must pass.

    These are the ranges JACCL preflight is gating on -- a regression
    that rejects any of these would silently disable RDMA placement on
    real clusters.
    """
    for ip in (
        "192.168.10.10",
        "192.168.10.255",
        "10.0.0.1",
        "172.16.0.42",
        "1.2.3.4",
        "223.255.255.254",  # last unicast address before Class D
    ):
        assert _is_routable_jaccl_ipv4(ip), f"{ip} unexpectedly rejected"


def test_is_routable_jaccl_ipv4_rejects_non_unicast_ranges() -> None:
    """Multicast (224..239), reserved (240..254), and broadcast (255)
    must be rejected.

    Codex (PR #11 round 3) flagged that ``255.255.255.255`` was
    previously accepted because the syntactic check passed. A
    misconfigured Thunderbolt/``maybe_ethernet`` interface with a
    non-unicast address would otherwise pass preflight and fail
    later during JACCL backend init -- defeating the purpose of
    failing early with actionable guidance.
    """
    for ip in (
        # Multicast 224..239
        "224.0.0.1",
        "239.255.255.255",
        # Reserved 240..254
        "240.0.0.1",
        "254.0.0.1",
        # Limited broadcast 255.255.255.255 (specifically called out
        # by the codex review)
        "255.255.255.255",
        "255.0.0.1",
    ):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_first_octet_zero() -> None:
    """First octet 0 is still rejected by the prefix block."""
    assert not _is_routable_jaccl_ipv4("0.1.2.3")
    assert not _is_routable_jaccl_ipv4("0.0.0.1")


def test_is_routable_jaccl_ipv4_rejects_out_of_range_octets() -> None:
    """Octets outside 0..255 must be rejected.

    Codex (PR #11 round 2) flagged that the previous implementation
    accepted ``"999.1.1.1"`` because it only checked
    ``len(split('.')) == 4``. That let malformed interface data
    pass preflight and reach the JACCL backend, where it fails with
    a far less actionable error.
    """
    for ip in ("999.1.1.1", "256.0.0.1", "1.256.1.1", "1.1.256.1", "1.1.1.256"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_empty_or_missing_octets() -> None:
    """Strings with the right number of dots but empty/missing octets
    must be rejected.

    ``"1..2.3"`` has four split components but the second is empty.
    ``"1.2.3."`` has four components but the last is empty. The old
    implementation accepted both."""
    for ip in ("1..2.3", "1.2.3.", ".1.2.3", "...", ""):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip!r} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_non_digit_octets() -> None:
    """Non-numeric octets must be rejected (letters, signs, hex)."""
    for ip in ("1.2.3.x", "abc.1.2.3", "-1.2.3.4", "1.2.3.-4", "1.2.3.0xff"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip!r} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_leading_zero_octets() -> None:
    """Leading zeros in octets must be rejected.

    ``networksetup`` never emits them and they historically trigger
    octal-style parsing in some libc tools, so we treat them as
    malformed even though numerically valid."""
    for ip in ("01.2.3.4", "1.02.3.4", "1.2.03.4", "1.2.3.04", "001.2.3.4"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip!r} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_wrong_octet_count() -> None:
    """Strings with the wrong number of octets must be rejected."""
    for ip in ("1.2.3", "1.2.3.4.5", "1.2", "1", "1.2.3.4.5.6.7"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip!r} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_link_local_and_loopback() -> None:
    """The existing prefix block (loopback, link-local, all-zero) must
    still be enforced after octet validation tightens."""
    for ip in ("127.0.0.1", "169.254.10.10", "0.0.0.0"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip} unexpectedly accepted"


def test_is_routable_jaccl_ipv4_rejects_ipv6() -> None:
    """IPv6 addresses must be rejected (any colon disqualifies)."""
    for ip in ("::1", "fe80::1", "2001:db8::1"):
        assert not _is_routable_jaccl_ipv4(ip), f"{ip} unexpectedly accepted"


def test_jaccl_placement_singleton_fallback_picks_best_node_regardless_of_tb(
    model_card: ModelCard,
) -> None:
    """Codex P2 (PR #11 round 4): the candidate-time JACCL prefilter
    must NOT restrict singleton cycles, because a ``MlxJaccl`` request
    with ``min_nodes=1`` always downgrades to ``MlxRing`` further down
    (single-node JACCL is meaningless because target ranks have no
    peers to dial over Thunderbolt RDMA). Pre-fix the prefilter
    rejected non-TB nodes from the candidate pool, so the selector
    picked the TB-equipped node even when a non-TB node had more
    available memory or a better download score -- a worse single-node
    placement.

    Cluster shape: two unconnected solo nodes::

        wifi_node  -- only Wi-Fi, more memory
        tb_node    -- Thunderbolt + Ethernet, less memory

    Both are length-1 RDMA cycles (singletons trivially pass
    ``is_rdma_cycle``). Pre-fix, the prefilter eliminated
    ``wifi_node`` (no TB-IPv4) and the selector was forced to pick
    ``tb_node``. Post-fix, the selector sees both candidates and
    picks the higher-memory one.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(800),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    wifi_node = NodeId()
    tb_node = NodeId()
    topology.add_node(wifi_node)
    topology.add_node(tb_node)

    node_network = {
        wifi_node: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
            ]
        ),
        tb_node: create_jaccl_node_network("192.168.10.2", "192.168.1.51"),
    }

    # Bias the wifi-only node to have MORE memory so the selector
    # would pick it if not blocked by the prefilter. Pre-fix the
    # prefilter dropped it from the candidate pool so the selector
    # was forced to pick ``tb_node`` regardless.
    node_memory = {
        wifi_node: create_node_memory(2000),
        tb_node: create_node_memory(1000),
    }

    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_rdma_ctl={
            wifi_node: NodeRdmaCtlStatus(enabled=True),
            tb_node: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    # Must downgrade to ring (singleton placement).
    assert isinstance(instance, MlxRingInstance)
    selected_node_ids = set(instance.shard_assignments.node_to_runner.keys())
    # Must pick the higher-memory node (wifi), not the TB one. Pre-fix
    # the wifi node was eliminated by the JACCL prefilter and the
    # selector was forced to pick the lower-memory TB node.
    assert selected_node_ids == {wifi_node}, (
        "min_nodes=1 placement must consider non-TB candidates because "
        "the singleton fallback downgrades to MlxRing (which doesn't "
        f"need TB-IPv4); got {selected_node_ids!r}"
    )


def test_is_routable_jaccl_ipv4_rejects_unicode_digit_octets() -> None:
    """Codex P3 (PR #11 round 4): ``str.isdigit()`` returns True for
    Unicode digit characters that ``int()`` then rejects. Pre-fix
    these strings reached ``int(octet)`` and raised ``ValueError``,
    aborting placement instead of cleanly returning False.
    """
    # Superscript digits ('\u00b2' = '²', '\u00b9' = '¹') are
    # ``isdigit() == True`` but not parseable by ``int()``.
    # Arabic-Indic digits ('\u0660'..) and bengali digits ('\u09e6')
    # also satisfy ``isdigit()`` but ``int()`` does accept some of
    # them, so the regression we're guarding against is the
    # superscript / fractional / no-base-10-mapping case.
    superscript_two = "\u00b2"
    superscript_three = "\u00b3"
    superscript_one = "\u00b9"
    cases = [
        f"{superscript_one}.2.3.4",
        f"1.{superscript_two}.3.4",
        f"1.2.{superscript_three}.4",
        f"1.2.3.{superscript_one}",
        # Mixed ASCII + superscript (e.g. ``1²``) -- entire octet is
        # rejected because ``isascii()`` fails on the non-ASCII char.
        f"1{superscript_two}.2.3.4",
    ]
    for ip in cases:
        # Must not raise; must return False cleanly.
        assert not _is_routable_jaccl_ipv4(ip), (
            f"unicode-digit octet {ip!r} unexpectedly accepted"
        )


def test_is_routable_jaccl_ipv4_rejects_oversized_octet_strings() -> None:
    """Codex P2 (PR #11 round-(N+8), placement.py): ``int(octet)`` can
    raise ``ValueError`` for very long numeric strings because CPython
    enforces ``sys.set_int_max_str_digits`` (default 4300). Pre-fix the
    function only checked ``isascii()``/``isdigit()`` before calling
    ``int()``, so an input like ``"9" * 4301 + ".1.1.1"`` reached
    ``int(octet)`` and aborted placement preflight rather than
    returning False. The contract for this helper is "never raise on
    malformed network payloads", so all oversized digit strings must
    cleanly return False.
    """
    pathological_octet = "9" * 4301
    cases = [
        f"{pathological_octet}.1.1.1",
        f"1.{pathological_octet}.1.1",
        f"1.1.{pathological_octet}.1",
        f"1.1.1.{pathological_octet}",
        # Just over the IPv4 max-octet width (3 digits) -- still
        # rejected before ``int()`` is reached, before any
        # ``set_int_max_str_digits`` worry.
        "1234.1.1.1",
        "1.1.1.0001",
    ]
    for ip in cases:
        # Must not raise (incl. ``ValueError`` from CPython's
        # int-digit limit); must return False cleanly.
        assert not _is_routable_jaccl_ipv4(ip), (
            f"oversized-octet input {ip[:20]}... unexpectedly accepted"
        )


def test_jaccl_placement_allows_nodes_with_unknown_network_info(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round 5): ``State.node_network`` is populated
    by a best-effort async watcher, so on cold-boot (or after a
    transient ``info_gatherer`` failure) some nodes have no entry in
    the map. Pre-fix the JACCL preflight collapsed
    "no entry in node_network" and "node has interfaces but none are
    Thunderbolt IPv4" into the same negative verdict, blocking
    ``MlxJaccl`` placements on healthy RDMA topologies whenever the
    gatherer hadn't run yet -- with a misleading "run bb rdma repair"
    error. We now treat missing entries as "unknown" and let
    placement proceed; only nodes with positive evidence of a
    non-TB-IPv4 setup are rejected.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    # libp2p establishes both RDMA and Socket edges per direction in
    # real deployments; including the socket edges lets the JACCL
    # coordinator selector resolve a peer IP from topology metadata
    # alone (via ``_find_connection_ip``) when ``node_network`` is
    # empty. This is the realistic cold-boot shape for the regression
    # we're guarding against.
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )

    # ``node_network`` is empty -- simulates the pre-watcher cold-boot
    # window or a transient gatherer failure on both nodes. The RDMA
    # topology is healthy, so placement should proceed.
    node_network: dict[NodeId, NodeNetworkInfo] = {}

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance), (
        "MlxJaccl placement must succeed when node_network has no "
        "entries (best-effort gatherer hasn't reported yet); the "
        "JACCL preflight must distinguish 'unknown' from 'known-no-TB'."
    )


def test_jaccl_placement_allows_nodes_with_unclassified_interface_typing(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+2)): when the upstream
    ``system_info._get_interface_types_from_networksetup`` parse
    fails, ``NodeNetworkInfo.interfaces`` is populated with IPs but
    every entry's ``interface_type`` is ``None``/``"unknown"``. Pre-
    fix this collapsed into ``known_no_path`` and rejected placement
    even though we had no positive evidence of bad config -- the
    gatherer just couldn't classify. Post-fix, this case is treated
    as ``"unknown"`` (permissive) and placement proceeds, leaving
    the JACCL backend to surface a clearer per-link error if the
    IP turns out to be unusable at bind time.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )

    # Both nodes report interfaces but with NO interface_type info
    # (the system_info parser's "we couldn't classify" output writes
    # interface_type="unknown", which is also the field's default).
    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.10",
                    interface_type="unknown",
                ),
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.11",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance), (
        "MlxJaccl placement must succeed when interface typing is "
        "unavailable for every interface (gatherer parse failure "
        "case); without typing data we have no positive evidence of "
        "bad config and must defer to topology-derived RDMA edges."
    )


def test_jaccl_placement_still_rejects_nodes_with_known_non_tb_paths(
    model_card: ModelCard,
) -> None:
    """Sibling regression to the unknown-info test above: when
    ``node_network`` *does* contain an entry for a node and that
    entry has no qualifying Thunderbolt IPv4 interface (e.g. only
    Wi-Fi, or only link-local 169.254 addresses), preflight must
    still reject with the actionable repair-guidance error message.
    Otherwise loosening the preflight to allow ``unknown`` nodes
    would also let through nodes with positive evidence of bad
    network configuration, defeating the purpose of the check.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    # ``node_a`` has a TB-IPv4 path; ``node_b`` has only Wi-Fi
    # (positive evidence of bad config). Placement must reject.
    node_network = {
        node_a: create_jaccl_node_network("192.168.10.10"),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )


def test_jaccl_placement_rejects_nodes_with_only_loopback_unknown_typing(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+12), placement.py:589): the
    round-(N+11) widening of ``_interface_typing_is_missing`` to
    ``any(...)`` was too permissive. ``get_network_interfaces``
    assigns ``"unknown"`` to interfaces not present in
    ``networksetup`` output (loopback, tunnel, etc.), so almost
    every node has at least one unknown interface and the JACCL
    preflight reverted to permissive behavior -- placement could
    proceed even when the only proper-typed candidate interfaces
    were Wi-Fi (no TB).

    Round-(N+12) couples the unknown check with routable-IPv4
    candidacy: an ``"unknown"``-typed loopback (``127.x.x.x``) or
    link-local (``169.254.x.x``) interface no longer triggers the
    permissive branch because :func:`_is_routable_jaccl_ipv4`
    filters them out. So the rejection guard fires again on a node
    whose only proper-typed candidate is Wi-Fi, even when an
    unknown-typed loopback also exists.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.10.10"),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                # Unknown loopback: pre-(N+12) this would have flipped
                # the verdict to "unknown" and bypassed the preflight,
                # but loopback is filtered by _is_routable_jaccl_ipv4
                # so we still classify the node as known_no_path.
                NetworkInterfaceInfo(
                    name="lo0",
                    ip_address="127.0.0.1",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )


def test_jaccl_placement_allows_nodes_with_partial_interface_typing(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+11), placement.py:589): mixed-typing
    case. When a node's network info contains *some* classified
    interfaces (e.g. Wi-Fi) plus *some* unclassified candidates
    (e.g. an ``en3`` Thunderbolt bridge whose
    ``networksetup -listallhardwareports`` line failed to parse and
    fell back to ``"unknown"``), pre-fix ``_interface_typing_is_missing``
    returned ``False`` because not *every* interface was unknown,
    so the verdict collapsed to ``known_no_path`` and the placement
    was rejected with bb-rdma-repair guidance even though the
    unknown interface might be the working TB link.

    Post-fix, *any* unknown interface is enough signal to defer the
    verdict to ``"unknown"`` (permissive). Placement proceeds and
    the JACCL backend surfaces a clearer per-link error if the IP
    turns out to be unusable at bind time.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )

    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.10",
                    interface_type="unknown",
                ),
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.51",
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="en3",
                    ip_address="192.168.10.11",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance), (
        "MlxJaccl placement must succeed when ANY candidate interface "
        "has unknown typing (gatherer parse partial failure case); the "
        "unknown interface might be the working TB link, so we have no "
        "positive evidence of bad config. Pre-fix the verdict was "
        "``known_no_path`` and placement was rejected."
    )


def test_jaccl_placement_allows_bridge0_thunderbolt_with_unknown_typing(
    model_card: ModelCard,
) -> None:
    """Codex P2 (PR #11 round-(N+13), placement.py:578): the
    round-(N+13) narrowing of the unknown-typing fallback to
    ``en\\d+`` was too restrictive. ``info_gatherer`` explicitly
    models the macOS Thunderbolt Bridge as ``bridge0`` (see
    ``utils.info_gatherer.info_gatherer._extract_bridge_services``).
    That device does NOT appear in ``networksetup
    -listallhardwareports``, so it lands here with
    ``interface_type='unknown'`` and a routable IPv4 -- the exact
    scenario this fallback is meant to tolerate. Pre-(N+14) the
    ``en\\d+`` regex rejected ``bridge0``, regressing real
    Thunderbolt-Bridge deployments to the ``bb rdma repair`` error
    even when the bridge was correctly carrying the JACCL path.

    Round-(N+14) widens the candidate regex to ``^(en|bridge)\\d+$``
    so a node whose only proper-typed candidate is Wi-Fi but ALSO
    has an unclassified ``bridge0`` with a routable IPv4 still
    resolves to ``unknown`` (permissive) rather than
    ``known_no_path`` (rejected). The legitimate rejection paths
    are still covered by
    ``test_jaccl_placement_rejects_nodes_with_only_vpn_tunnel_unknown_typing``
    and
    ``test_jaccl_placement_rejects_nodes_with_only_loopback_unknown_typing``.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )

    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                # Thunderbolt Bridge service device: ``info_gatherer``
                # models this as ``bridge0`` and it does not appear in
                # ``networksetup -listallhardwareports`` so it lands
                # here as ``"unknown"`` with the routable IPv4 the
                # JACCL path actually uses.
                NetworkInterfaceInfo(
                    name="bridge0",
                    ip_address="192.168.10.10",
                    interface_type="unknown",
                ),
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.51",
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="bridge0",
                    ip_address="192.168.10.11",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance), (
        "MlxJaccl placement must succeed when the only TB-bridge "
        "candidate is named ``bridge0`` (the canonical macOS "
        "Thunderbolt Bridge device); the round-(N+13) ``en\\d+`` "
        "regex was too narrow and regressed real ``bridge0`` "
        "deployments to ``known_no_path``. Round-(N+14) accepts "
        "``bridge\\d+`` as well."
    )


def test_jaccl_placement_allows_non_zero_bridge_index_thunderbolt(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+15), placement.py:567): the
    round-(N+15) hard-coded ``bridge0`` was too narrow. The
    info_gatherer's
    :func:`exo.utils.info_gatherer.info_gatherer._get_bridge_services`
    and :func:`_find_thunderbolt_bridge` enumerate **arbitrary**
    ``bridgeX`` devices and intersect their member set with the
    Thunderbolt hardware-port device list. A user with multiple
    bridges -- e.g. an existing ``bridge0`` already claimed by
    another service, or a manually-configured second bridge for
    a multi-host TB cable -- can have a real Thunderbolt Bridge
    exposed as ``bridge1``/``bridge2``/etc. Hard-coding
    ``bridge0`` rejected those configurations with the
    ``bb rdma repair`` error even though a perfectly valid TB
    peer path was available.

    Round-(N+16) widens the bridge half of
    :data:`_THUNDERBOLT_CANDIDATE_INTERFACE_NAME` to
    ``bridge[0-9]{1,2}`` (i.e. ``bridge0``..``bridge99``). macOS
    Internet Sharing reserves ``bridge100``+ for NAT/Parallels/
    VirtualBox VM stacks (see ``man 8 bridge``), so this still
    rejects VM-stack bridges (covered by
    ``test_jaccl_placement_rejects_nodes_with_vm_stack_bridges_and_primary_en``)
    while admitting legitimate TB indices below 100.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )

    # Both nodes expose their Thunderbolt Bridge as ``bridge1``
    # (because ``bridge0`` is already claimed elsewhere on each
    # host). The bridge service device does not appear in
    # ``networksetup -listallhardwareports`` so it lands here as
    # ``"unknown"`` with a routable IPv4 -- exactly the scenario
    # the permissive fallback is meant to tolerate.
    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="bridge1",
                    ip_address="192.168.10.10",
                    interface_type="unknown",
                ),
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.51",
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="bridge2",
                    ip_address="192.168.10.11",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
        node_network,
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance), (
        "MlxJaccl placement must succeed when the only TB-bridge "
        "candidates are named ``bridge1``/``bridge2``; the "
        "info_gatherer enumerates arbitrary ``bridgeX`` devices and "
        "matches them by Thunderbolt-member intersection, so any "
        "low-index bridge is a legitimate TB candidate."
    )


def test_jaccl_placement_rejects_nodes_with_vm_stack_bridges_and_primary_en(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+14), placement.py:548): the
    round-(N+14) widening to ``^(en|bridge)\\d+$`` was too broad in
    two distinct ways:

    * ``en0`` and ``en1`` are reserved for Wi-Fi/primary NIC by
      Apple convention, so an unknown-typed ``en0`` on a Wi-Fi
      node could fire the permissive fallback and bypass the
      preflight.
    * Higher bridge indices (``bridge100``/``bridge101`` from
      Parallels Desktop, ``bridge2``+ from VirtualBox/VMware) are
      virtualised networking stacks, NOT Thunderbolt. Admitting
      them as plausible candidates re-opened the same Wi-Fi-only-
      on-VPN bypass class that round-(N+13)/(N+14) was supposed to
      close.

    Round-(N+15) (this test) narrows the regex to the exact
    Thunderbolt-naming convention: ``en[2-9]`` / ``en[1-9]\\d+``
    (excluding ``en0``/``en1``) and ``bridge0`` only. A node whose
    only ``"unknown"``-typed interfaces are ``en0`` (Wi-Fi primary)
    plus ``bridge100`` (Parallels VM bridge) -- both with routable
    IPv4 -- is now correctly classified as ``known_no_path`` and
    placement is rejected with the actionable ``bb rdma repair``
    error.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.10.10"),
        node_b: NodeNetworkInfo(
            interfaces=[
                # Wi-Fi primary, properly typed -- this prevents the
                # "all unknown" fallback in
                # ``_interface_typing_is_missing`` from firing, so
                # the verdict depends on the plausibility check
                # below.
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                # Parallels Desktop VM bridge -- a virtualised
                # networking stack, NOT Thunderbolt. Pre-(N+15)
                # the ``^(en|bridge)\\d+$`` regex admitted this as
                # a plausible candidate and re-opened the bypass.
                NetworkInterfaceInfo(
                    name="bridge100",
                    ip_address="10.211.55.2",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )


def test_jaccl_placement_rejects_nodes_with_unknown_en0_and_typed_wifi(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+14), placement.py:548) follow-up:
    in addition to ``bridge\\d+`` for ``\\d>0``, the
    round-(N+14) regex also admitted ``en0`` and ``en1`` -- which
    by Apple convention are Wi-Fi/primary NIC, NOT Thunderbolt.
    Round-(N+15) restricts the ``en`` arm to ``en[2-9]`` /
    ``en[1-9]\\d+`` to mirror the ``maybe_ethernet``
    reclassification convention in
    ``info_gatherer.system_info._get_interface_types_from_networksetup``.

    This test pins the ``en0``-bypass scenario: a node with a
    Thunderbolt-typed ``en1`` (test fixture treats ``en1`` as the
    TB leaf for legacy reasons) AND an ``"unknown"``-typed ``en0``
    with a routable IPv4. Pre-fix the unknown ``en0`` matched the
    regex and fired the permissive branch even though the node had
    a real TB candidate via ``en1`` -- which is fine for this
    case, BUT the same bypass on a Wi-Fi-only node (Wi-Fi typed,
    en0 mistakenly unknown-typed too) would fall through to
    placement instead of ``bb rdma repair``.

    Mirror the realistic failure mode: target node has Wi-Fi
    (``wifi`` typed) plus an unknown-typed ``en0`` with the same
    routable IP as Wi-Fi. ``en0`` post-(N+15) no longer matches
    the candidate regex, so the unknown-typing fallback does not
    fire, and placement is rejected with the expected error.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.10.10"),
        node_b: NodeNetworkInfo(
            interfaces=[
                # Wi-Fi primary advertised on a different name (e.g.
                # ``en2`` typed as ``"wifi"`` -- which never
                # happens in practice but ensures the test
                # doesn't conflate the typed-en0 vs unknown-en0
                # cases).
                NetworkInterfaceInfo(
                    name="en2",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                # ``en0`` mistakenly typed as unknown (e.g. brief
                # ``networksetup`` parse hiccup). Pre-(N+15) the
                # ``^(en|bridge)\\d+$`` regex matched ``en0`` and
                # the routable IPv4 fired the permissive branch.
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.51",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )


def test_jaccl_placement_rejects_nodes_with_only_vpn_tunnel_unknown_typing(
    model_card: ModelCard,
) -> None:
    """Codex P1 (PR #11 round-(N+12) follow-up, placement.py:597):
    the round-(N+12) "couple unknown-typing with routable IPv4"
    refinement was still too permissive. ``get_network_interfaces``
    assigns ``"unknown"`` to interfaces missing from
    ``networksetup -listallhardwareports``, which matches every
    macOS VPN/tunnel adapter (``utun*`` for Tailscale/Wireguard,
    ``tun*`` / ``tap*`` for OpenVPN, ``ipsec*`` for IPsec, etc.).
    Those tunnels typically advertise routable ``10.x``/``100.x``
    IPv4 addresses, so the round-(N+12) ``unknown`` + routable-IPv4
    combo still fired on Wi-Fi-only nodes that happened to be on a
    Tailscale tailnet -- the JACCL preflight was bypassed and
    placement progressed to a runtime JACCL failure instead of the
    intended early ``bb rdma repair`` error.

    Round-(N+13) further restricts the permissive fallback to the
    Apple ``en\\d+`` naming convention via
    :func:`_is_plausible_thunderbolt_candidate`. Tunnel adapters
    (``utun3``, ``wg0``, ``tun0``) and Apple Wireless Direct Link
    (``awdl0``) all fail the name check, so this Wi-Fi-only +
    Tailscale node correctly resolves to ``known_no_path`` and the
    placement is rejected with the actionable ``bb rdma repair``
    error. The legitimate Thunderbolt-bridge case (``en3`` with a
    routable IPv4 whose hardware-port line failed to parse) is
    still covered by
    ``test_jaccl_placement_allows_nodes_with_partial_interface_typing``.
    """
    topology = Topology()
    model_card = model_card.model_copy(
        update={
            "storage_size": Memory.from_bytes(1500),
            "n_layers": 12,
            "hidden_size": 32,
            "num_key_value_heads": 8,
            "supports_tensor": True,
        }
    )

    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(2))
    )

    node_network = {
        node_a: create_jaccl_node_network("192.168.10.10"),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.1.50",
                    interface_type="wifi",
                ),
                # Tailscale tunnel: ``utun*`` is unknown-typed AND
                # has a routable ``100.x`` IPv4. Pre-(N+13) this
                # tripped the permissive branch; post-fix the
                # ``en\\d+`` name check rejects it.
                NetworkInterfaceInfo(
                    name="utun3",
                    ip_address="100.67.7.42",
                    interface_type="unknown",
                ),
                # WireGuard / OpenVPN tunnel: same trap, different
                # naming convention. The plausibility check should
                # still reject ``wg0``.
                NetworkInterfaceInfo(
                    name="wg0",
                    ip_address="10.0.0.5",
                    interface_type="unknown",
                ),
            ]
        ),
    }

    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="bb rdma repair"):
        place_instance(
            command,
            topology,
            {},
            {node_a: create_node_memory(1000), node_b: create_node_memory(1000)},
            node_network,
            node_rdma_ctl={
                node_a: NodeRdmaCtlStatus(enabled=True),
                node_b: NodeRdmaCtlStatus(enabled=True),
            },
        )
