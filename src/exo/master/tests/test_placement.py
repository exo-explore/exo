from typing import Callable

import pytest

from exo.master.placement import (
    get_instance_placements_after_create,
    get_transition_events,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import CreateInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.instances import Instance, InstanceStatus
from exo.shared.types.worker.runners import ShardAssignments


@pytest.fixture
def topology() -> Topology:
    return Topology()


@pytest.fixture
def instance() -> Instance:
    return Instance(
        instance_id=InstanceId(),
        instance_type=InstanceStatus.Active,
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


def create_instance_command(model_meta: ModelMetadata) -> CreateInstance:
    return CreateInstance(
        command_id=CommandId(),
        model_meta=model_meta,
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

    create_instance_command = CreateInstance(
        command_id=CommandId(),
        model_meta=model_meta,
    )
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
    placements = get_instance_placements_after_create(
        create_instance_command, topology, {}
    )

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
    create_instance_command = CreateInstance(
        command_id=CommandId(),
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = get_instance_placements_after_create(
        create_instance_command, topology, {}
    )

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
    create_instance_command = CreateInstance(
        command_id=CommandId(),
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    placements = get_instance_placements_after_create(
        create_instance_command, topology, {}
    )

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
    create_instance_command = CreateInstance(
        command_id=CommandId(),
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1001),
            pretty_name="Test Model",
            n_layers=10,
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        get_instance_placements_after_create(create_instance_command, topology, {})


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

    create_instance_command = CreateInstance(
        command_id=CommandId(),
        model_meta=model_meta,
    )

    # Act
    placements = get_instance_placements_after_create(create_instance_command, topology, {})

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
