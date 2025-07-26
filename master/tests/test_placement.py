from typing import Callable

import pytest

from master.placement import get_instance_placements, get_transition_events
from shared.topology import Topology
from shared.types.common import CommandId, NodeId
from shared.types.events._events import (
    _EventType,  # pyright: ignore[reportPrivateUsage]
)
from shared.types.events.commands import CreateInstanceCommand
from shared.types.models import ModelMetadata
from shared.types.topology import Connection, Node
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import Instance, InstanceStatus
from shared.types.worker.runners import ShardAssignments


@pytest.fixture
def topology() -> Topology:
    return Topology()

@pytest.fixture
def instance() -> Instance:
    return Instance(
        instance_id=InstanceId(),
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=ShardAssignments(
            model_id="test-model",
            runner_to_shard={},
            node_to_runner={}
        ),
        hosts=[]
    )

@pytest.fixture
def model_meta() -> ModelMetadata:
    return ModelMetadata(
        model_id="test-model",
        storage_size_kilobytes=1000,
        pretty_name="Test Model",
        n_layers=10
    )

def create_instance_command(model_meta: ModelMetadata) -> CreateInstanceCommand:
    return CreateInstanceCommand(
        command_id=CommandId(),
        model_meta=model_meta,
        instance_id=InstanceId(),
    )


@pytest.mark.parametrize("available_memory,total_layers,expected_layers", [
    ((500, 500, 1000), 12, (3, 3, 6)),
    ((500, 500, 500), 12, (4, 4, 4)),
    ((312, 518, 1024), 12, (2, 3, 7))
])
def test_get_instance_placements_create_instance(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    topology: Topology,
    model_meta: ModelMetadata,
    create_node: Callable[[int, NodeId | None], Node],
    create_connection: Callable[[NodeId, NodeId], Connection]
):
    # arrange
    model_meta.n_layers = total_layers
    
    create_instance_command = CreateInstanceCommand(
        command_id=CommandId(),
        model_meta=model_meta,
        instance_id=InstanceId(),
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
    placements = get_instance_placements(create_instance_command, topology, {})

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


def test_get_transition_events_no_change(topology: Topology, instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {
        instance_id: instance
    }
    target_instances = {
        instance_id: instance
    }

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(topology: Topology, instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {
        instance_id: instance
    }

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert events[0].event_type == _EventType.InstanceCreated


def test_get_transition_events_delete_instance(topology: Topology, instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {
        instance_id: instance
    }
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert events[0].event_type == _EventType.InstanceDeleted
    assert events[0].instance_id == instance_id
