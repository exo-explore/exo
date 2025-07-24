
from collections.abc import Mapping
from copy import deepcopy
from functools import singledispatch
from typing import Sequence

from master.utils.placement_utils import (
    filter_cycles_by_memory,
    get_shard_assignments,
    get_smallest_cycles,
)
from shared.topology import Topology
from shared.types.events import Event, InstanceCreated, InstanceDeleted
from shared.types.events.commands import CreateInstanceCommand, DeleteInstanceCommand
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import InstanceParams, TypeOfInstance


@singledispatch
def get_instance_placements(
    command: CreateInstanceCommand,
    topology: Topology,
    current_instances: dict[InstanceId, InstanceParams],
) -> dict[InstanceId, InstanceParams]:
    available_models = [current_instances[instance].shard_assignments.model_id for instance in current_instances]
    if command.model_meta.model_id in available_models:
        raise ValueError(f"Instance for {command.model_meta.model_id} already exists")
    
    candidate_cycles = topology.get_cycles()
    cycles = filter_cycles_by_memory(candidate_cycles, command.model_meta.storage_size_kilobytes)
    if not cycles:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles)
    selected_cycle = max(smallest_cycles, key=lambda cycle: sum(node.node_profile.memory.ram_available for node in cycle if node.node_profile is not None))
    
    shard_assignments = get_shard_assignments(command.model_meta, selected_cycle)
    
    instance_id = InstanceId()
    target_instances = deepcopy(current_instances)
    target_instances[instance_id] = InstanceParams(
        shard_assignments=shard_assignments,
        hosts=[]
    )
    return target_instances


@get_instance_placements.register
def _(command: DeleteInstanceCommand, topology: Topology, current_instances: dict[InstanceId, InstanceParams]) -> dict[InstanceId, InstanceParams]:
    target_instances = deepcopy(current_instances)
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceParams],
    target_instances: Mapping[InstanceId, InstanceParams],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance_params in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance_id=instance_id,
                    instance_params=instance_params,
                    instance_type=TypeOfInstance.ACTIVE
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )
    
    return events
