import random
from collections.abc import Mapping
from copy import deepcopy
from functools import singledispatch
from typing import Sequence

from exo.master.utils.placement_utils import (
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.topology import Topology
from exo.shared.types.common import Host
from exo.shared.types.events import Event, InstanceCreated, InstanceDeleted
from exo.shared.types.events.commands import (
    CreateInstanceCommand,
    DeleteInstanceCommand,
)
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.instances import Instance, InstanceStatus


def random_ephemeral_port() -> int:
    return random.randint(49152, 65535)


@singledispatch
def get_instance_placements(
    command: CreateInstanceCommand,
    topology: Topology,
    current_instances: dict[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    available_models = [
        current_instances[instance].shard_assignments.model_id
        for instance in current_instances
    ]
    if command.model_meta.model_id in available_models:
        raise ValueError(f"Instance for {command.model_meta.model_id} already exists")

    all_nodes = list(topology.list_nodes())
    cycles = topology.get_cycles()
    # we can also always just have a node on its own
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = cycles + singleton_cycles
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, command.model_meta.storage_size_kilobytes * 1024
    )
    if not cycles_with_sufficient_memory:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)
    selected_cycle = None

    has_thunderbolt_cycle = any(
        [
            topology.get_subgraph_from_nodes(cycle).is_thunderbolt_cycle(cycle)
            for cycle in smallest_cycles
        ]
    )
    if has_thunderbolt_cycle:
        smallest_cycles = [
            cycle
            for cycle in smallest_cycles
            if topology.get_subgraph_from_nodes(cycle).is_thunderbolt_cycle(cycle)
        ]

    selected_cycle = max(
        smallest_cycles,
        key=lambda cycle: sum(
            node.node_profile.memory.ram_available
            for node in cycle
            if node.node_profile is not None
        ),
    )

    shard_assignments = get_shard_assignments(command.model_meta, selected_cycle)

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)
    hosts: list[Host] = get_hosts_from_subgraph(cycle_digraph)

    instance_id = command.instance_id
    target_instances = deepcopy(current_instances)
    target_instances[instance_id] = Instance(
        instance_id=instance_id,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=[
            Host(
                ip=host.ip,
                # NOTE: it's fine to have non-deterministic ports here since this is in a command decision
                port=random_ephemeral_port(),
            )
            for host in hosts
        ],
    )
    return target_instances


@get_instance_placements.register
def _(
    command: DeleteInstanceCommand,
    topology: Topology,
    current_instances: dict[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = deepcopy(current_instances)
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
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
