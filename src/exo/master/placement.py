import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from loguru import logger

from exo.master.placement_utils import (
    filter_cycles_by_memory,
    get_mlx_ibv_devices_matrix,
    get_mlx_jaccl_coordinators,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CreateInstance,
    DeleteInstance,
    PlaceInstance,
)
from exo.shared.types.events import Event, InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else 52414


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    all_nodes = list(topology.list_nodes())

    logger.info("finding cycles:")
    cycles = topology.get_cycles()
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = list(
        filter(lambda it: len(it) >= command.min_nodes, cycles + singleton_cycles)
    )
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, command.model_meta.storage_size
    )
    if not cycles_with_sufficient_memory:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_tb_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.get_subgraph_from_nodes(cycle).is_thunderbolt_cycle(cycle)
    ]

    if smallest_tb_cycles != []:
        smallest_cycles = smallest_tb_cycles

    cycles_with_leaf_nodes: list[list[NodeInfo]] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node.node_id) for node in cycle)
    ]

    selected_cycle = max(
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles,
        key=lambda cycle: sum(
            (
                node.node_profile.memory.ram_available
                for node in cycle
                if node.node_profile is not None
            ),
            start=Memory(),
        ),
    )

    shard_assignments = get_shard_assignments(
        command.model_meta, selected_cycle, command.sharding
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if len(selected_cycle) == 1:
        logger.warning(
            "You have likely selected ibv for a single node instance; falling back to MlxRing"
        )

        command.instance_meta = InstanceMeta.MlxRing

    # TODO: Single node instances
    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            mlx_ibv_devices = get_mlx_ibv_devices_matrix(
                selected_cycle,
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                selected_cycle,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                ibv_devices=mlx_ibv_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
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
