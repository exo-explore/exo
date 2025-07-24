from typing import TypeGuard, cast

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.models import ModelMetadata
from shared.types.profiling import NodePerformanceProfile
from shared.types.topology import Node
from shared.types.worker.common import RunnerId
from shared.types.worker.runners import ShardAssignments
from shared.types.worker.shards import PipelineShardMetadata


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile

def narrow_all_nodes(nodes: list[Node]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)

def filter_cycles_by_memory(cycles: list[list[Node]], required_memory: int) -> list[list[Node]]:
    filtered_cycles: list[list[Node]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_mem = sum(node.node_profile.memory.ram_available for node in cycle)
        if total_mem >= required_memory:
            filtered_cycles.append(cast(list[Node], cycle))
    return filtered_cycles


def get_smallest_cycles(cycles: list[list[Node]]) -> list[list[Node]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]

def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[Node],
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")

    cycle_memory = sum(node.node_profile.memory.ram_available for node in selected_cycle)
    total_layers = model_meta.n_layers
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_layers = round(total_layers * (node.node_profile.memory.ram_available / cycle_memory))
            node_layers = max(1, node_layers)

        runner_id = RunnerId()
        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=len(selected_cycle),
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner
    )

    return shard_assignments
