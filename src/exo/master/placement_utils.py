from typing import TypeGuard, cast

from pydantic import BaseModel

from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


def narrow_all_nodes(nodes: list[NodeInfo]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)


def filter_cycles_by_memory(
    cycles: list[list[NodeInfo]], required_memory: Memory
) -> list[list[NodeInfo]]:
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_mem = sum(
            (node.node_profile.memory.ram_available for node in cycle), start=Memory()
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def get_smallest_cycles(cycles: list[list[NodeInfo]]) -> list[list[NodeInfo]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")

    cycle_memory = sum(
        (node.node_profile.memory.ram_available for node in selected_cycle),
        start=Memory(),
    )
    total_layers = model_meta.n_layers
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_layers = round(
                total_layers
                * (
                    node.node_profile.memory.ram_available.in_bytes
                    / cycle_memory.in_bytes
                )
            )
            node_layers = max(1, node_layers)

        runner_id = RunnerId()
        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=len(selected_cycle),
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    cycles = cycle_digraph.get_cycles()
    if not cycles:
        return []

    get_thunderbolt = False
    if cycle_digraph.is_thunderbolt_cycle(cycles[0]):
        get_thunderbolt = True

    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        for connection in cycle_digraph.list_connections():
            if (
                connection.local_node_id == current_node.node_id
                and connection.send_back_node_id == next_node.node_id
            ):
                if get_thunderbolt and not connection.is_thunderbolt():
                    continue
                assert connection.send_back_multiaddr is not None
                host = Host(
                    ip=connection.send_back_multiaddr.ip_address,
                    port=connection.send_back_multiaddr.port,
                )
                hosts.append(host)
                break

    return hosts
