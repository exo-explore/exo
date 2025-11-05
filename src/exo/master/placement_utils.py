from collections.abc import Generator
from typing import TypeGuard, cast

from loguru import logger
from pydantic import BaseModel

from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.parallelisation_strategy import ParallelisationStrategyType
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)


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


def get_shard_assignments_for_pipeline_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    parallelisation_strategy: ParallelisationStrategyType,
):
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")

    cycle_memory = sum(
        (node.node_profile.memory.ram_available for node in selected_cycle),
        start=Memory(),
    )
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
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
            world_size=world_size,
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers,
            strategy=parallelisation_strategy,
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


def get_shard_assignments_for_tensor_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    parallelisation_strategy: ParallelisationStrategyType,
):
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")

    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node in enumerate(selected_cycle):
        shard = TensorShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
            strategy=parallelisation_strategy,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    parallelisation_strategy: ParallelisationStrategyType,
) -> ShardAssignments:
    match parallelisation_strategy:
        case "auto":
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                parallelisation_strategy=parallelisation_strategy,
            )
        case "pipeline":
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                parallelisation_strategy=parallelisation_strategy,
            )
        case "pipeline_rdma":
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                parallelisation_strategy=parallelisation_strategy,
            )
        case "tensor":
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                parallelisation_strategy=parallelisation_strategy,
            )
        case "tensor_rdma":
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                parallelisation_strategy=parallelisation_strategy,
            )


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


def get_mlx_ibv_devices_matrix(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            # just for debugging for now...
            for connection_ip in _find_connection_ip(node_i, node_j, cycle_digraph):
                interface_name = _find_interface_name_for_ip(connection_ip, node_i)
                logger.info(
                    f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                )

            matrix[i][j] = "rdma_en3"  # TODO: hack, for now it's always en3
            continue

            for connection_ip in _find_connection_ip(node_i, node_j, cycle_digraph):
                # Set the first valid rmda i -> j connection - if there are multiple, we set essentially randomly - this is fine, the connection doesn't appear to have to be bidirectional
                if (
                    interface_name := _find_interface_name_for_ip(
                        connection_ip,
                        node_i,
                    )
                ) is not None:
                    matrix[i][j] = interface_name
                    break
            else:
                raise ValueError(
                    "Current ibv backend requires all-to-all rdma connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeInfo,
    node_j: NodeInfo,
    cycle_digraph: Topology,
) -> Generator[str]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.list_connections():
        if (
            connection.local_node_id == node_j.node_id
            and connection.send_back_node_id == node_i.node_id
            and connection.send_back_multiaddr is not None
        ):
            yield connection.send_back_multiaddr.ip_address


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if node_info.node_profile is None:
        return None

    for interface in node_info.node_profile.network_interfaces:
        logger.info(
            f"Checking interface {interface.name} for IP {interface.ip_address} == {ip_address}: {interface.ip_address == ip_address}"
        )
        if interface.name not in ["en2", "en3", "en4", "en5", "en6", "en7"]:
            continue
        if interface.ip_address == ip_address:
            return f"rdma_{interface.name}"

    return None


def get_mlx_ibv_coordinator(
    selected_cycle: list[NodeInfo],
    coordinator_port: int,
) -> str | None:
    """Get the coordinator address for MLX IBV (rank 0 device).

    Selects a non-thunderbolt IP address from rank 0 node as a heuristic for
    ethernet accessibility. Returns address in format "X.X.X.X:PORT".
    """

    if len(selected_cycle) == 0:
        logger.warning("No nodes in selected cycle, cannot determine coordinator")
        return None

    rank_0_node = selected_cycle[0]
    logger.info(f"Selecting coordinator from rank 0 node: {rank_0_node.node_id}")
    assert rank_0_node.node_profile is not None
    for iface in rank_0_node.node_profile.network_interfaces:
        if iface.name == "en0" and "." in iface.ip_address:
            return f"{iface.ip_address}:{coordinator_port}"

    raise ValueError("No en0 iface found for device")
