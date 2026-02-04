from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
):
    """Create shard assignments for pipeline parallel execution.

    Routes to CFG-aware or standard pipeline placement based on model_card.uses_cfg.
    """
    # Determine CFG parallelism topology
    # CFG parallel only for even node counts with CFG models (2+ nodes)
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(model_card, cycle, node_memory)


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution (returns CfgShardMetadata)."""
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")

    cycle_memory = sum(
        (node_memory[node_id].ram_available for node_id in cycle.node_ids),
        start=Memory(),
    )
    if cycle_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")

    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # For CFG parallel, we only need to allocate layers for one pipeline group
    # (both CFG groups run the same layers). Use the first pipeline group's nodes.
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = sum(
        (node_memory[node_id].ram_available for node_id in pipeline_node_ids),
        start=Memory(),
    )

    layer_allocations = allocate_layers_proportionally(
        total_layers=total_layers,
        memory_fractions=[
            node_memory[node_id].ram_available.in_bytes / pipeline_memory.in_bytes
            for node_id in pipeline_node_ids
        ],
    )

    # Validate each pipeline node has sufficient memory for its assigned layers
    # Use integer arithmetic to avoid floating point precision issues
    total_storage_bytes = model_card.storage_size.in_bytes
    for i, node_id in enumerate(pipeline_node_ids):
        node_layers = layer_allocations[i]
        # Integer division then multiply to get conservative estimate
        required_memory = (total_storage_bytes * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available.in_bytes
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory / (1024**3):.2f} GB for {node_layers} layers, "
                f"but only has {available_memory / (1024**3):.2f} GB available"
            )

    # CFG group 0: pipeline ranks in ascending order (0, 1, 2, ...)
    # CFG group 1: pipeline ranks in descending order (reversed)
    # This places both "last stages" as ring neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    for i, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[i]

        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        runner_id = RunnerId()

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=total_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution (returns PipelineShardMetadata)."""
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")

    cycle_memory = sum(
        (node_memory[node_id].ram_available for node_id in cycle.node_ids),
        start=Memory(),
    )
    if cycle_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")

    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layer_allocations = allocate_layers_proportionally(
        total_layers=total_layers,
        memory_fractions=[
            node_memory[node_id].ram_available.in_bytes / cycle_memory.in_bytes
            for node_id in cycle.node_ids
        ],
    )

    # Validate each node has sufficient memory for its assigned layers
    total_storage_bytes = model_card.storage_size.in_bytes
    for i, node_id in enumerate(cycle.node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage_bytes * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available.in_bytes
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory / (1024**3):.2f} GB for {node_layers} layers, "
                f"but only has {available_memory / (1024**3):.2f} GB available"
            )

    for i, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:i])
        node_layers = layer_allocations[i]

        runner_id = RunnerId()

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=total_layers,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
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

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


def _find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Priority: ethernet > wifi > unknown > thunderbolt
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }
    priority = {
        "ethernet": 0,
        "wifi": 1,
        "unknown": 2,
        "maybe_ethernet": 3,
        "thunderbolt": 4,
    }
    return min(ips, key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2))


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = _find_ip_prioritised(n, coordinator, cycle_digraph, node_network)
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
