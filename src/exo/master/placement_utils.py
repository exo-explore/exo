from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import InterfaceType, MemoryUsage, NodeNetworkInfo
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


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Memory:
    total_memory = sum(
        (node_memory[node_id].ram_available for node_id in node_ids),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _validate_layer_allocations(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    layer_allocations: list[int],
    model_card: ModelCard,
) -> None:
    """Reject an allocation that does not fit in a node's available memory."""
    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )


def _allocate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    node_bandwidth: Mapping[NodeId, int] | None,
) -> list[int]:
    """Split the model's layers across nodes, by bandwidth where it is known.

    Falls back to RAM-proportional allocation when any node is missing bandwidth
    data, so a cluster that has not finished profiling still places instances.
    Both paths are validated against the same memory constraint.
    """
    if node_bandwidth is not None and all(
        node_id in node_bandwidth for node_id in node_ids
    ):
        logger.info("Using bandwidth-aware shard assignment")
        layer_allocations = allocate_layers_by_bandwidth(
            total_layers=model_card.n_layers,
            bandwidths=[float(node_bandwidth[node_id]) for node_id in node_ids],
            layer_capacities=_layer_capacities(node_ids, node_memory, model_card),
        )
    else:
        if node_bandwidth:
            logger.info(
                "Bandwidth data missing for some nodes, "
                "falling back to RAM-proportional assignment"
            )
        total_memory = _compute_total_memory(node_ids, node_memory)
        layer_allocations = allocate_layers_proportionally(
            total_layers=model_card.n_layers,
            memory_fractions=[
                node_memory[node_id].ram_available / total_memory
                for node_id in node_ids
            ],
        )

    _validate_layer_allocations(node_ids, node_memory, layer_allocations, model_card)
    return layer_allocations


def _layer_capacities(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
) -> list[int]:
    """Maximum layers each node can hold, using the same arithmetic as validation.

    ``_validate_layer_allocations`` rejects a node when
    ``storage_size * layers // n_layers > ram_available``, so the capacity is the
    largest layer count that keeps that expression within budget. Deriving both
    from the same equation means an allocation that respects these capacities
    always passes validation.
    """
    storage_bytes = model_card.storage_size.in_bytes
    if storage_bytes <= 0:
        return [model_card.n_layers for _ in node_ids]

    return [
        (node_memory[node_id].ram_available.in_bytes * model_card.n_layers)
        // storage_bytes
        for node_id in node_ids
    ]


def allocate_layers_by_bandwidth(
    total_layers: int,
    bandwidths: list[float],
    layer_capacities: list[int],
) -> list[int]:
    """Distribute layers proportionally to memory bandwidth, capped by capacity.

    Layers are handed out one at a time to whichever node is furthest below its
    bandwidth-proportional target and still has room, so a node that runs out of
    memory spills onto the next-fastest node instead of stalling the allocation.

    Every node receives at least one layer: a node in the cycle with no layers
    would still sit in the ring and pay the hop cost without doing any work.
    """
    n = len(bandwidths)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    total_bandwidth = sum(bandwidths)
    if total_bandwidth <= 0:
        raise ValueError("Cannot allocate layers: total memory bandwidth is 0")

    targets = [total_layers * bandwidth / total_bandwidth for bandwidth in bandwidths]

    # One layer per node up front, then the rest by largest shortfall against target.
    result = [1] * n
    for _ in range(total_layers - n):
        candidates = [i for i in range(n) if result[i] < layer_capacities[i]]
        if not candidates:
            raise ValueError(
                f"Cannot allocate {total_layers} layers across {n} nodes: "
                "every node is at its memory capacity"
            )
        # Highest bandwidth wins ties so the faster node absorbs the odd layer.
        result[
            max(candidates, key=lambda i: (targets[i] - result[i], bandwidths[i]))
        ] += 1

    return result


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_bandwidth: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(
            model_card, cycle, node_memory, node_bandwidth
        )
    else:
        return _get_shard_assignments_for_pure_pipeline(
            model_card, cycle, node_memory, node_bandwidth
        )


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_bandwidth: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    layer_allocations = _allocate_layers(
        pipeline_node_ids, node_memory, model_card, node_bandwidth
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
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
    node_bandwidth: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    layer_allocations = _allocate_layers(
        cycle.node_ids, node_memory, model_card, node_bandwidth
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
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
    node_bandwidth: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
                node_bandwidth=node_bandwidth,
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
) -> Generator[SocketConnection, None, None]:
    """Find all socket connections from node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection


def find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Ring connections prefer the lowest measured probe latency, falling back to
    interface type (thunderbolt first) when latency is unmeasured. RDMA
    coordinator selection prefers ethernet.
    """
    connections = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not connections:
        return None
    # Deduplicate in first-seen order: `min` breaks ties on iteration order, and
    # unmeasured links all tie at infinity, so a set here would make the choice
    # vary with hash seed between master restarts.
    candidate_ips = list(
        dict.fromkeys(
            connection.sink_multiaddr.ip_address for connection in connections
        )
    )
    latency_by_ip: dict[str, float] = {}
    for connection in connections:
        ip_address = connection.sink_multiaddr.ip_address
        if connection.latency_ms is None:
            continue
        known = latency_by_ip.get(ip_address)
        if known is None or connection.latency_ms < known:
            latency_by_ip[ip_address] = connection.latency_ms

    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }

    # Ring should prioritise the fastest connection: measured latency first,
    # then interface type as a tie-break / fallback for unmeasured links.
    if ring:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }
        return min(
            candidate_ips,
            key=lambda ip: (
                latency_by_ip.get(ip, float("inf")),
                priority.get(ip_to_type.get(ip, "unknown"), 2),
            ),
        )

    # RDMA prefers ethernet coordinator
    priority = {
        "ethernet": 0,
        "wifi": 1,
        "unknown": 2,
        "maybe_ethernet": 3,
        "thunderbolt": 4,
    }
    return min(
        candidate_ips,
        key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2),
    )


# Fallback per-hop cost when a reachability probe has not measured the selected
# link yet. The ordering mirrors the ring priority in find_ip_prioritised.
LINK_SECONDS_BY_INTERFACE: dict[InterfaceType, float] = {
    "thunderbolt": 0.0005,
    "maybe_ethernet": 0.0010,
    "ethernet": 0.0015,
    "wifi": 0.0050,
    "unknown": 0.0080,
}


def _hop_seconds(
    node_id: NodeId,
    other_node_id: NodeId,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> float:
    """Measured RTT for the selected ring link, or its interface fallback."""
    ip = find_ip_prioritised(node_id, other_node_id, topology, node_network, ring=True)
    if ip is None:
        return LINK_SECONDS_BY_INTERFACE["unknown"]

    measured_latency_ms = [
        connection.latency_ms
        for connection in _find_connection_ip(node_id, other_node_id, topology)
        if connection.sink_multiaddr.ip_address == ip
        and connection.latency_ms is not None
    ]
    if measured_latency_ms:
        return min(measured_latency_ms) / 1000

    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    for interface in other_network.interfaces:
        if interface.ip_address == ip:
            return LINK_SECONDS_BY_INTERFACE[interface.interface_type]
    return LINK_SECONDS_BY_INTERFACE["unknown"]


def estimate_token_seconds(
    cycle: Cycle,
    model_card: ModelCard,
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_bandwidth: Mapping[NodeId, int],
) -> float:
    """Estimate the time to produce one token on this cycle.

    This is the objective from issue #957: the time for one token is the sum over
    devices of the compute time ``C_i = M * (N_i / N) / B_i`` plus the latency
    ``L_i`` of the hop to the next device.

    The compute term uses the capacity-aware layer allocation that will actually
    be assigned. This matters when a fast node cannot hold its proportional
    share and layers spill onto slower nodes.

    Hop latency uses the reachability probe's measured RTT when available and
    falls back to the selected interface type while measurements are pending.
    The caller only compares cycles with bandwidth measurements for every node.
    ``M`` is taken as the model's storage size, which overstates bytes read per
    token for MoE models by a constant factor that does not affect the ranking.
    """
    node_ids = cycle.node_ids

    layer_allocations = _allocate_layers(
        node_ids, node_memory, model_card, node_bandwidth
    )
    compute_seconds = sum(
        model_card.storage_size.in_bytes
        * layer_count
        / model_card.n_layers
        / node_bandwidth[node_id]
        for node_id, layer_count in zip(node_ids, layer_allocations, strict=True)
    )

    # A single node still runs the ring backend, but talks to nobody.
    hop_seconds = 0.0
    if len(node_ids) > 1:
        hop_seconds = sum(
            _hop_seconds(
                node_id,
                node_ids[(rank + 1) % len(node_ids)],
                topology,
                node_network,
            )
            for rank, node_id in enumerate(node_ids)
        )

    return compute_seconds + hop_seconds


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

            connection_ip = find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
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

        ip = find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
