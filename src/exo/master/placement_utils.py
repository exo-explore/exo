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
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


RPC_BASE_PORT: int = 60000


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
    selected_cycle: list[NodeWithProfile],
):
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
    selected_cycle: list[NodeWithProfile],
):
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
    sharding: Sharding,
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )


def _is_valid_external_ip(ip: str) -> bool:
    """Check if an IP address is a valid external (non-loopback) address."""
    if not ip:
        return False
    if ip in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
        return False
    if ip.startswith("127."):
        return False
    # Skip IPv6 addresses (contain colons)
    if ":" in ip:
        return False
    return True


def _is_preferred_network_ip(ip: str) -> bool:
    """Check if IP is on a preferred private network (10.x.x.x, 192.168.x.x, 172.16-31.x.x)."""
    if not _is_valid_external_ip(ip):
        return False
    # Prefer private network ranges commonly used for local clusters
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        # Check for 172.16.0.0 - 172.31.255.255
        try:
            second_octet = int(ip.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def _get_external_ip_from_node_profile(node: NodeInfo) -> str | None:
    """Extract the best external IP from a node's profile.
    
    Prefers private network IPs (10.x, 192.168.x, 172.16-31.x) over other IPs.
    """
    if node.node_profile is None:
        logger.warning(f"Node {node.node_id} has no profile, cannot get IP")
        return None
    
    if not node.node_profile.network_interfaces:
        logger.warning(f"Node {node.node_id} has no network interfaces in profile")
        return None
    
    # Log all interfaces for debugging
    logger.info(f"Node {node.node_id} network interfaces:")
    for iface in node.node_profile.network_interfaces:
        logger.info(f"  - {iface.name}: {iface.ip_address}")
    
    # First pass: look for preferred private network IPs
    for interface in node.node_profile.network_interfaces:
        ip = interface.ip_address
        if _is_preferred_network_ip(ip):
            logger.info(f"Selected preferred IP {ip} from interface {interface.name} for node {node.node_id}")
            return ip
    
    # Second pass: any valid external IP
    for interface in node.node_profile.network_interfaces:
        ip = interface.ip_address
        if _is_valid_external_ip(ip):
            logger.info(f"Selected fallback IP {ip} from interface {interface.name} for node {node.node_id}")
            return ip
    
    logger.warning(f"No valid external IP found for node {node.node_id}")
    return None


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    """Get host addresses for nodes in the topology.
    
    Attempts to find real network IPs (not localhost) for each node.
    Falls back to node profile IPs if connection topology only has localhost.
    """
    cycles = cycle_digraph.get_cycles()
    expected_length = len(list(cycle_digraph.list_nodes()))
    cycles = [cycle for cycle in cycles if len(cycle) == expected_length]
    if not cycles:
        if expected_length > 1:
            logger.warning(
                f"No cycles of length {expected_length} found even though chosen subgraph contained {expected_length} nodes"
            )
        return []

    get_thunderbolt = False
    if cycle_digraph.is_thunderbolt_cycle(cycles[0]):
        get_thunderbolt = True

    logger.info(f"Using thunderbolt cycle: {get_thunderbolt}")

    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        best_host: Host | None = None
        fallback_port: int = 0
        
        # First try: find external IP from connection topology
        for connection in cycle_digraph.list_connections():
            if (
                connection.local_node_id == current_node.node_id
                and connection.send_back_node_id == next_node.node_id
            ):
                if get_thunderbolt and not connection.is_thunderbolt():
                    continue
                assert connection.send_back_multiaddr is not None
                ip = connection.send_back_multiaddr.ip_address
                fallback_port = connection.send_back_multiaddr.port
                
                # Skip loopback addresses - prefer real network IPs
                if not _is_valid_external_ip(ip):
                    logger.debug(f"Skipping localhost IP {ip} from connection {current_node.node_id} -> {next_node.node_id}")
                    continue
                    
                host = Host(
                    ip=ip,
                    port=connection.send_back_multiaddr.port,
                )
                best_host = host
                break
        
        # Second try: if no valid IP from connections, try node profile
        if best_host is None:
            external_ip = _get_external_ip_from_node_profile(next_node)
            if external_ip:
                logger.info(f"Using node profile IP {external_ip} for {next_node.node_id} (connection had localhost)")
                best_host = Host(
                    ip=external_ip,
                    port=fallback_port if fallback_port else 52415,
                )
        
        if best_host is not None:
            hosts.append(best_host)
        else:
            logger.warning(f"Could not find valid external IP for node {next_node.node_id}")

    return hosts


def get_hosts_for_llamacpp(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
) -> list[Host]:
    """Get host addresses specifically for llama.cpp distributed inference.
    
    For llama.cpp RPC, we need the IP addresses that the master (rank 0)
    can use to reach each worker (rank > 0).
    
    Returns a list of Host objects ordered by device_rank.
    """
    hosts: list[Host] = []
    
    if len(selected_cycle) <= 1:
        return hosts
    
    master_node = selected_cycle[0]
    
    # Log all available connections for debugging
    all_connections = list(cycle_digraph.list_connections())
    logger.info(f"=== DEBUG: Building hosts for {len(selected_cycle)} nodes ===")
    logger.info(f"Master node: {master_node.node_id}")
    logger.info(f"Total connections in topology: {len(all_connections)}")
    for conn in all_connections:
        logger.info(
            f"  Connection: {conn.local_node_id} -> {conn.send_back_node_id} "
            f"at {conn.send_back_multiaddr.ip_address}:{conn.send_back_multiaddr.port}"
        )
    
    # Collect all node IPs from node profiles for reference
    node_profile_ips: dict[str, str] = {}
    for node in selected_cycle:
        ip = _get_external_ip_from_node_profile(node)
        if ip:
            node_profile_ips[str(node.node_id)] = ip
            logger.info(f"  Node {node.node_id} profile IP: {ip}")
    
    for rank, node in enumerate(selected_cycle):
        if rank == 0:
            # Master doesn't need a host entry for RPC (it makes outgoing connections)
            # But we include a placeholder to keep indices aligned with device_rank
            hosts.append(Host(ip="0.0.0.0", port=0))
            continue
        
        logger.info(f"=== Finding IP for worker rank {rank}: {node.node_id} ===")
        
        # Find the IP the master can use to reach this worker
        worker_ip: str | None = None
        
        # PRIORITY 1: Use node profile IP (most reliable for Android)
        # Node profiles contain the actual network interface IPs
        worker_ip = _get_external_ip_from_node_profile(node)
        if worker_ip:
            logger.info(f"  [PROFILE] Found worker {rank} IP from node profile: {worker_ip}")
        
        # PRIORITY 2: Look for connection FROM this worker (worker as local_node_id)
        # When worker connects to master, it reports its own external IP
        if worker_ip is None:
            for connection in all_connections:
                if connection.local_node_id == node.node_id:
                    # This is a connection FROM the worker, check what IP others see it at
                    # Actually, we need to find connections where OTHERS connect TO this worker
                    pass  # Skip this approach
            
            # Look for connections where this worker is the destination
            for connection in all_connections:
                if connection.send_back_node_id == node.node_id:
                    ip = connection.send_back_multiaddr.ip_address
                    logger.info(
                        f"  [CONN] Considering connection from {connection.local_node_id}: "
                        f"send_back to {connection.send_back_node_id} at {ip}"
                    )
                    if _is_valid_external_ip(ip):
                        worker_ip = ip
                        logger.info(f"  [CONN] Using IP from connection: {worker_ip}")
                        break
                    else:
                        logger.info(f"  [CONN] Rejected invalid IP: {ip}")
        
        if worker_ip is None:
            logger.warning(
                f"Could not find external IP for worker node {node.node_id} (rank {rank}). "
                "Distributed inference may fail."
            )
            worker_ip = "localhost"  # Fallback, but will likely fail
        
        # All workers use the same port (60000) since they're on different IPs
        rpc_port = RPC_BASE_PORT
        hosts.append(Host(ip=worker_ip, port=rpc_port))
        logger.info(f"=== Worker {rank} ({node.node_id}): FINAL IP = {worker_ip}:{rpc_port} ===")
    
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

            # Find the IP J uses to talk to I
            for connection_ip in _find_connection_ip(node_j, node_i, cycle_digraph):
                # This is a local IP on I, which is attached to an interface: find that interface
                if interface_name := _find_interface_name_for_ip(connection_ip, node_i):
                    matrix[i][j] = interface_name
                    logger.info(
                        f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                    )
                    break
            else:
                logger.warning(
                    f"Failed to find interface name between {node_i.node_id} and {node_j.node_id}"
                )
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
            connection.local_node_id == node_i.node_id
            and connection.send_back_node_id == node_j.node_id
        ):
            yield connection.send_back_multiaddr.ip_address


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if node_info.node_profile is None:
        return None

    logger.info(f"Searching {node_info.node_id} for ip {ip_address}:")
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in ["en2", "en3", "en4", "en5", "en6", "en7"]:
            continue
        logger.info(f" | {interface.name}: {interface.ip_address}")
        if interface.ip_address != ip_address:
            continue

        logger.info("Found")
        return f"rdma_{interface.name}"

    return None


def get_mlx_ibv_coordinators(
    selected_cycle: list[NodeInfo],
    coordinator_port: int,
    cycle_digraph: Topology,
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX IBV (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    rank_0_node = selected_cycle[0]
    logger.info(f"Selecting coordinator from rank 0 node: {rank_0_node.node_id}")

    def get_ip_for_node(n: NodeInfo) -> str:
        if n.node_id == rank_0_node.node_id:
            return "0.0.0.0"

        for ip in _find_connection_ip(n, rank_0_node, cycle_digraph):
            return ip

        logger.warning(
            f"Failed to find directly connected ip between {n.node_id} and {rank_0_node.node_id}"
        )
        raise ValueError("Current ibv backend requires all-to-all rdma connections")

    return {
        n.node_id: f"{get_ip_for_node(n)}:{coordinator_port}" for n in selected_cycle
    }


def get_rpc_ports_for_llamacpp(
    selected_cycle: list[NodeInfo],
) -> dict[NodeId, int]:
    """
    Assign RPC ports for llama.cpp distributed inference.

    Master node (device_rank 0) doesn't need an RPC port (it connects to others).
    Worker nodes (device_rank > 0) all use the same RPC_BASE_PORT (60000) since
    they are on different IPs.

    Returns a dict mapping node_id to RPC port (0 for master node).
    """
    rpc_ports: dict[NodeId, int] = {}

    for device_rank, node in enumerate(selected_cycle):
        if device_rank == 0:
            rpc_ports[node.node_id] = 0
        else:
            # All workers use the same port - they're on different IPs
            rpc_ports[node.node_id] = RPC_BASE_PORT

    return rpc_ports


def get_tensor_split_for_llamacpp(
    selected_cycle: list[NodeInfo],
) -> list[float]:
    """
    Calculate tensor split ratios for llama.cpp distributed inference.

    The tensor_split parameter controls how layers are distributed across devices.
    Ratios are based on available memory on each node.

    Returns a list of floats representing the fraction of layers for each device.
    For single-node instances, returns an empty list.
    """
    if len(selected_cycle) <= 1:
        return []

    if not narrow_all_nodes(selected_cycle):
        # If any node doesn't have a profile, use equal split
        equal_ratio = 1.0 / len(selected_cycle)
        return [equal_ratio for _ in selected_cycle]

    total_memory = sum(
        (node.node_profile.memory.ram_available.in_bytes for node in selected_cycle),
    )

    if total_memory == 0:
        equal_ratio = 1.0 / len(selected_cycle)
        return [equal_ratio for _ in selected_cycle]

    tensor_split: list[float] = []
    for node in selected_cycle:
        ratio = node.node_profile.memory.ram_available.in_bytes / total_memory
        tensor_split.append(round(ratio, 4))

    return tensor_split
