"""FLASH plugin placement logic."""

from collections.abc import Mapping
from copy import deepcopy

from loguru import logger

from exo.plugins.implementations.flash.types import FLASHInstance, LaunchFLASH
from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.topology import SocketConnection
from exo.shared.types.worker.instances import BaseInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def place_flash_instance(
    command: LaunchFLASH,
    topology: Topology,
    current_instances: Mapping[InstanceId, BaseInstance],
) -> dict[InstanceId, BaseInstance]:
    """Place a FLASH simulation instance across available nodes.

    Unlike MLX instances which use ring/JACCL topology for tensor parallelism,
    FLASH instances use MPI for communication. We just need to provide the
    node IPs so the runner can generate an MPI hostfile.
    """
    instance_id = InstanceId()
    target_instances: dict[InstanceId, BaseInstance] = dict(deepcopy(current_instances))

    all_nodes = list(topology.list_nodes())

    if len(all_nodes) < command.min_nodes:
        raise ValueError(
            f"Not enough nodes: need {command.min_nodes}, have {len(all_nodes)}"
        )

    # Select nodes (take the first min_nodes)
    selected_nodes = all_nodes[: command.min_nodes]

    logger.info(
        f"Placing FLASH instance '{command.simulation_name}' on {len(selected_nodes)} nodes"
    )

    # Build shard assignments (one runner per node for FLASH)
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    # Create a dummy ModelCard for FLASH (required by ShardMetadata interface)
    flash_model_card = ModelCard(
        model_id=ModelId(command.simulation_name),
        storage_size=Memory(in_bytes=0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[],
    )

    for i, node_id in enumerate(selected_nodes):
        runner_id = RunnerId()
        node_to_runner[node_id] = runner_id
        runner_to_shard[runner_id] = PipelineShardMetadata(
            device_rank=i,
            world_size=len(selected_nodes),
            model_card=flash_model_card,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )

    shard_assignments = ShardAssignments(
        model_id=ModelId(command.simulation_name),
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    # Build hosts_by_node - get hostnames/IPs for MPI hostfile generation
    hosts_by_node: dict[NodeId, list[Host]] = {}

    # If explicit hosts are provided, use them directly
    if command.hosts:
        explicit_hosts = [h.strip() for h in command.hosts.split(",") if h.strip()]
        logger.info(f"FLASH placement: explicit hosts provided: {explicit_hosts}")
        for i, node_id in enumerate(selected_nodes):
            if i < len(explicit_hosts):
                hosts_by_node[node_id] = [Host(ip=explicit_hosts[i], port=0)]
                logger.info(
                    f"FLASH placement: node {node_id} (rank {i}) -> IP {explicit_hosts[i]}"
                )
            else:
                logger.warning(
                    f"Not enough hosts provided for node {i}, using localhost"
                )
                hosts_by_node[node_id] = [Host(ip="127.0.0.1", port=0)]
        logger.info(
            f"FLASH placement: coordinator will be rank 0 at IP {explicit_hosts[0]}"
        )
    else:
        # Try to get IPs from topology edges
        for node_id in selected_nodes:
            node_hosts: list[Host] = []

            # Get IP from outgoing edges (connections to other nodes via mDNS discovery)
            for conn in topology.out_edges(node_id):
                if isinstance(conn.edge, SocketConnection):
                    # Extract IP from multiaddr
                    ip = conn.edge.sink_multiaddr.ip_address
                    # Skip link-local and localhost addresses
                    if not ip.startswith("169.254.") and not ip.startswith("127."):
                        node_hosts.append(Host(ip=ip, port=0))
                        break

            # Last resort: use localhost (will only work for single-node)
            if not node_hosts:
                logger.warning(
                    f"Could not determine IP for node {node_id}, using localhost"
                )
                node_hosts.append(Host(ip="127.0.0.1", port=0))

            hosts_by_node[node_id] = node_hosts

    total_ranks = len(selected_nodes) * command.ranks_per_node

    # Determine coordinator IP - first node's first host IP
    first_node_id: NodeId = next(iter(hosts_by_node.keys()))
    coordinator_ip: str = (
        hosts_by_node[first_node_id][0].ip
        if hosts_by_node[first_node_id]
        else "127.0.0.1"
    )

    target_instances[instance_id] = FLASHInstance(
        instance_id=instance_id,
        shard_assignments=shard_assignments,
        hosts_by_node=hosts_by_node,
        flash_executable_path=command.flash_executable_path,
        parameter_file_path=command.parameter_file_path,
        working_directory=command.working_directory,
        ranks_per_node=command.ranks_per_node,
        total_ranks=total_ranks,
        simulation_name=command.simulation_name,
        coordinator_ip=coordinator_ip,
    )

    logger.info(f"Created FLASH instance {instance_id} with {total_ranks} total ranks")

    return target_instances
