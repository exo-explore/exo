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
    LaunchFLASH,
    PlaceInstance,
)
from exo.shared.types.common import Host
from exo.shared.types.events import Event, InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.instances import (
    FLASHInstance,
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


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

    if command.sharding == Sharding.Tensor:
        if not command.model_meta.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_meta.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_meta.hidden_size % len(cycle) == 0
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                f"No tensor sharding found for model with hidden_size {command.model_meta.hidden_size} candidate cycles"
            )
    if command.sharding == Sharding.Pipeline and command.model_meta.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )

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


def place_flash_instance(
    command: LaunchFLASH,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    """Place a FLASH simulation instance across available nodes.

    Unlike MLX instances which use ring/JACCL topology for tensor parallelism,
    FLASH instances use MPI for communication. We just need to provide the
    node IPs so the runner can generate an MPI hostfile.
    """
    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

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
    node_to_runner: dict = {}

    # Create a dummy ModelMetadata for FLASH (required by ShardMetadata interface)
    flash_model_meta = ModelMetadata(
        model_id=ModelId(command.simulation_name),
        pretty_name=f"FLASH: {command.simulation_name}",
        storage_size=Memory(in_bytes=0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
    )

    for i, node_info in enumerate(selected_nodes):
        runner_id = RunnerId()
        node_to_runner[node_info.node_id] = runner_id
        runner_to_shard[runner_id] = PipelineShardMetadata(
            device_rank=i,
            world_size=len(selected_nodes),
            model_meta=flash_model_meta,
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
    hosts_by_node: dict = {}

    # If explicit hosts are provided, use them directly
    if command.hosts:
        explicit_hosts = [h.strip() for h in command.hosts.split(",") if h.strip()]
        logger.info(f"FLASH placement: explicit hosts provided: {explicit_hosts}")
        for i, node_info in enumerate(selected_nodes):
            if i < len(explicit_hosts):
                hosts_by_node[node_info.node_id] = [Host(ip=explicit_hosts[i], port=0)]
                logger.info(
                    f"FLASH placement: node {node_info.node_id} (rank {i}) -> IP {explicit_hosts[i]}"
                )
            else:
                logger.warning(
                    f"Not enough hosts provided for node {i}, using localhost"
                )
                hosts_by_node[node_info.node_id] = [Host(ip="127.0.0.1", port=0)]
        logger.info(
            f"FLASH placement: coordinator will be rank 0 at IP {explicit_hosts[0]}"
        )
    else:
        # Try to get IPs from topology edges
        for node_info in selected_nodes:
            node_hosts: list[Host] = []

            # Get IP from outgoing edges (connections to other nodes via mDNS discovery)
            for _, edge_data in topology.out_edges(node_info.node_id):
                if hasattr(edge_data, "send_back_multiaddr"):
                    # Extract IP from multiaddr like /ip4/192.168.1.100/tcp/52415
                    multiaddr = str(edge_data.send_back_multiaddr)
                    if "/ip4/" in multiaddr:
                        parts = multiaddr.split("/")
                        try:
                            ip_idx = parts.index("ip4") + 1
                            ip = parts[ip_idx]
                            # Skip link-local and localhost addresses
                            if not ip.startswith("169.254.") and not ip.startswith(
                                "127."
                            ):
                                node_hosts.append(Host(ip=ip, port=0))
                                break
                        except (ValueError, IndexError):
                            pass

            # Last resort: use localhost (will only work for single-node)
            if not node_hosts:
                logger.warning(
                    f"Could not determine IP for node {node_info.node_id}, using localhost"
                )
                node_hosts.append(Host(ip="127.0.0.1", port=0))

            hosts_by_node[node_info.node_id] = node_hosts

    total_ranks = len(selected_nodes) * command.ranks_per_node

    # Determine coordinator IP - first node's first host IP
    first_node_id = list(hosts_by_node.keys())[0]
    coordinator_ip = (
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
