from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from exo.master.placement_utils import (
    Cycle,
    estimate_token_seconds,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NodeBandwidth,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import Sharding
from exo.utils.ports import random_ephemeral_port

INSTANCE_META_BACKENDS: dict[InstanceMeta, list[Backend]] = {
    InstanceMeta.MlxRing: [Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    InstanceMeta.MlxJaccl: [Backend.MlxMetal],
}


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0–1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def _filter_cycles_for_sharding(
    command: PlaceInstance,
    cycles: list[Cycle],
) -> list[Cycle]:
    """Keep cycles whose width is compatible with the requested sharding.

    Raises if the model cannot be sharded this way at all, or if no candidate
    cycle has a compatible width.
    """
    model_card = command.model_card

    if command.sharding == Sharding.Tensor:
        if not model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        # DeepSeek V4 is MQA (num_key_value_heads=1) but its sharding strategy
        # head-parallelises wq_b/wo_a and shards MoE experts instead of splitting
        # KV heads, so the kv-head divisibility check doesn't apply.
        is_deepseek_v4 = model_card.base_model.startswith("DeepSeek V4")
        kv_heads = model_card.num_key_value_heads
        compatible_cycles = [
            cycle
            for cycle in cycles
            if model_card.hidden_size % len(cycle) == 0
            and (is_deepseek_v4 or kv_heads is None or kv_heads % len(cycle) == 0)
        ]
        if not compatible_cycles:
            raise ValueError(
                f"No tensor sharding found for model with "
                f"hidden_size={model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles"
            )
        return compatible_cycles

    if command.sharding == Sharding.Pipeline:
        if model_card.model_id == ModelId("mlx-community/DeepSeek-V3.1-8bit"):
            raise ValueError(
                "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
            )
        if model_card.base_model.startswith("Gemma 4"):
            single_node_cycles = [cycle for cycle in cycles if len(cycle) == 1]
            if not single_node_cycles:
                raise ValueError(
                    "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
                )
            return single_node_cycles

    return cycles


def _filter_cycles_by_backends(
    command: PlaceInstance,
    cycles: list[Cycle],
    node_backends: Mapping[NodeId, list[Backend]],
) -> list[Cycle]:
    """Keep cycles where every node supports a backend the engine and model share."""
    required_backends = set(INSTANCE_META_BACKENDS[command.instance_meta]) & set(
        command.model_card.backends
    )
    if not required_backends:
        raise ValueError(
            f"Model {command.model_card.model_id} backends "
            f"{sorted(b.value for b in command.model_card.backends)} cannot satisfy engine "
            f"{command.instance_meta.value} which requires "
            f"{sorted(b.value for b in INSTANCE_META_BACKENDS[command.instance_meta])}"
        )

    supported_cycles = [
        cycle
        for cycle in cycles
        if all(
            set(node_backends.get(node_id, [])) & required_backends for node_id in cycle
        )
    ]
    if not supported_cycles:
        raise ValueError(
            f"No cycle where every node supports a backend in "
            f"{sorted(b.value for b in required_backends)} for {command.model_card.model_id}"
        )
    return supported_cycles


def _filter_cycles_by_rdma(
    cycles: list[Cycle],
    topology: Topology,
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus],
) -> list[Cycle]:
    """Keep cycles that are fully RDMA-connected and have rdma_ctl enabled everywhere."""

    def all_rdma_ctl_enabled(cycle: Cycle) -> bool:
        return all(
            ((status := node_rdma_ctl.get(node_id)) is not None and status.enabled)
            for node_id in cycle
        )

    rdma_cycles = [
        cycle
        for cycle in cycles
        if topology.is_rdma_cycle(cycle) and all_rdma_ctl_enabled(cycle)
    ]
    if not rdma_cycles:
        raise ValueError(
            "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
        )
    return rdma_cycles


def _select_cycle(
    command: PlaceInstance,
    cycles: list[Cycle],
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_bandwidth: Mapping[NodeId, int],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Cycle:
    """Choose which of the remaining candidate cycles to place the instance on.

    Fully profiled MLX ring pipeline cycles are ranked by estimated steady-state
    token time. Operational preferences only break ties. Before profiling has
    produced a complete candidate, retain the existing smallest-cycle behavior.
    """
    profiled_cycles = (
        [
            cycle
            for cycle in cycles
            if all(node_bandwidth.get(node_id, 0) > 0 for node_id in cycle)
        ]
        if command.sharding == Sharding.Pipeline
        and command.instance_meta == InstanceMeta.MlxRing
        and not command.model_card.uses_cfg
        else []
    )
    cycle_estimates: list[tuple[Cycle, float]] = []
    for cycle in profiled_cycles:
        try:
            estimate = estimate_token_seconds(
                cycle,
                command.model_card,
                topology,
                node_memory,
                node_network,
                node_bandwidth,
            )
        except ValueError:
            # Total RAM can be sufficient while an individual node cannot hold
            # the mandatory one-layer minimum. Such a cycle is not placeable.
            continue
        cycle_estimates.append((cycle, estimate))

    if cycle_estimates:

        def performance_score(
            cycle_and_estimate: tuple[Cycle, float],
        ) -> tuple[float, bool, float, Memory]:
            cycle, estimate = cycle_and_estimate
            return (
                -estimate,
                any(topology.node_is_leaf(node_id) for node_id in cycle),
                _cycle_download_score(
                    cycle, command.model_card.model_id, download_status
                ),
                sum(
                    (node_memory[node_id].ram_available for node_id in cycle),
                    start=Memory(),
                ),
            )

        return max(cycle_estimates, key=performance_score)[0]

    cycles = get_smallest_cycles(cycles)
    cycles_with_leaf_nodes = [
        cycle
        for cycle in cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]
    candidate_cycles = cycles_with_leaf_nodes or cycles

    def cycle_score(cycle: Cycle) -> tuple[float, Memory]:
        return (
            _cycle_download_score(cycle, command.model_card.model_id, download_status),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        )

    return max(candidate_cycles, key=cycle_score)


def _build_instance(
    command: PlaceInstance,
    instance_id: InstanceId,
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    shard_assignments: ShardAssignments,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> Instance:
    """Construct the engine-specific instance for an already-selected cycle."""
    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                [node_id for node_id in selected_cycle],
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            return MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            return MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_backends: Mapping[NodeId, list[Backend]],
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] | None = None,
    node_bandwidth: Mapping[NodeId, NodeBandwidth] | None = None,
) -> dict[InstanceId, Instance]:
    cycles = [
        cycle for cycle in topology.get_cycles() if len(cycle) >= command.min_nodes
    ]

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        cycles = [cycle for cycle in cycles if required_nodes.issubset(cycle.node_ids)]

    cycles = filter_cycles_by_memory(
        cycles, node_memory, command.model_card.storage_size
    )
    if not cycles:
        raise ValueError("No cycles found with sufficient memory")

    cycles = _filter_cycles_for_sharding(command, cycles)
    cycles = _filter_cycles_by_backends(command, cycles, node_backends)

    if command.instance_meta == InstanceMeta.MlxJaccl:
        cycles = _filter_cycles_by_rdma(cycles, topology, node_rdma_ctl or {})

    memory_bandwidth_by_node = (
        {
            node_id: bandwidth.memory_bandwidth
            for node_id, bandwidth in node_bandwidth.items()
        }
        if node_bandwidth
        else {}
    )
    selected_cycle = _select_cycle(
        command,
        cycles,
        topology,
        node_memory,
        node_network,
        memory_bandwidth_by_node,
        download_status or {},
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command = command.model_copy(
            update={
                "instance_meta": InstanceMeta.MlxRing,
                "sharding": Sharding.Pipeline,
            }
        )

    shard_assignments = get_shard_assignments(
        command.model_card,
        selected_cycle,
        command.sharding,
        node_memory,
        memory_bandwidth_by_node or None,
    )

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))
    target_instances[instance_id] = _build_instance(
        command,
        instance_id,
        selected_cycle,
        topology.get_subgraph_from_nodes(selected_cycle.node_ids),
        shard_assignments,
        node_network,
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
    tasks: Mapping[TaskId, Task],
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
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
