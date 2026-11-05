from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from copy import deepcopy
from datetime import datetime
from typing import Sequence

from exo.master.placement_utils import (
    Cycle,
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
    DownloadCommand,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.placement import PlacementRequest
from exo.shared.types.profiling import (
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
)
from exo.shared.types.state import State
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
    InstanceRemovalReason,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
    ShardAssignments,
)
from exo.shared.types.worker.shards import (
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.utils.ports import random_ephemeral_port
from exo.worker.engines.mlx.models.spec_registry import placement_rules_for_model_type

INSTANCE_META_BACKENDS: dict[InstanceMeta, list[Backend]] = {
    InstanceMeta.MlxRing: [Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    InstanceMeta.MlxJaccl: [Backend.MlxMetal],
}


def launch_refusal(instance: Instance, state: State) -> str | None:
    """Why the master refuses to launch the instance, if it does: a node isn't in the cluster,
    or can't hold its shard besides the instances already there (some perhaps still loading).

    A node outside the cluster: the master's next check would delete the instance at once, as
    one that lost a node. Refused instead: the API sends a launch again until it shows, by when
    a node that is coming back (a split healing, say) is in the cluster again. Memory: two
    launches sent at the same time were otherwise both placed in memory that holds one. The
    API checks a launch the same way before sending it, but against its own state, which may
    not show the other launch yet."""
    if outside := [
        node_id
        for node_id in instance.shard_assignments.node_to_runner
        if not state.topology.contains_node(node_id)
    ]:
        return f"node(s) {outside} aren't in the cluster"
    return instance_beyond_memory(instance, state)


def instance_beyond_memory(instance: Instance, state: State) -> str | None:
    """Why a node can't hold its shard of the instance besides the instances in the state (some
    of them perhaps still loading), or None if every node can. The master checks a launch
    with it against its own state: two launches sent at the same time were otherwise both
    placed in memory that holds one. The API checks a launch the same way before sending it,
    but against its own state, which may not show the other launch yet."""
    return shard_beyond_memory(
        instance.shard_assignments,
        usable_memory(
            memory_once_loaded(state.node_memory, state.instances, state.runners)
        ),
        state.node_identities,
    )


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


def _not_enough_memory(
    request: PlacementRequest,
    cycles: Sequence[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> str:
    reported = [cycle for cycle in cycles if all(node in node_memory for node in cycle)]
    if not reported:
        return "Waiting for the nodes to report their memory"
    free = {
        node_id: usage.ram_available.in_bytes for node_id, usage in node_memory.items()
    }
    most_free = Memory.from_bytes(
        max(sum(free[node] for node in cycle) for cycle in reported)
    )
    return (
        f"Not enough memory: {request.model_card.model_id} needs "
        f"{request.model_card.resident_size.in_gb:.1f} GB, but connected nodes can use at most "
        f"{most_free.in_gb:.1f} GB between them"
    )


def _node_name(node_id: NodeId, node_identities: Mapping[NodeId, NodeIdentity]) -> str:
    identity = node_identities.get(node_id)
    if identity is None or identity.friendly_name == NodeIdentity().friendly_name:
        return str(node_id)
    return identity.friendly_name


def _tensor_share_too_large(
    request: PlacementRequest,
    cycles: Sequence[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> str:
    # Explain the group of nodes that comes closest to holding its shares
    cycle = max(
        cycles,
        key=lambda c: min(node_memory[n].ram_available for n in c).in_bytes * len(c),
    )
    smallest = min(cycle, key=lambda n: node_memory[n].ram_available)
    share = request.model_card.resident_size // len(cycle)
    return (
        f"Tensor sharding gives each of {len(cycle)} nodes {share.in_gb:.1f} GB of "
        f"{request.model_card.model_id}, but {_node_name(smallest, node_identities)} can "
        f"use only about {node_memory[smallest].ram_available.in_gb:.1f} GB. Use Pipeline "
        "sharding, which splits the model by each node's memory, or nodes with more memory"
    )


def shard_beyond_memory(
    shard_assignments: ShardAssignments,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> str | None:
    """Why a node can't hold its shard of the model, or None if every node can."""
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        name = _node_name(node_id, node_identities)
        usage = node_memory.get(node_id)
        if usage is None:
            return f"Waiting for {name} to report its memory"
        shard = shard_assignments.runner_to_shard[runner_id]
        share = Memory.from_bytes(shard_bytes(shard))
        if share > usage.ram_available:
            return (
                f"Not enough memory: {name} can use only about "
                f"{usage.ram_available.in_gb:.1f} GB, but its shard of "
                f"{shard.model_card.model_id} needs {share.in_gb:.1f} GB"
            )
    return None


def usable_memory(
    node_memory: Mapping[NodeId, MemoryUsage],
) -> dict[NodeId, MemoryUsage]:
    """Each node's memory with `ram_available` cut to what a model can use there.

    A Mac's GPU can use only its working set (about 75% of the RAM), so an idle Mac has more
    memory free than a model can use. A runner whose shard needs more aborts while loading.
    """
    return {
        node_id: usage
        if usage.gpu_working_set is None or usage.gpu_working_set >= usage.ram_available
        else usage.model_copy(update={"ram_available": usage.gpu_working_set})
        for node_id, usage in node_memory.items()
    }


# Runners whose memory is already in their node's report (loaded), or about to be freed
_NOT_PENDING = (
    RunnerLoaded,
    RunnerWarmingUp,
    RunnerReady,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerShutdown,
)


def shard_bytes(shard: ShardMetadata) -> int:
    """The memory a runner needs for its shard of the model."""
    storage = shard.model_card.resident_size.in_bytes
    if isinstance(shard, TensorShardMetadata):
        return storage // shard.world_size
    if shard.n_layers == 0:
        return storage
    return storage * (shard.end_layer - shard.start_layer) // shard.n_layers


def memory_once_loaded(
    node_memory: Mapping[NodeId, MemoryUsage],
    instances: Mapping[InstanceId, Instance],
    runners: Mapping[RunnerId, RunnerStatus],
) -> dict[NodeId, MemoryUsage]:
    """Each node's free memory once the runners placed on it have loaded.

    A node reports the memory its loaded models use. A runner that is still starting or loading
    (or failed, and will load again) hasn't taken its memory yet, so without this the next
    placement would see that memory as free and could put a second model in it. Of a runner
    that is loading, only the layers it hasn't loaded yet are counted: the node's report has the
    others. Counting the whole shard counted those layers twice, and refused a second model that
    fitted while the first one was finishing its load.
    """
    pending: defaultdict[NodeId, int] = defaultdict(int)
    for instance in instances.values():
        assignments = instance.shard_assignments
        for node_id, runner_id in assignments.node_to_runner.items():
            status = runners.get(runner_id)
            if isinstance(status, _NOT_PENDING):
                continue
            shard = shard_bytes(assignments.runner_to_shard[runner_id])
            if isinstance(status, RunnerLoading) and status.total_layers > 0:
                unloaded = max(0, status.total_layers - status.layers_loaded)
                shard = shard * unloaded // status.total_layers
            pending[node_id] += shard
    return {
        node_id: usage.model_copy(
            update={
                "ram_available": Memory.from_bytes(
                    max(0, usage.ram_available.in_bytes - pending[node_id])
                )
            }
        )
        for node_id, usage in node_memory.items()
    }


def place_instance(
    request: PlacementRequest,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_backends: Mapping[NodeId, list[Backend]],
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] | None = None,
    runners: Mapping[RunnerId, RunnerStatus] | None = None,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    allowed_nodes: AbstractSet[NodeId] | None = None,
    cycles: Sequence[Cycle] | None = None,
) -> dict[InstanceId, Instance]:
    """Place an instance of the requested model on the best group of nodes.

    Pass the cluster's `runners` so that memory taken by instances that haven't loaded yet is
    counted as used; without them every instance is assumed loaded. `node_identities` name the
    nodes in errors. With `allowed_nodes`, only groups of those nodes are considered.
    `cycles` are the topology's, if the caller has them: finding them is the costly part, and
    grows quickly with the number of nodes.
    """
    if runners is not None:
        node_memory = memory_once_loaded(node_memory, current_instances, runners)
    node_memory = usable_memory(node_memory)
    if cycles is None:
        cycles = topology.get_cycles()
    if allowed_nodes is not None:
        cycles = [cycle for cycle in cycles if set(cycle.node_ids) <= allowed_nodes]
    if not cycles:
        raise ValueError("No nodes are available")
    candidate_cycles = list(filter(lambda it: len(it) >= request.min_nodes, cycles))
    if not candidate_cycles:
        largest = max(len(cycle) for cycle in cycles)
        raise ValueError(
            f"Needs {request.min_nodes} connected nodes, but at most {largest} are connected to each other"
        )

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
        if not candidate_cycles:
            raise ValueError("The chosen nodes aren't all connected to each other")
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, node_memory, request.model_card.resident_size
    )
    if len(cycles_with_sufficient_memory) == 0:
        raise ValueError(_not_enough_memory(request, candidate_cycles, node_memory))

    placement_rules = placement_rules_for_model_type(request.model_card.model_type)
    if request.sharding == Sharding.Tensor:
        if not request.model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {request.model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        splits_kv_heads = placement_rules.tensor_parallel_splits_kv_heads
        kv_heads = request.model_card.num_key_value_heads
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if request.model_card.hidden_size % len(cycle) == 0
            and (not splits_kv_heads or kv_heads is None or kv_heads % len(cycle) == 0)
            and all(
                number % len(cycle) == 0
                for number in placement_rules.tensor_parallel_divides
            )
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                f"No tensor sharding found for model with "
                f"hidden_size={request.model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles"
            )
        # Tensor sharding splits every layer evenly, so each node needs an equal share: enough
        # memory between the nodes is no use if one of them can't hold its share
        divisible_cycles = cycles_with_sufficient_memory
        cycles_with_sufficient_memory = [
            cycle
            for cycle in divisible_cycles
            if all(
                node_memory[node].ram_available
                >= request.model_card.resident_size // len(cycle)
                for node in cycle
            )
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                _tensor_share_too_large(
                    request, divisible_cycles, node_memory, node_identities or {}
                )
            )
    if request.sharding == Sharding.Pipeline:
        for model_id, error in placement_rules.pipeline_unsupported_model_ids:
            if request.model_card.model_id == ModelId(model_id):
                raise ValueError(error)
    if (
        request.sharding == Sharding.Pipeline
        and placement_rules.multi_node_pipeline_error is not None
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(placement_rules.multi_node_pipeline_error)

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    required_backends = set(INSTANCE_META_BACKENDS[request.instance_meta]) & set(
        request.model_card.backends
    )
    if not required_backends:
        raise ValueError(
            f"Model {request.model_card.model_id} backends "
            f"{sorted(b.value for b in request.model_card.backends)} cannot satisfy engine "
            f"{request.instance_meta.value} which requires "
            f"{sorted(b.value for b in INSTANCE_META_BACKENDS[request.instance_meta])}"
        )
    smallest_cycles = [
        cycle
        for cycle in smallest_cycles
        if all(
            set(node_backends.get(node_id, [])) & required_backends for node_id in cycle
        )
    ]
    if not smallest_cycles:
        raise ValueError(
            f"No cycle where every node supports a backend in "
            f"{sorted(b.value for b in required_backends)} for {request.model_card.model_id}"
        )

    rdma_ctl_status = node_rdma_ctl or {}

    def _all_rdma_ctl_enabled(cycle: Cycle) -> bool:
        return all(
            ((status := rdma_ctl_status.get(node_id)) is not None and status.enabled)
            for node_id in cycle
        )

    smallest_rdma_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.is_rdma_cycle(cycle) and _all_rdma_ctl_enabled(cycle)
    ]

    if request.instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles

    cycles_with_leaf_nodes: list[Cycle] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]

    resolved_download_status = download_status or {}
    candidate_cycles = (
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles
    )

    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
            _cycle_download_score(
                cycle, request.model_card.model_id, resolved_download_status
            ),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        ),
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        request = request.model_copy(
            update={
                "instance_meta": InstanceMeta.MlxRing,
                "sharding": Sharding.Pipeline,
            }
        )

    shard_assignments = get_shard_assignments(
        request.model_card, selected_cycle, request.sharding, node_memory
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    match request.instance_meta:
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
            target_instances[instance_id] = MlxJacclInstance(
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
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


def removal_events(
    instance_id: InstanceId,
    reason: InstanceRemovalReason,
    tasks: Mapping[TaskId, Task],
    at: datetime,
) -> Sequence[Event]:
    """Remove an instance, cancelling the requests it had not finished first."""
    events: list[Event] = [
        TaskStatusUpdated(task_status=TaskStatus.Cancelled, task_id=task.task_id)
        for task in tasks.values()
        if task.instance_id == instance_id
        and task.task_status in (TaskStatus.Pending, TaskStatus.Running)
    ]
    events.append(
        InstanceDeleted(instance_id=instance_id, reason=reason, at=at.isoformat())
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
