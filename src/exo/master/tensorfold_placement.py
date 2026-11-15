from collections.abc import Mapping, Sequence, Set

from exo.shared.tensorfold_support import (
    available_bytes,
    native_backend,
    runtime_reserve,
)
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.placement import PlacementRequest
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.tensorfold import NativeCapabilities
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId, TensorFoldInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import Sharding, whole_model_shard


def place_tensorfold(
    request: PlacementRequest,
    topology: Topology,
    instances: Mapping[InstanceId, Instance],
    memory: Mapping[NodeId, MemoryUsage],
    backends: Mapping[NodeId, list[Backend]],
    capabilities: Mapping[NodeId, NativeCapabilities],
    required_nodes: Set[NodeId] | None,
    allowed_nodes: Set[NodeId] | None,
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
) -> dict[InstanceId, Instance]:
    if request.min_nodes != 1 or request.sharding != Sharding.Pipeline:
        raise ValueError("TensorFold currently supports one complete model on one node")
    if required_nodes and len(required_nodes) != 1:
        raise ValueError("TensorFold cannot span multiple pinned nodes")
    card = request.model_card
    if card.model_type is None or card.tensorfold_format is None:
        raise ValueError("The model card must declare model_type and tensorfold_format")
    needed = card.resident_size.in_bytes + runtime_reserve(card)
    candidates: list[tuple[bool, int, NodeId]] = []
    qualified = False
    for node in sorted(topology.list_nodes()):
        if allowed_nodes is not None and node not in allowed_nodes:
            continue
        if required_nodes and node not in required_nodes:
            continue
        backend = native_backend(card, capabilities.get(node), backends.get(node, []))
        if backend is None:
            continue
        qualified = True
        usage = memory.get(node)
        available = available_bytes(usage, backend) if usage is not None else 0
        if available < needed:
            continue
        downloaded = any(
            isinstance(progress, DownloadCompleted)
            and progress.shard_metadata.model_card.model_id == card.model_id
            for progress in downloads.get(node, [])
        )
        candidates.append((downloaded, available, node))
    if not candidates:
        if qualified:
            raise ValueError(
                f"TensorFold needs at least {needed / 1024**3:.2f} GiB of available GPU memory "
                "for weights and runtime reserve; native loading validates cache capacity"
            )
        raise ValueError(
            "No TensorFold node supports this model family, weight format and stream protocol"
        )
    node = max(candidates)[2]
    runner_id = RunnerId()
    instance_id = InstanceId()
    instance = TensorFoldInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: whole_model_shard(card)},
            node_to_runner={node: runner_id},
        ),
        context_length=request.tensorfold_context_length,
        parallel=request.tensorfold_parallel,
    )
    return {**instances, instance_id: instance}


def tensorfold_launch_refusal(
    instance: TensorFoldInstance,
    memory: Mapping[NodeId, MemoryUsage],
    backends: Mapping[NodeId, list[Backend]],
    capabilities: Mapping[NodeId, NativeCapabilities],
) -> str | None:
    assignments = instance.shard_assignments
    node = next(iter(assignments.node_to_runner))
    card = next(iter(assignments.runner_to_shard.values())).model_card
    backend = native_backend(card, capabilities.get(node), backends.get(node, []))
    if backend is None:
        return "The selected node cannot serve this model with TensorFold"
    usage = memory.get(node)
    available = available_bytes(usage, backend) if usage is not None else 0
    if available < card.resident_size.in_bytes + runtime_reserve(card):
        return "The selected TensorFold node has insufficient available GPU memory"
    return None
