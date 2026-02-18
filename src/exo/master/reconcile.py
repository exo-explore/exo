from collections.abc import Mapping, Sequence
from typing import NamedTuple

from loguru import logger

from exo.master.placement import get_transition_events, place_instance
from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import MetaInstanceId, NodeId
from exo.shared.types.events import Event
from exo.shared.types.meta_instance import MetaInstance
from exo.shared.types.profiling import MemoryUsage, NodeIdentity, NodeNetworkInfo
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.topology import RDMAConnection, SocketConnection
from exo.shared.types.worker.instances import (
    BaseInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerShutdown,
    RunnerStatus,
)


class PlacementResult(NamedTuple):
    """Result of a placement attempt: events to apply and optional error reason."""

    events: Sequence[Event]
    error: str | None


def _get_ring_order(instance: BaseInstance) -> list[NodeId]:
    """Reconstruct ring order from shard device_rank."""
    node_ranks: list[tuple[NodeId, int]] = []
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        node_ranks.append((node_id, shard.device_rank))
    node_ranks.sort(key=lambda x: x[1])
    return [node_id for node_id, _ in node_ranks]


def _ring_connections_healthy(instance: MlxRingInstance, topology: Topology) -> bool:
    """Check that the specific IPs used by a ring instance still exist in the topology."""
    ring = _get_ring_order(instance)
    n = len(ring)
    for node in ring:
        hosts = instance.hosts_by_node[node]
        for idx in range(n):
            host = hosts[idx]
            if host.ip in ("0.0.0.0", "198.51.100.1"):
                continue  # self or placeholder
            # Real connection: node → ring[idx]. Check specific IP.
            connections = topology.get_all_connections_between(node, ring[idx])
            if not any(
                isinstance(c, SocketConnection)
                and c.sink_multiaddr.ip_address == host.ip
                for c in connections
            ):
                return False
    return True


def _jaccl_connections_healthy(instance: MlxJacclInstance, topology: Topology) -> bool:
    """Check that the specific RDMA interfaces used by a JACCL instance still exist."""
    ring = _get_ring_order(instance)
    n = len(ring)
    for i in range(n):
        for j in range(n):
            iface = instance.jaccl_devices[i][j]
            if iface is None:
                continue
            connections = topology.get_all_connections_between(ring[i], ring[j])
            if not any(
                isinstance(c, RDMAConnection) and c.source_rdma_iface == iface
                for c in connections
            ):
                return False
    return True


def instance_connections_healthy(instance: Instance, topology: Topology) -> bool:
    """Check that an instance's nodes and specific connections are still in the topology."""
    instance_nodes = set(instance.shard_assignments.node_to_runner.keys())
    if not all(topology.contains_node(n) for n in instance_nodes):
        return False
    if len(instance_nodes) <= 1:
        return True
    match instance:
        case MlxRingInstance():
            return _ring_connections_healthy(instance, topology)
        case MlxJacclInstance():
            return _jaccl_connections_healthy(instance, topology)


def instance_runners_failed(
    instance: Instance,
    runners: Mapping[RunnerId, RunnerStatus],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> tuple[bool, str | None]:
    """Check if an instance's runners have all reached terminal failure states.

    Returns ``(True, error_message)`` when ALL runners are terminal
    (``RunnerFailed`` or ``RunnerShutdown``) and at least one is ``RunnerFailed``.

    Returns ``(False, None)`` when runners are still active, haven't reported
    yet, or all gracefully shut down (no ``RunnerFailed``).
    """
    instance_runner_ids = set(instance.shard_assignments.node_to_runner.values())

    if not instance_runner_ids:
        return False, None

    # Build reverse mapping: runner_id -> node_id
    runner_to_node: dict[RunnerId, NodeId] = {
        runner_id: node_id
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    }

    has_any_failed = False
    error_messages: list[str] = []

    for runner_id in instance_runner_ids:
        status = runners.get(runner_id)
        if status is None:
            # Runner hasn't reported yet — instance is still starting
            return False, None
        if isinstance(status, RunnerFailed):
            has_any_failed = True
            if status.error_message:
                node_id = runner_to_node.get(runner_id)
                name = (
                    node_identities[node_id].friendly_name
                    if node_id and node_id in node_identities
                    else node_id or "unknown"
                )
                error_messages.append(f"{name}: {status.error_message}")
        elif isinstance(status, RunnerShutdown):
            pass  # Terminal but not a failure indicator on its own
        else:
            # Runner is still active (connecting, loading, running, etc.)
            return False, None

    if has_any_failed:
        return True, "; ".join(error_messages) if error_messages else "Runner failed"

    # All runners are Shutdown but none Failed — graceful shutdown, not a failure
    return False, None


def instance_satisfies_meta_instance(
    meta_instance: MetaInstance,
    instance: Instance,
) -> bool:
    """Check if a single instance satisfies a meta-instance's constraints.

    This is a pure constraint check (model, min_nodes, node_ids).
    Use ``instance_connections_healthy`` separately for topology health.
    """
    if instance.shard_assignments.model_id != meta_instance.model_id:
        return False

    instance_nodes = set(instance.shard_assignments.node_to_runner.keys())

    if len(instance_nodes) < meta_instance.min_nodes:
        return False

    return meta_instance.node_ids is None or set(meta_instance.node_ids).issubset(
        instance_nodes
    )


def find_unsatisfied_meta_instances(
    meta_instances: Mapping[MetaInstanceId, MetaInstance],
    instances: Mapping[InstanceId, Instance],
    topology: Topology,
) -> Sequence[MetaInstance]:
    """Return meta-instances that have no healthy backing instance."""
    unsatisfied: list[MetaInstance] = []
    for meta_id, meta_instance in meta_instances.items():
        has_healthy_backing = any(
            instance.meta_instance_id == meta_id
            and instance_connections_healthy(instance, topology)
            for instance in instances.values()
        )
        if not has_healthy_backing:
            unsatisfied.append(meta_instance)
    return unsatisfied


def try_place_for_meta_instance(
    meta_instance: MetaInstance,
    model_card: ModelCard,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    tasks: Mapping[TaskId, Task] | None = None,
) -> PlacementResult:
    """Try to place an instance satisfying the meta-instance constraints.

    Returns a :class:`PlacementResult` with events on success, or an error
    reason on failure.
    """
    command = PlaceInstance(
        model_card=model_card,
        sharding=meta_instance.sharding,
        instance_meta=meta_instance.instance_meta,
        min_nodes=meta_instance.min_nodes,
    )
    try:
        target_instances = place_instance(
            command,
            topology,
            current_instances,
            node_memory,
            node_network,
            required_nodes=(
                set(meta_instance.node_ids) if meta_instance.node_ids else None
            ),
        )
        # Tag the new instance with meta_instance_id
        new_instance_ids = set(target_instances.keys()) - set(current_instances.keys())
        if new_instance_ids:
            new_id = next(iter(new_instance_ids))
            target_instances[new_id] = target_instances[new_id].model_copy(
                update={"meta_instance_id": meta_instance.meta_instance_id}
            )
        return PlacementResult(
            events=list(
                get_transition_events(current_instances, target_instances, tasks or {})
            ),
            error=None,
        )
    except ValueError as e:
        logger.debug(
            f"MetaInstance placement not possible for {meta_instance.model_id}: {e}"
        )
        return PlacementResult(events=[], error=str(e))
