from collections.abc import Mapping, Sequence

from loguru import logger

from exo.master.placement import get_transition_events, place_instance
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event
from exo.shared.types.meta_instance import MetaInstance, MetaInstanceId
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import RDMAConnection, SocketConnection
from exo.shared.types.worker.instances import (
    BaseInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)


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


def instance_satisfies_meta_instance(
    meta_instance: MetaInstance,
    instance: Instance,
) -> bool:
    """Check if a single instance satisfies a meta-instance's constraints.

    This is a pure constraint check (model, min_nodes, node_ids).
    Use ``instance_connections_healthy`` separately for topology health.
    """
    if instance.shard_assignments.model_id != meta_instance.model_card.model_id:
        return False

    instance_nodes = set(instance.shard_assignments.node_to_runner.keys())

    if len(instance_nodes) < meta_instance.min_nodes:
        return False

    return meta_instance.node_ids is None or meta_instance.node_ids.issubset(
        instance_nodes
    )


def find_satisfying_instance(
    meta_instance: MetaInstance,
    instances: Mapping[InstanceId, Instance],
    topology: Topology,
) -> InstanceId | None:
    """Find an existing instance that is healthy and satisfies a meta-instance's constraints."""
    for instance_id, instance in instances.items():
        if instance_connections_healthy(
            instance, topology
        ) and instance_satisfies_meta_instance(meta_instance, instance):
            return instance_id
    return None


def find_unsatisfied_meta_instances(
    meta_instances: Mapping[MetaInstanceId, MetaInstance],
    instances: Mapping[InstanceId, Instance],
    topology: Topology,
) -> Sequence[MetaInstance]:
    """Return meta-instances that have no healthy, satisfying backing instance."""
    return [
        meta_instance
        for meta_instance in meta_instances.values()
        if find_satisfying_instance(meta_instance, instances, topology) is None
    ]


def try_place_for_meta_instance(
    meta_instance: MetaInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> Sequence[Event]:
    """Try to place an instance satisfying the meta-instance constraints.

    Returns InstanceCreated events on success, empty sequence on failure.
    """
    command = PlaceInstance(
        model_card=meta_instance.model_card,
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
        return list(get_transition_events(current_instances, target_instances))
    except ValueError as e:
        logger.debug(
            f"MetaInstance placement not possible for {meta_instance.model_card.model_id}: {e}"
        )
        return []
