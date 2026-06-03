from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, final

from loguru import logger
from pydantic import TypeAdapter

from exo.api.load_balancing import load_instance_links
from exo.shared.models.model_cards import ModelId
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskFailed,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    SystemPerformanceProfile,
    ThunderboltBridgeStatus,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
    SocketConnections,
)
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerShutdown,
    RunnerStatus,
)
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MacmonMetrics,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    MiscData,
    NodeBackends,
    NodeConfig,
    NodeDiskUsage,
    NodeNetworkInterfaces,
    RdmaCtlStatus,
    StaticNodeInformation,
    ThunderboltBridgeInfo,
)
from exo.utils.info_gatherer.info_gatherer import (
    MemoryUsage as GatheredMemoryUsage,
)
from exo.utils.pydantic_ext import FrozenModel

_DOWNLOAD_PROGRESS_ADAPTER = TypeAdapter[DownloadProgress](DownloadProgress)
_GATHERED_INFO_ADAPTER = TypeAdapter[GatheredInfo](GatheredInfo)
_RUNNER_STATUS_ADAPTER = TypeAdapter[RunnerStatus](RunnerStatus)
_SOCKET_CONNECTIONS_ADAPTER = TypeAdapter[SocketConnections](SocketConnections)
_TASK_ADAPTER = TypeAdapter[Task](Task)
_NODE_METRICS_PREFIX = "node_metrics"
_RUNNERS_TAG = "runners"
_TASKS_TAG = "tasks"
_SOCKET_CONNECTIONS_TAG = "socket_connections"
_TOPOLOGY_RELEVANT_TAGS = frozenset(
    (
        MacThunderboltIdentifiers.tag(),
        MacThunderboltConnections.tag(),
        NodeNetworkInterfaces.tag(),
        RdmaCtlStatus.tag(),
        ThunderboltBridgeInfo.tag(),
        _SOCKET_CONNECTIONS_TAG,
    )
)


DashboardEventName = Literal[
    "bootstrap",
    "node_update",
    "download_update",
    "topology_update",
    "instance_update",
    "runner_update",
    "task_update",
]


@final
class DashboardNodeIdentityPatch(FrozenModel):
    model_id: str | None = None
    chip_id: str | None = None
    friendly_name: str | None = None
    os_version: str | None = None
    os_build_version: str | None = None


@final
class DashboardTopology(FrozenModel):
    nodes: Sequence[NodeId] = []
    connections: Mapping[
        NodeId, Mapping[NodeId, Sequence[SocketConnection | RDMAConnection]]
    ] = {}


@final
class DashboardBootstrap(FrozenModel):
    feature_flags: Mapping[str, bool] = {}
    instances: Mapping[InstanceId, Instance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    tasks: Mapping[TaskId, Task] = {}
    instance_links: Mapping[InstanceLinkId, InstanceLink] = {}


@final
class DashboardNodeUpdate(FrozenModel):
    node_id: NodeId
    identity: DashboardNodeIdentityPatch | None = None
    memory: MemoryUsage | None = None
    disk: DiskUsage | None = None
    system: SystemPerformanceProfile | None = None
    network: NodeNetworkInfo | None = None
    thunderbolt: NodeThunderboltInfo | None = None
    thunderbolt_bridge: ThunderboltBridgeStatus | None = None
    rdma_ctl: NodeRdmaCtlStatus | None = None
    backends: Sequence[Backend] | None = None


@final
class DashboardDownloadUpdate(FrozenModel):
    node_id: NodeId
    model_id: ModelId
    download: DownloadProgress | None


@final
class DashboardTopologyUpdate(FrozenModel):
    topology: DashboardTopology
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []


@final
class DashboardInstanceUpdate(FrozenModel):
    instance_id: InstanceId
    instance: Instance | None


@final
class DashboardRunnerUpdate(FrozenModel):
    runner_id: RunnerId
    runner: RunnerStatus | None


@final
class DashboardTaskUpdate(FrozenModel):
    task_id: TaskId
    task: Task | None = None
    task_status: TaskStatus | None = None
    error_type: str | None = None
    error_message: str | None = None


DashboardPayload = (
    DashboardBootstrap
    | DashboardNodeUpdate
    | DashboardDownloadUpdate
    | DashboardTopologyUpdate
    | DashboardInstanceUpdate
    | DashboardRunnerUpdate
    | DashboardTaskUpdate
)


@dataclass(frozen=True)
class NodeMetricsTopologyInputs:
    node_ids: frozenset[NodeId]
    node_network: Mapping[NodeId, NodeNetworkInfo]
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo]
    node_thunderbolt_bridge: Mapping[NodeId, ThunderboltBridgeStatus]
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus]
    node_thunderbolt_connections: Mapping[NodeId, MacThunderboltConnections]
    node_socket_connections: Mapping[
        NodeId, Mapping[NodeId, Sequence[SocketConnection]]
    ]


def format_dashboard_sse(
    event_name: DashboardEventName, payload: DashboardPayload
) -> str:
    return (
        f"event: {event_name}\n"
        f"data: {payload.model_dump_json(by_alias=True, exclude_unset=True)}\n\n"
    )


def dashboard_bootstrap(
    *,
    feature_flags: Mapping[str, bool],
    instance_link_values: Mapping[str, str],
    instances: Mapping[InstanceId, Instance],
    runners: Mapping[RunnerId, RunnerStatus],
    tasks: Mapping[TaskId, Task],
) -> DashboardBootstrap:
    instance_links = load_instance_links(list(instance_link_values.values()))
    return DashboardBootstrap(
        feature_flags=feature_flags,
        instances=instances,
        runners=runners,
        tasks=tasks,
        instance_links={link.link_id: link for link in instance_links},
    )


def dashboard_event_from_lv_update(
    key: str, payload: str | None
) -> tuple[DashboardEventName, DashboardPayload] | None:
    parsed = _parse_node_metrics_key(key)
    if parsed is None:
        return None

    node_id, tag, subject_id = parsed
    if tag == "downloads":
        if subject_id is None:
            logger.warning(f"Ignoring malformed dashboard download metric key: {key}")
            return None
        download = (
            None
            if payload is None
            else _DOWNLOAD_PROGRESS_ADAPTER.validate_json(payload)
        )
        return (
            "download_update",
            DashboardDownloadUpdate(
                node_id=node_id,
                model_id=ModelId(subject_id),
                download=download,
            ),
        )

    if tag == _RUNNERS_TAG:
        if subject_id is None:
            logger.warning(f"Ignoring malformed dashboard runner metric key: {key}")
            return None
        return (
            "runner_update",
            DashboardRunnerUpdate(
                runner_id=RunnerId(subject_id),
                runner=None
                if payload is None
                else _RUNNER_STATUS_ADAPTER.validate_json(payload),
            ),
        )

    if tag == _TASKS_TAG:
        if subject_id is None:
            logger.warning(f"Ignoring malformed dashboard task metric key: {key}")
            return None
        task = None if payload is None else _TASK_ADAPTER.validate_json(payload)
        return (
            "task_update",
            DashboardTaskUpdate(
                task_id=TaskId(subject_id),
                task=task,
                task_status=None if task is None else task.task_status,
            ),
        )

    if tag == _SOCKET_CONNECTIONS_TAG:
        return None

    if payload is None:
        node_update = _node_delete_update_from_tag(node_id, tag)
    else:
        node_update = _node_update_from_gathered_info(
            node_id, _GATHERED_INFO_ADAPTER.validate_json(payload)
        )
    if node_update is None:
        return None
    return ("node_update", node_update)


def lv_update_affects_topology(key: str) -> bool:
    parsed = _parse_node_metrics_key(key)
    if parsed is None:
        return False
    _, tag, _ = parsed
    return tag in _TOPOLOGY_RELEVANT_TAGS


def dashboard_event_from_state_event(
    event: Event,
) -> tuple[DashboardEventName, DashboardPayload] | None:
    match event:
        case InstanceCreated():
            return (
                "instance_update",
                DashboardInstanceUpdate(
                    instance_id=event.instance.instance_id,
                    instance=event.instance,
                ),
            )
        case InstanceDeleted():
            return (
                "instance_update",
                DashboardInstanceUpdate(
                    instance_id=event.instance_id,
                    instance=None,
                ),
            )
        case RunnerStatusUpdated():
            return (
                "runner_update",
                DashboardRunnerUpdate(
                    runner_id=event.runner_id,
                    runner=None
                    if isinstance(event.runner_status, RunnerShutdown)
                    else event.runner_status,
                ),
            )
        case TaskCreated():
            return (
                "task_update",
                DashboardTaskUpdate(task_id=event.task_id, task=event.task),
            )
        case TaskStatusUpdated():
            return (
                "task_update",
                DashboardTaskUpdate(
                    task_id=event.task_id,
                    task_status=event.task_status,
                ),
            )
        case TaskFailed():
            return (
                "task_update",
                DashboardTaskUpdate(
                    task_id=event.task_id,
                    error_type=event.error_type,
                    error_message=event.error_message,
                ),
            )
        case TopologyEdgeCreated() | TopologyEdgeDeleted():
            return None
        case _:
            return None


def state_event_affects_topology(event: Event) -> bool:
    return isinstance(event, (TopologyEdgeCreated, TopologyEdgeDeleted))


def dashboard_topology_update(
    *,
    node_metric_values: Mapping[str, str],
) -> DashboardTopologyUpdate:
    metrics = _topology_inputs_from_last_values(node_metric_values)
    topology = _topology_from_inputs(metrics)
    return DashboardTopologyUpdate(
        topology=DashboardTopology(
            nodes=list(topology.list_nodes()),
            connections=topology.map_connections(),
        ),
        thunderbolt_bridge_cycles=topology.get_thunderbolt_bridge_cycles(
            metrics.node_thunderbolt_bridge, metrics.node_network
        ),
    )


def _parse_node_metrics_key(key: str) -> tuple[NodeId, str, str | None] | None:
    parts = key.split("/")
    if parts and parts[0] == _NODE_METRICS_PREFIX:
        parts = parts[1:]
    if len(parts) < 2:
        return None
    node_id = NodeId(parts[0])
    tag = parts[1]
    if tag == "downloads" and len(parts) > 2:
        return (node_id, tag, "/".join(parts[2:]))
    if tag == _RUNNERS_TAG and len(parts) == 4 and parts[3] == "status":
        return (node_id, tag, parts[2])
    if tag == _TASKS_TAG and len(parts) == 3:
        return (node_id, tag, parts[2])
    return (node_id, tag, None)


def _node_update_from_gathered_info(
    node_id: NodeId, info: GatheredInfo
) -> DashboardNodeUpdate | None:
    match info:
        case MacmonMetrics():
            return DashboardNodeUpdate(
                node_id=node_id,
                system=info.system_profile,
                memory=info.memory,
            )
        case GatheredMemoryUsage():
            return DashboardNodeUpdate(node_id=node_id, memory=info)
        case NodeDiskUsage():
            return DashboardNodeUpdate(node_id=node_id, disk=info.disk_usage)
        case NodeConfig():
            return None
        case MiscData():
            return DashboardNodeUpdate(
                node_id=node_id,
                identity=DashboardNodeIdentityPatch(friendly_name=info.friendly_name),
            )
        case StaticNodeInformation():
            return DashboardNodeUpdate(
                node_id=node_id,
                identity=DashboardNodeIdentityPatch(
                    model_id=info.model,
                    chip_id=info.chip,
                    os_version=info.os_version,
                    os_build_version=info.os_build_version,
                ),
            )
        case NodeNetworkInterfaces():
            return DashboardNodeUpdate(
                node_id=node_id,
                network=NodeNetworkInfo(interfaces=info.ifaces),
            )
        case MacThunderboltIdentifiers():
            return DashboardNodeUpdate(
                node_id=node_id,
                thunderbolt=NodeThunderboltInfo(interfaces=info.idents),
            )
        case MacThunderboltConnections():
            return None
        case ThunderboltBridgeInfo():
            return DashboardNodeUpdate(
                node_id=node_id,
                thunderbolt_bridge=info.status,
            )
        case RdmaCtlStatus():
            return DashboardNodeUpdate(
                node_id=node_id,
                rdma_ctl=NodeRdmaCtlStatus(enabled=info.enabled),
            )
        case NodeBackends():
            return DashboardNodeUpdate(node_id=node_id, backends=info.backends)


def _node_delete_update_from_tag(
    node_id: NodeId, tag: str
) -> DashboardNodeUpdate | None:
    if tag == MacmonMetrics.tag():
        return DashboardNodeUpdate(node_id=node_id, system=None, memory=None)
    if tag == GatheredMemoryUsage.tag():
        return DashboardNodeUpdate(node_id=node_id, memory=None)
    if tag == NodeDiskUsage.tag():
        return DashboardNodeUpdate(node_id=node_id, disk=None)
    if tag in (MiscData.tag(), StaticNodeInformation.tag()):
        return DashboardNodeUpdate(node_id=node_id, identity=None)
    if tag == NodeNetworkInterfaces.tag():
        return DashboardNodeUpdate(node_id=node_id, network=None)
    if tag == MacThunderboltIdentifiers.tag():
        return DashboardNodeUpdate(node_id=node_id, thunderbolt=None)
    if tag == ThunderboltBridgeInfo.tag():
        return DashboardNodeUpdate(node_id=node_id, thunderbolt_bridge=None)
    if tag == RdmaCtlStatus.tag():
        return DashboardNodeUpdate(node_id=node_id, rdma_ctl=None)
    if tag == NodeBackends.tag():
        return DashboardNodeUpdate(node_id=node_id, backends=None)
    return None


def _topology_inputs_from_last_values(
    values: Mapping[str, str],
) -> NodeMetricsTopologyInputs:
    node_ids: set[NodeId] = set()
    node_network: dict[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: dict[NodeId, NodeThunderboltInfo] = {}
    node_thunderbolt_bridge: dict[NodeId, ThunderboltBridgeStatus] = {}
    node_rdma_ctl: dict[NodeId, NodeRdmaCtlStatus] = {}
    node_thunderbolt_connections: dict[NodeId, MacThunderboltConnections] = {}
    node_socket_connections: dict[NodeId, Mapping[NodeId, Sequence[SocketConnection]]] = {}

    for key, value in values.items():
        parsed = _parse_node_metrics_key(key)
        if parsed is None:
            continue
        node_id, tag, _ = parsed
        if tag in ("downloads", _RUNNERS_TAG, _TASKS_TAG):
            continue
        node_ids.add(node_id)
        if tag == _SOCKET_CONNECTIONS_TAG:
            try:
                socket_connections = _SOCKET_CONNECTIONS_ADAPTER.validate_json(value)
            except Exception:
                continue
            node_socket_connections[node_id] = socket_connections.connections
            continue
        try:
            info = _GATHERED_INFO_ADAPTER.validate_json(value)
        except Exception:
            continue
        match info:
            case NodeNetworkInterfaces():
                node_network[node_id] = NodeNetworkInfo(interfaces=info.ifaces)
            case MacThunderboltIdentifiers():
                node_thunderbolt[node_id] = NodeThunderboltInfo(interfaces=info.idents)
            case MacThunderboltConnections():
                node_thunderbolt_connections[node_id] = info
            case ThunderboltBridgeInfo():
                node_thunderbolt_bridge[node_id] = info.status
            case RdmaCtlStatus():
                node_rdma_ctl[node_id] = NodeRdmaCtlStatus(enabled=info.enabled)
            case _:
                pass

    return NodeMetricsTopologyInputs(
        node_ids=frozenset(node_ids),
        node_network=node_network,
        node_thunderbolt=node_thunderbolt,
        node_thunderbolt_bridge=node_thunderbolt_bridge,
        node_rdma_ctl=node_rdma_ctl,
        node_thunderbolt_connections=node_thunderbolt_connections,
        node_socket_connections=node_socket_connections,
    )


def _topology_from_inputs(
    metrics: NodeMetricsTopologyInputs,
) -> Topology:
    topology = Topology()
    for node_id in metrics.node_ids:
        topology.add_node(node_id)

    for source, sinks in metrics.node_socket_connections.items():
        if source not in metrics.node_ids:
            continue
        for sink, connections in sinks.items():
            if sink not in metrics.node_ids:
                continue
            for connection in connections:
                topology.add_connection(
                    Connection(source=source, sink=sink, edge=connection)
                )

    thunderbolt_by_uuid = {
        identity.domain_uuid: (node_id, identity.rdma_interface)
        for node_id, info in metrics.node_thunderbolt.items()
        for identity in info.interfaces
    }
    for source, connections in metrics.node_thunderbolt_connections.items():
        if not metrics.node_rdma_ctl.get(
            source, NodeRdmaCtlStatus(enabled=False)
        ).enabled:
            continue
        for connection in connections.conns:
            source_iface = thunderbolt_by_uuid.get(connection.source_uuid)
            sink_iface = thunderbolt_by_uuid.get(connection.sink_uuid)
            if source_iface is None or sink_iface is None:
                continue
            if source_iface[0] != source:
                logger.warning(
                    "Skipping Thunderbolt connection with invalid source uuid"
                )
                continue
            if not metrics.node_rdma_ctl.get(
                sink_iface[0], NodeRdmaCtlStatus(enabled=False)
            ).enabled:
                continue
            if (
                source_iface[0] not in metrics.node_ids
                or sink_iface[0] not in metrics.node_ids
            ):
                continue
            topology.add_connection(
                Connection(
                    source=source_iface[0],
                    sink=sink_iface[0],
                    edge=RDMAConnection(
                        source_rdma_iface=source_iface[1],
                        sink_rdma_iface=sink_iface[1],
                    ),
                )
            )

    return topology
