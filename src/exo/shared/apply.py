import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeTimedOut,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TestEvent,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.profiling import (
    NodeIdentity,
    NodeNetworkInfo,
    NodeThunderboltInfo,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import Connection, RDMAConnection
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import (
    MacmonMetrics,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    MemoryUsage,
    MiscData,
    NodeConfig,
    NodeNetworkInterfaces,
    StaticNodeInformation,
)


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent() | ChunkGenerated() | TaskAcknowledged() | InputChunkReceived()
        ):  # Pass-through events that don't modify state
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeGatheredInfo():
            return apply_node_gathered_info(event, state)
        case RunnerDeleted():
            return apply_runner_deleted(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)


def apply(state: State, event: IndexedEvent) -> State:
    # Just to test that events are only applied in correct order
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(
            f"Expected event {state.last_event_applied_idx + 1} but received {event.idx}"
        )
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})


def apply_node_download_progress(event: NodeDownloadProgress, state: State) -> State:
    """
    Update or add a node download progress to state.
    """
    dp = event.download_progress
    node_id = dp.node_id

    current = list(state.downloads.get(node_id, ()))

    replaced = False
    for i, existing_dp in enumerate(current):
        if existing_dp.shard_metadata == dp.shard_metadata:
            current[i] = dp
            replaced = True
            break

    if not replaced:
        current.append(dp)

    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        node_id: current,
    }
    return state.model_copy(update={"downloads": new_downloads})


def apply_task_created(event: TaskCreated, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {
        tid: task for tid, task in state.tasks.items() if tid != event.task_id
    }
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_status_updated(event: TaskStatusUpdated, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    update: dict[str, TaskStatus | None] = {
        "task_status": event.task_status,
    }
    if event.task_status != TaskStatus.Failed:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = state.tasks[event.task_id].model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_failed(event: TaskFailed, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    updated_task = state.tasks[event.task_id].model_copy(
        update={"error_type": event.error_type, "error_message": event.error_message}
    )
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_instance_created(event: InstanceCreated, state: State) -> State:
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {
        **state.instances,
        instance.instance_id: instance,
    }
    return state.model_copy(update={"instances": new_instances})


def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    new_instances: Mapping[InstanceId, Instance] = {
        iid: inst for iid, inst in state.instances.items() if iid != event.instance_id
    }
    return state.model_copy(update={"instances": new_instances})


def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    new_runners: Mapping[RunnerId, RunnerStatus] = {
        **state.runners,
        event.runner_id: event.runner_status,
    }
    return state.model_copy(update={"runners": new_runners})


def apply_runner_deleted(event: RunnerDeleted, state: State) -> State:
    assert event.runner_id in state.runners, (
        "RunnerDeleted before any RunnerStatusUpdated events"
    )
    new_runners: Mapping[RunnerId, RunnerStatus] = {
        rid: rs for rid, rs in state.runners.items() if rid != event.runner_id
    }
    return state.model_copy(update={"runners": new_runners})


def apply_node_timed_out(event: NodeTimedOut, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_node(event.node_id)
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    downloads = {
        key: value for key, value in state.downloads.items() if key != event.node_id
    }
    # Clean up all granular node mappings
    node_identities = {
        key: value
        for key, value in state.node_identities.items()
        if key != event.node_id
    }
    node_memory = {
        key: value for key, value in state.node_memory.items() if key != event.node_id
    }
    node_system = {
        key: value for key, value in state.node_system.items() if key != event.node_id
    }
    node_network = {
        key: value for key, value in state.node_network.items() if key != event.node_id
    }
    node_thunderbolt = {
        key: value
        for key, value in state.node_thunderbolt.items()
        if key != event.node_id
    }
    return state.model_copy(
        update={
            "downloads": downloads,
            "topology": topology,
            "last_seen": last_seen,
            "node_identities": node_identities,
            "node_memory": node_memory,
            "node_system": node_system,
            "node_network": node_network,
            "node_thunderbolt": node_thunderbolt,
        }
    )


def apply_node_gathered_info(event: NodeGatheredInfo, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_node(event.node_id)
    info = event.info

    # Build update dict with only the mappings that change
    update: dict[str, object] = {
        "last_seen": {
            **state.last_seen,
            event.node_id: datetime.fromisoformat(event.when),
        },
        "topology": topology,
    }

    match info:
        case MacmonMetrics():
            update["node_system"] = {
                **state.node_system,
                event.node_id: info.system_profile,
            }
            update["node_memory"] = {**state.node_memory, event.node_id: info.memory}
        case MemoryUsage():
            update["node_memory"] = {**state.node_memory, event.node_id: info}
        case NodeConfig():
            pass
        case MiscData():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={"friendly_name": info.friendly_name}
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case StaticNodeInformation():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={"model_id": info.model, "chip_id": info.chip}
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case NodeNetworkInterfaces():
            update["node_network"] = {
                **state.node_network,
                event.node_id: NodeNetworkInfo(interfaces=info.ifaces),
            }
        case MacThunderboltIdentifiers():
            update["node_thunderbolt"] = {
                **state.node_thunderbolt,
                event.node_id: NodeThunderboltInfo(interfaces=info.idents),
            }
        case MacThunderboltConnections():
            conn_map = {
                tb_ident.domain_uuid: (nid, tb_ident.rdma_interface)
                for nid in state.node_thunderbolt
                for tb_ident in state.node_thunderbolt[nid].interfaces
            }
            as_rdma_conns = [
                Connection(
                    source=event.node_id,
                    sink=conn_map[tb_conn.sink_uuid][0],
                    edge=RDMAConnection(
                        source_rdma_iface=conn_map[tb_conn.source_uuid][1],
                        sink_rdma_iface=conn_map[tb_conn.sink_uuid][1],
                    ),
                )
                for tb_conn in info.conns
                if tb_conn.source_uuid in conn_map
                if tb_conn.sink_uuid in conn_map
            ]
            topology.replace_all_out_rdma_connections(event.node_id, as_rdma_conns)

    return state.model_copy(update=update)


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_connection(event.conn)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_connection(event.conn)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})
