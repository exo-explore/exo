import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeCreated,
    NodeDownloadProgress,
    NodeIdentityMeasured,
    NodeMemoryMeasured,
    NodeNetworkMeasured,
    NodeSystemMeasured,
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
    MemoryPerformanceProfile,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent() | ChunkGenerated() | TaskAcknowledged()
        ):  # TaskAcknowledged should never be sent by a worker but i dont mind if it just gets ignored
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeCreated():
            return apply_topology_node_created(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeIdentityMeasured():
            return apply_node_identity_measured(event, state)
        case NodeSystemMeasured():
            return apply_node_system_measured(event, state)
        case NodeNetworkMeasured():
            return apply_node_network_measured(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeMemoryMeasured():
            return apply_node_memory_measured(event, state)
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
    topology = copy.copy(state.topology)
    state.topology.remove_node(event.node_id)
    node_identities = {
        key: value
        for key, value in state.node_identities.items()
        if key != event.node_id
    }
    node_memories = {
        key: value for key, value in state.node_memories.items() if key != event.node_id
    }
    node_systems = {
        key: value for key, value in state.node_systems.items() if key != event.node_id
    }
    node_networks = {
        key: value for key, value in state.node_networks.items() if key != event.node_id
    }
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    return state.model_copy(
        update={
            "topology": topology,
            "node_identities": node_identities,
            "node_memories": node_memories,
            "node_systems": node_systems,
            "node_networks": node_networks,
            "last_seen": last_seen,
        }
    )


def _reconstruct_profile(
    node_id: NodeId,
    state: State,
    *,
    identity: NodeIdentity | None = None,
    memory: MemoryPerformanceProfile | None = None,
    system: SystemPerformanceProfile | None = None,
    network_interfaces: list[NetworkInterfaceInfo] | None = None,
) -> NodePerformanceProfile:
    """Reconstruct a NodePerformanceProfile from split state storage.

    Uses provided overrides, falling back to state values.
    """
    ident = identity or state.node_identities.get(node_id)
    mem = memory or state.node_memories.get(node_id)
    sys = system or state.node_systems.get(node_id)
    nets = (
        network_interfaces
        if network_interfaces is not None
        else state.node_networks.get(node_id, [])
    )

    return NodePerformanceProfile(
        model_id=ident.model_id if ident else None,
        chip_id=ident.chip_id if ident else None,
        friendly_name=ident.friendly_name if ident else None,
        memory=mem,
        network_interfaces=nets,
        system=sys,
    )


def apply_node_identity_measured(event: NodeIdentityMeasured, state: State) -> State:
    topology = copy.copy(state.topology)

    identity = NodeIdentity(
        model_id=event.model_id,
        chip_id=event.chip_id,
        friendly_name=event.friendly_name,
    )
    new_identities: Mapping[NodeId, NodeIdentity] = {
        **state.node_identities,
        event.node_id: identity,
    }
    last_seen: Mapping[NodeId, datetime] = {
        **state.last_seen,
        event.node_id: datetime.fromisoformat(event.when),
    }
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    reconstructed = _reconstruct_profile(event.node_id, state, identity=identity)
    topology.update_node_profile(event.node_id, reconstructed)
    return state.model_copy(
        update={
            "node_identities": new_identities,
            "topology": topology,
            "last_seen": last_seen,
        }
    )


def apply_node_system_measured(event: NodeSystemMeasured, state: State) -> State:
    topology = copy.copy(state.topology)

    new_systems: Mapping[NodeId, SystemPerformanceProfile] = {
        **state.node_systems,
        event.node_id: event.system,
    }
    last_seen: Mapping[NodeId, datetime] = {
        **state.last_seen,
        event.node_id: datetime.fromisoformat(event.when),
    }
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    reconstructed = _reconstruct_profile(event.node_id, state, system=event.system)
    topology.update_node_profile(event.node_id, reconstructed)
    return state.model_copy(
        update={
            "node_systems": new_systems,
            "topology": topology,
            "last_seen": last_seen,
        }
    )


def apply_node_network_measured(event: NodeNetworkMeasured, state: State) -> State:
    topology = copy.copy(state.topology)

    new_networks: Mapping[NodeId, list[NetworkInterfaceInfo]] = {
        **state.node_networks,
        event.node_id: event.network_interfaces,
    }
    last_seen: Mapping[NodeId, datetime] = {
        **state.last_seen,
        event.node_id: datetime.fromisoformat(event.when),
    }
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    reconstructed = _reconstruct_profile(
        event.node_id, state, network_interfaces=event.network_interfaces
    )
    topology.update_node_profile(event.node_id, reconstructed)
    return state.model_copy(
        update={
            "node_networks": new_networks,
            "topology": topology,
            "last_seen": last_seen,
        }
    )


def apply_node_memory_measured(event: NodeMemoryMeasured, state: State) -> State:
    topology = copy.copy(state.topology)

    new_memories: Mapping[NodeId, MemoryPerformanceProfile] = {
        **state.node_memories,
        event.node_id: event.memory,
    }
    last_seen: Mapping[NodeId, datetime] = {
        **state.last_seen,
        event.node_id: datetime.fromisoformat(event.when),
    }
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    reconstructed = _reconstruct_profile(event.node_id, state, memory=event.memory)
    topology.update_node_profile(event.node_id, reconstructed)
    return state.model_copy(
        update={
            "node_memories": new_memories,
            "topology": topology,
            "last_seen": last_seen,
        }
    )


def apply_topology_node_created(event: NodeCreated, state: State) -> State:
    topology = copy.copy(state.topology)
    topology.add_node(NodeInfo(node_id=event.node_id))
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.copy(state.topology)
    topology.add_connection(event.edge)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.copy(state.topology)
    if not topology.contains_connection(event.edge):
        return state
    topology.remove_connection(event.edge)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})
