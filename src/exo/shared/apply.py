import copy
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
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
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus
# from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import (
MacmonMetrics    , MemoryUsage, NetworkInterfaceInfo, TBIdentifier, TBConnection, NodeConfig, MiscData, StaticNodeInformation
    )


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
    state.topology.remove_node(event.node_id)
    node_profiles = {
        key: value for key, value in state.node_profiles.items() if key != event.node_id
    }
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    return state.model_copy(
        update={
            "topology": topology,
            "node_profiles": node_profiles,
            "last_seen": last_seen,
        }
    )


def apply_node_gathered_info(event: NodeGatheredInfo, state: State) -> State:
    info = event.info
    profile = state.node_profiles.get(event.node_id, NodePerformanceProfile())
    match info:
        case MacmonMetrics():
            profile.system = info.system_profile
            profile.memory = info.memory
        case MemoryUsage():
            profile.memory = info
        case NodeConfig():
            pass
        case MiscData():
            profile.friendly_name = info.friendly_name
        case StaticNodeInformation():
            profile.model_id = info.model
            profile.chip_id = info.chip
        # TODO: makes me slightly sad
        case Sequence():
            if info != []:
                match info[0]:
                    case NetworkInterfaceInfo():
                        profile.network_interfaces = cast(Sequence[NetworkInterfaceInfo], info)
                    case TBIdentifier():
                        profile.tb_interfaces = cast(Sequence[TBIdentifier], info)
                    case TBConnection():
                        # TODO:
                        pass

    last_seen = {**state.last_seen, event.node_id: datetime.fromisoformat(event.when)}
    new_profiles = {**state.node_profiles, event.node_id: profile}
    return state.model_copy(update={"node_profiles": new_profiles, "last_seen": last_seen})



def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_connection(event.edge)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    if not topology.contains_connection(event.edge):
        return state
    topology.remove_connection(event.edge)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})
