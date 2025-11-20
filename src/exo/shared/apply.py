import copy
from collections.abc import Mapping, Sequence

from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeMemoryMeasured,
    NodePerformanceMeasured,
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
    TopologyNodeCreated,
)
from exo.shared.types.profiling import NodePerformanceProfile, SystemPerformanceProfile
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
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodePerformanceMeasured():
            return apply_node_performance_measured(event, state)
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
        case TopologyNodeCreated():
            return apply_topology_node_created(event, state)
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
    new_node_downloads: Sequence[DownloadProgress] = [
        event.download_progress
        if dp.shard_metadata == event.download_progress.shard_metadata
        else dp
        for dp in state.downloads.get(
            event.download_progress.node_id, [event.download_progress]
        )
    ]
    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        event.download_progress.node_id: new_node_downloads,
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


# TODO: This whole function needs fixing
def apply_node_performance_measured(
    event: NodePerformanceMeasured, state: State
) -> State:
    new_profiles: Mapping[NodeId, NodePerformanceProfile] = {
        **state.node_profiles,
        event.node_id: event.node_profile,
    }
    state = state.model_copy(update={"node_profiles": new_profiles})
    topology = copy.copy(state.topology)
    if not topology.contains_node(event.node_id):
        # TODO: figure out why this is happening in the first place
        topology.add_node(NodeInfo(node_id=event.node_id))
    topology.update_node_profile(event.node_id, event.node_profile)
    return state.model_copy(update={"topology": topology})


def apply_node_memory_measured(event: NodeMemoryMeasured, state: State) -> State:
    existing = state.node_profiles.get(event.node_id)
    topology = copy.copy(state.topology)

    if existing is None:
        created = NodePerformanceProfile(
            model_id="unknown",
            chip_id="unknown",
            friendly_name="Unknown",
            memory=event.memory,
            network_interfaces=[],
            system=SystemPerformanceProfile(
                flops_fp16=0.0,
                gpu_usage=0.0,
                temp=0.0,
                sys_power=0.0,
                pcpu_usage=0.0,
                ecpu_usage=0.0,
                ane_power=0.0,
            ),
        )
        created_profiles: Mapping[NodeId, NodePerformanceProfile] = {
            **state.node_profiles,
            event.node_id: created,
        }
        if not topology.contains_node(event.node_id):
            topology.add_node(NodeInfo(node_id=event.node_id))
        topology.update_node_profile(event.node_id, created)
        return state.model_copy(
            update={"node_profiles": created_profiles, "topology": topology}
        )

    updated = existing.model_copy(update={"memory": event.memory})
    updated_profiles: Mapping[NodeId, NodePerformanceProfile] = {
        **state.node_profiles,
        event.node_id: updated,
    }
    if not topology.contains_node(event.node_id):
        topology.add_node(NodeInfo(node_id=event.node_id))
    topology.update_node_profile(event.node_id, updated)
    return state.model_copy(
        update={"node_profiles": updated_profiles, "topology": topology}
    )


def apply_topology_node_created(event: TopologyNodeCreated, state: State) -> State:
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
