import copy
from functools import singledispatch
from typing import Mapping, TypeVar

# from shared.topology import Topology
from shared.types.common import NodeId
from shared.types.events import (
    ChunkGenerated,
    Event,
    EventFromEventLog,
    Heartbeat,
    InstanceActivated,
    InstanceCreated,
    InstanceDeactivated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    NodePerformanceMeasured,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStateUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TopologyEdgeReplacedAtomically,
    TopologyNodeCreated,
    WorkerStatusUpdated,
)
from shared.types.profiling import NodePerformanceProfile
from shared.types.state import State
from shared.types.tasks import Task, TaskId, TaskStatus
from shared.types.topology import Connection, Node
from shared.types.worker.common import NodeStatus, RunnerId
from shared.types.worker.instances import Instance, InstanceId, InstanceStatus
from shared.types.worker.runners import RunnerStatus

S = TypeVar("S", bound=State)

@singledispatch
def event_apply(event: Event, state: State) -> State:
    raise RuntimeError(f"no handler registered for event type {type(event).__name__}")

def apply(state: State, event: EventFromEventLog[Event]) -> State:
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx_in_log})

@event_apply.register(Heartbeat)
def apply_heartbeat(event: Heartbeat, state: State) -> State:
    return state

@event_apply.register(TaskCreated)
def apply_task_created(event: TaskCreated, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register(TaskDeleted)
def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {tid: task for tid, task in state.tasks.items() if tid != event.task_id}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register(TaskStateUpdated)
def apply_task_state_updated(event: TaskStateUpdated, state: State) -> State:
    if event.task_id not in state.tasks:
        return state
    
    update: dict[str, TaskStatus | None] = {
        "task_status": event.task_status,
    }
    if event.task_status != TaskStatus.FAILED:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = state.tasks[event.task_id].model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register(TaskFailed)
def apply_task_failed(event: TaskFailed, state: State) -> State:
    if event.task_id not in state.tasks:
        return state
    
    updated_task = state.tasks[event.task_id].model_copy(update={"error_type": event.error_type, "error_message": event.error_message})
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register(InstanceCreated)
def apply_instance_created(event: InstanceCreated, state: State) -> State:
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {**state.instances, instance.instance_id: instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register(InstanceActivated)
def apply_instance_activated(event: InstanceActivated, state: State) -> State:
    if event.instance_id not in state.instances:
        return state
    
    updated_instance = state.instances[event.instance_id].model_copy(update={"type": InstanceStatus.ACTIVE})
    new_instances: Mapping[InstanceId, Instance] = {**state.instances, event.instance_id: updated_instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register(InstanceDeactivated)
def apply_instance_deactivated(event: InstanceDeactivated, state: State) -> State:
    if event.instance_id not in state.instances:
        return state
    
    updated_instance = state.instances[event.instance_id].model_copy(update={"type": InstanceStatus.INACTIVE})
    new_instances: Mapping[InstanceId, Instance] = {**state.instances, event.instance_id: updated_instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register(InstanceDeleted)
def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    new_instances: Mapping[InstanceId, Instance] = {iid: inst for iid, inst in state.instances.items() if iid != event.instance_id}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register(InstanceReplacedAtomically)
def apply_instance_replaced_atomically(event: InstanceReplacedAtomically, state: State) -> State:
    new_instances = dict(state.instances)
    if event.instance_to_replace in new_instances:
        del new_instances[event.instance_to_replace]
    if event.new_instance_id in state.instances:
        new_instances[event.new_instance_id] = state.instances[event.new_instance_id]
    return state.model_copy(update={"instances": new_instances})

@event_apply.register(RunnerStatusUpdated)
def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    new_runners: Mapping[RunnerId, RunnerStatus] = {**state.runners, event.runner_id: event.runner_status}
    return state.model_copy(update={"runners": new_runners})

@event_apply.register(RunnerDeleted)
def apply_runner_deleted(event: RunnerDeleted, state: State) -> State:
    new_runners: Mapping[RunnerId, RunnerStatus] = {rid: rs for rid, rs in state.runners.items() if rid != event.runner_id}
    return state.model_copy(update={"runners": new_runners})

@event_apply.register(NodePerformanceMeasured)
def apply_node_performance_measured(event: NodePerformanceMeasured, state: State) -> State:
    new_profiles: Mapping[NodeId, NodePerformanceProfile] = {**state.node_profiles, event.node_id: event.node_profile}
    state = state.model_copy(update={"node_profiles": new_profiles})
    if not state.topology.contains_node(event.node_id):
        # TODO: figure out why this is happening in the first place
        return state
    topology = copy.copy(state.topology)
    topology.update_node_profile(event.node_id, event.node_profile)
    return state.model_copy(update={"topology": topology})

@event_apply.register(WorkerStatusUpdated)
def apply_worker_status_updated(event: WorkerStatusUpdated, state: State) -> State:
    new_node_status: Mapping[NodeId, NodeStatus] = {**state.node_status, event.node_id: event.node_state}
    return state.model_copy(update={"node_status": new_node_status})

@event_apply.register(ChunkGenerated)
def apply_chunk_generated(event: ChunkGenerated, state: State) -> State:
    return state

@event_apply.register(TopologyNodeCreated)
def apply_topology_node_created(event: TopologyNodeCreated, state: State) -> State:
    topology = copy.copy(state.topology)
    topology.add_node(Node(node_id=event.node_id))
    if event.role == "MASTER":
        topology.set_master_node_id(event.node_id)
    return state.model_copy(update={"topology": topology})

@event_apply.register(TopologyEdgeCreated)
def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.copy(state.topology)
    topology.add_connection(event.edge)
    return state.model_copy(update={"topology": topology})

@event_apply.register(TopologyEdgeReplacedAtomically)
def apply_topology_edge_replaced_atomically(event: TopologyEdgeReplacedAtomically, state: State) -> State:
    topology = copy.copy(state.topology)
    topology.update_connection_profile(event.edge)
    return state.model_copy(update={"topology": topology})

@event_apply.register(TopologyEdgeDeleted)
def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.copy(state.topology)
    if not topology.contains_connection(event.edge):
        return state
    topology.remove_connection(event.edge)
    opposite_edge = Connection(
        local_node_id=event.edge.send_back_node_id,
        send_back_node_id=event.edge.local_node_id,
        local_multiaddr=event.edge.send_back_multiaddr,
        send_back_multiaddr=event.edge.local_multiaddr
    )
    if not topology.contains_connection(opposite_edge):
        return state.model_copy(update={"topology": topology})
    topology.remove_connection(opposite_edge)
    return state.model_copy(update={"topology": topology})