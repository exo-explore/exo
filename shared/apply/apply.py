import copy
from functools import singledispatch
from typing import Mapping, TypeVar

# from shared.topology import Topology
from shared.types.common import NodeId
from shared.types.events import (
    ChunkGenerated,
    Event,
    EventFromEventLog,
    InstanceActivated,
    InstanceCreated,
    InstanceDeactivated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
    NodePerformanceMeasured,
    RunnerStatusUpdated,
    TaskCreated,
    TaskDeleted,
    TaskStateUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TopologyEdgeReplacedAtomically,
    WorkerStatusUpdated,
)
from shared.types.profiling import NodePerformanceProfile
from shared.types.state import State
from shared.types.tasks import Task, TaskId
from shared.types.worker.common import NodeStatus, RunnerId
from shared.types.worker.instances import BaseInstance, InstanceId, TypeOfInstance
from shared.types.worker.runners import RunnerStatus

S = TypeVar("S", bound=State)

@singledispatch
def event_apply(state: State, event: Event) -> State:
    raise RuntimeError(f"no handler for {type(event).__name__}")

def apply(state: State, event: EventFromEventLog[Event]) -> State:
    new_state: State = event_apply(state, event.event)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx_in_log})

@event_apply.register 
def apply_task_created(state: State, event: TaskCreated) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register
def apply_task_deleted(state: State, event: TaskDeleted) -> State:
    new_tasks: Mapping[TaskId, Task] = {tid: task for tid, task in state.tasks.items() if tid != event.task_id}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register
def apply_task_state_updated(state: State, event: TaskStateUpdated) -> State:
    if event.task_id not in state.tasks:
        return state
    
    updated_task = state.tasks[event.task_id].model_copy(update={"task_status": event.task_status})
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})

@event_apply.register
def apply_instance_created(state: State, event: InstanceCreated) -> State:
    instance = BaseInstance(instance_params=event.instance_params, instance_type=event.instance_type)
    new_instances: Mapping[InstanceId, BaseInstance] = {**state.instances, event.instance_id: instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register
def apply_instance_activated(state: State, event: InstanceActivated) -> State:
    if event.instance_id not in state.instances:
        return state
    
    updated_instance = state.instances[event.instance_id].model_copy(update={"type": TypeOfInstance.ACTIVE})
    new_instances: Mapping[InstanceId, BaseInstance] = {**state.instances, event.instance_id: updated_instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register
def apply_instance_deactivated(state: State, event: InstanceDeactivated) -> State:
    if event.instance_id not in state.instances:
        return state
    
    updated_instance = state.instances[event.instance_id].model_copy(update={"type": TypeOfInstance.INACTIVE})
    new_instances: Mapping[InstanceId, BaseInstance] = {**state.instances, event.instance_id: updated_instance}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register
def apply_instance_deleted(state: State, event: InstanceDeleted) -> State:
    new_instances: Mapping[InstanceId, BaseInstance] = {iid: inst for iid, inst in state.instances.items() if iid != event.instance_id}
    return state.model_copy(update={"instances": new_instances})

@event_apply.register
def apply_instance_replaced_atomically(state: State, event: InstanceReplacedAtomically) -> State:
    new_instances = dict(state.instances)
    if event.instance_to_replace in new_instances:
        del new_instances[event.instance_to_replace]
    if event.new_instance_id in state.instances:
        new_instances[event.new_instance_id] = state.instances[event.new_instance_id]
    return state.model_copy(update={"instances": new_instances})

@event_apply.register
def apply_runner_status_updated(state: State, event: RunnerStatusUpdated) -> State:
    new_runners: Mapping[RunnerId, RunnerStatus] = {**state.runners, event.runner_id: event.runner_status}
    return state.model_copy(update={"runners": new_runners})

@event_apply.register
def apply_node_performance_measured(state: State, event: NodePerformanceMeasured) -> State:
    new_profiles: Mapping[NodeId, NodePerformanceProfile] = {**state.node_profiles, event.node_id: event.node_profile}
    return state.model_copy(update={"node_profiles": new_profiles})

@event_apply.register
def apply_worker_status_updated(state: State, event: WorkerStatusUpdated) -> State:
    new_node_status: Mapping[NodeId, NodeStatus] = {**state.node_status, event.node_id: event.node_state}
    return state.model_copy(update={"node_status": new_node_status})

@event_apply.register
def apply_chunk_generated(state: State, event: ChunkGenerated) -> State:
    return state

@event_apply.register
def apply_topology_edge_created(state: State, event: TopologyEdgeCreated) -> State:
    topology = copy.copy(state.topology)
    topology.add_connection(event.edge)
    return state.model_copy(update={"topology": topology})

@event_apply.register
def apply_topology_edge_replaced_atomically(state: State, event: TopologyEdgeReplacedAtomically) -> State:
    topology = copy.copy(state.topology)
    topology.update_connection_profile(event.edge)
    return state.model_copy(update={"topology": topology})

@event_apply.register
def apply_topology_edge_deleted(state: State, event: TopologyEdgeDeleted) -> State:
    topology = copy.copy(state.topology)
    topology.remove_connection(event.edge)
    return state.model_copy(update={"topology": topology})

@event_apply.register
def apply_mlx_inference_saga_prepare(state: State, event: MLXInferenceSagaPrepare) -> State:
    return state

@event_apply.register
def apply_mlx_inference_saga_start_prepare(state: State, event: MLXInferenceSagaStartPrepare) -> State:
    return state