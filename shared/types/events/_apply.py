from functools import singledispatch
from typing import Mapping, TypeVar

# from shared.topology import Topology
from shared.types.common import NodeId
from shared.types.events._events import Event
from shared.types.events.components import EventFromEventLog
from shared.types.profiling import NodePerformanceProfile
from shared.types.state import State
from shared.types.tasks import Task, TaskId
from shared.types.worker.common import NodeStatus, RunnerId
from shared.types.worker.instances import BaseInstance, InstanceId, TypeOfInstance
from shared.types.worker.runners import RunnerStatus

from ._events import (
    ChunkGenerated,
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
    WorkerConnected,
    WorkerDisconnected,
    WorkerStatusUpdated,
)

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

# TODO implemente these
@event_apply.register
def apply_worker_connected(state: State, event: WorkerConnected) -> State:
    # source_node_id = event.edge.source_node_id
    # sink_node_id = event.edge.sink_node_id
    
    # new_node_status = dict(state.node_status)
    # if source_node_id not in new_node_status:
    #     new_node_status[source_node_id] = NodeStatus.Idle
    # if sink_node_id not in new_node_status:
    #     new_node_status[sink_node_id] = NodeStatus.Idle
    
    # new_topology = Topology() 
    # new_topology.add_connection(event.edge)
    
    # return state.model_copy(update={"node_status": new_node_status, "topology": new_topology})
    return state

@event_apply.register
def apply_worker_disconnected(state: State, event: WorkerDisconnected) -> State:
    # new_node_status: Mapping[NodeId, NodeStatus] = {nid: status for nid, status in state.node_status.items() if nid != event.vertex_id}
    
    # new_topology = Topology()
    
    # new_history = list(state.history) + [state.topology]
    
    # return state.model_copy(update={
    #     "node_status": new_node_status,
    #     "topology": new_topology,
    #     "history": new_history
    # })
    return state


@event_apply.register
def apply_topology_edge_created(state: State, event: TopologyEdgeCreated) -> State:
    # new_topology = Topology()
    # new_topology.add_node(event.vertex, event.vertex.node_id)
    # return state.model_copy(update={"topology": new_topology})
    return state

@event_apply.register
def apply_topology_edge_replaced_atomically(state: State, event: TopologyEdgeReplacedAtomically) -> State:
    # new_topology = Topology()
    # new_topology.add_connection(event.edge)
    # updated_connection = event.edge.model_copy(update={"connection_profile": event.edge_profile})
    # new_topology.update_connection_profile(updated_connection)
    # return state.model_copy(update={"topology": new_topology})
    return state

@event_apply.register
def apply_topology_edge_deleted(state: State, event: TopologyEdgeDeleted) -> State:
    # new_topology = Topology()
    # return state.model_copy(update={"topology": new_topology})
    return state

@event_apply.register
def apply_mlx_inference_saga_prepare(state: State, event: MLXInferenceSagaPrepare) -> State:
    return state

@event_apply.register
def apply_mlx_inference_saga_start_prepare(state: State, event: MLXInferenceSagaStartPrepare) -> State:
    return state