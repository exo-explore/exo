from collections.abc import Mapping, Sequence
from queue import Queue

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import Event, EventTypes
from shared.types.graphs.resource_graph import ResourceGraph
from shared.types.networking.topology import NetworkState
from shared.types.profiling.common import NodeProfile
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeState
from shared.types.worker.instances import InstanceData, InstanceId


class ExternalCommand(BaseModel): ...


class MasterState(SharedState):
    network_state: NetworkState
    node_profiles: Mapping[NodeId, NodeProfile]
    node_states: Mapping[NodeId, NodeState]
    job_inbox: Queue[ExternalCommand]
    job_outbox: Queue[ExternalCommand]


def get_inference_plan(
    inbox: Queue[ExternalCommand],
    outbox: Queue[ExternalCommand],
    resource_graph: ResourceGraph,
    current_instances: Mapping[InstanceId, InstanceData],
) -> Mapping[InstanceId, InstanceData]: ...


TransitionEventTypes = EventTypes


def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceData],
    target_instances: Mapping[InstanceId, InstanceData],
) -> Sequence[Event[TransitionEventTypes]]: ...
