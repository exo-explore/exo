from collections.abc import Mapping, Sequence
from enum import Enum
from queue import Queue
from typing import Generic, TypeVar

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import (
    EdgeEventTypes,
    Event,
    EventTypes,
    NodeProfileEventTypes,
    NodeStatusEventTypes,
    State,
)
from shared.types.graphs.resource_graph import ResourceGraph
from shared.types.networking.edges import (
    AddressingProtocol,
    ApplicationProtocol,
    EdgeId,
    NetworkEdge,
)
from shared.types.networking.topology import OrphanedPartOfTopology, Topology
from shared.types.profiling.common import NodeProfile
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeStatus
from shared.types.worker.instances import InstanceData, InstanceId


class ExternalCommand(BaseModel): ...


class CachePolicyType(str, Enum):
    KeepAll = "KeepAll"


CachePolicyTypeT = TypeVar("CachePolicyTypeT", bound=CachePolicyType)


class CachePolicy(BaseModel, Generic[CachePolicyTypeT]):
    policy_type: CachePolicyTypeT


class NodeProfileState(State[NodeProfileEventTypes]):
    node_profiles: Mapping[NodeId, NodeProfile]


class NodeStatusState(State[NodeStatusEventTypes]):
    node_status: Mapping[NodeId, NodeStatus]


class NetworkState(State[EdgeEventTypes]):
    topology: Topology
    history: Sequence[OrphanedPartOfTopology]

    def delete_edge(self, edge_id: EdgeId) -> None: ...
    def add_edge(
        self, edge: NetworkEdge[AddressingProtocol, ApplicationProtocol]
    ) -> None: ...


class MasterState(SharedState):
    network_state: NetworkState
    node_profiles: NodeProfileState
    node_status: NodeStatusState
    job_inbox: Queue[ExternalCommand]
    job_outbox: Queue[ExternalCommand]
    cache_policy: CachePolicy[CachePolicyType]


def get_inference_plan(
    inbox: Queue[ExternalCommand],
    outbox: Queue[ExternalCommand],
    resource_graph: ResourceGraph,
    current_instances: Mapping[InstanceId, InstanceData],
    cache_policy: CachePolicy[CachePolicyType],
) -> Mapping[InstanceId, InstanceData]: ...


TransitionEventTypes = EventTypes


def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceData],
    target_instances: Mapping[InstanceId, InstanceData],
) -> Sequence[Event[TransitionEventTypes]]: ...
