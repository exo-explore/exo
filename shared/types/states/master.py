from collections.abc import Mapping, Sequence
from enum import Enum
from queue import Queue
from typing import Generic, TypeVar

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import (
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
    Event,
    EventTypes,
    NodePerformanceEventTypes,
    State,
)
from shared.types.graphs.resource_graph import ResourceGraph
from shared.types.networking.data_plane import (
    AddressingProtocol,
    ApplicationProtocol,
    DataPlaneEdge,
    DataPlaneEdgeId,
)
from shared.types.networking.topology import (
    ControlPlaneTopology,
    DataPlaneTopology,
    OrphanedPartOfControlPlaneTopology,
    OrphanedPartOfDataPlaneTopology,
)
from shared.types.profiling.common import NodePerformanceProfile
from shared.types.states.shared import SharedState
from shared.types.worker.instances import InstanceData, InstanceId


class ExternalCommand(BaseModel): ...


class CachePolicyType(str, Enum):
    KeepAll = "KeepAll"


CachePolicyTypeT = TypeVar("CachePolicyTypeT", bound=CachePolicyType)


class CachePolicy(BaseModel, Generic[CachePolicyTypeT]):
    policy_type: CachePolicyTypeT


class NodePerformanceProfileState(State[NodePerformanceEventTypes]):
    node_profiles: Mapping[NodeId, NodePerformanceProfile]


class DataPlaneNetworkState(State[DataPlaneEventTypes]):
    topology: DataPlaneTopology
    history: Sequence[OrphanedPartOfDataPlaneTopology]

    def delete_edge(self, edge_id: DataPlaneEdgeId) -> None: ...
    def add_edge(
        self, edge: DataPlaneEdge[AddressingProtocol, ApplicationProtocol]
    ) -> None: ...


class ControlPlaneNetworkState(State[ControlPlaneEventTypes]):
    topology: ControlPlaneTopology
    history: Sequence[OrphanedPartOfControlPlaneTopology]

    def delete_edge(self, edge_id: DataPlaneEdgeId) -> None: ...
    def add_edge(
        self, edge: DataPlaneEdge[AddressingProtocol, ApplicationProtocol]
    ) -> None: ...


class MasterState(SharedState):
    data_plane_network_state: DataPlaneNetworkState
    control_plane_network_state: ControlPlaneNetworkState
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
