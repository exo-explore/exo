from collections.abc import Mapping, Sequence
from enum import Enum
from queue import Queue
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter

from shared.types.common import NodeId
from shared.types.events.common import (
    BaseEvent,
    EventCategory,
    EventCategoryEnum,
    State,
)
from shared.types.graphs.resource_graph import ResourceGraph
from shared.types.networking.data_plane import (
    DataPlaneEdge,
    DataPlaneEdgeAdapter,
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
from shared.types.tasks.common import TaskParams, TaskType
from shared.types.worker.common import NodeStatus
from shared.types.worker.instances import InstanceId, InstanceParams


class ExternalCommand(BaseModel): ...


class CachePolicyType(str, Enum):
    KeepAll = "KeepAll"


CachePolicyTypeT = TypeVar("CachePolicyTypeT", bound=CachePolicyType)


class CachePolicy(BaseModel, Generic[CachePolicyTypeT]):
    policy_type: CachePolicyTypeT


class NodePerformanceProfileState(State[EventCategoryEnum.MutatesNodePerformanceState]):
    node_profiles: Mapping[NodeId, NodePerformanceProfile]


class DataPlaneNetworkState(State[EventCategoryEnum.MutatesDataPlaneState]):
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = (
        EventCategoryEnum.MutatesDataPlaneState
    )
    topology: DataPlaneTopology = DataPlaneTopology(
        edge_base=DataPlaneEdgeAdapter, vertex_base=TypeAdapter(None)
    )
    history: Sequence[OrphanedPartOfDataPlaneTopology] = []

    def delete_edge(self, edge_id: DataPlaneEdgeId) -> None: ...
    def add_edge(self, edge: DataPlaneEdge) -> None: ...


class ControlPlaneNetworkState(State[EventCategoryEnum.MutatesControlPlaneState]):
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = (
        EventCategoryEnum.MutatesControlPlaneState
    )
    topology: ControlPlaneTopology = ControlPlaneTopology(
        edge_base=TypeAdapter(None), vertex_base=TypeAdapter(NodeStatus)
    )
    history: Sequence[OrphanedPartOfControlPlaneTopology] = []

    def delete_edge(self, edge_id: DataPlaneEdgeId) -> None: ...
    def add_edge(self, edge: DataPlaneEdge) -> None: ...


class MasterState(SharedState):
    data_plane_network_state: DataPlaneNetworkState = DataPlaneNetworkState()
    control_plane_network_state: ControlPlaneNetworkState = ControlPlaneNetworkState()
    job_inbox: Queue[TaskParams[TaskType]] = Queue()
    job_outbox: Queue[TaskParams[TaskType]] = Queue()
    cache_policy: CachePolicy[CachePolicyType] = CachePolicy[CachePolicyType](
        policy_type=CachePolicyType.KeepAll
    )


def get_shard_assignments(
    inbox: Queue[ExternalCommand],
    outbox: Queue[ExternalCommand],
    resource_graph: ResourceGraph,
    current_instances: Mapping[InstanceId, InstanceParams],
    cache_policy: CachePolicy[CachePolicyType],
) -> Mapping[InstanceId, InstanceParams]: ...


def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceParams],
    target_instances: Mapping[InstanceId, InstanceParams],
) -> Sequence[BaseEvent[EventCategory]]: ...
