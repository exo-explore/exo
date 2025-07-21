from collections.abc import Mapping, Sequence
from enum import Enum
from queue import Queue
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter

from shared.types.common import NodeId
from shared.types.events.common import EventCategoryEnum, State
from shared.types.graphs.topology import (
    OrphanedPartOfTopology,
    Topology,
    TopologyEdge,
    TopologyEdgeId,
    TopologyNode,
)
from shared.types.profiling.common import NodePerformanceProfile
from shared.types.states.shared import SharedState
from shared.types.tasks.common import Task


class ExternalCommand(BaseModel): ...


class CachePolicyType(str, Enum):
    KeepAll = "KeepAll"


CachePolicyTypeT = TypeVar("CachePolicyTypeT", bound=CachePolicyType)


class CachePolicy(BaseModel, Generic[CachePolicyTypeT]):
    policy_type: CachePolicyTypeT


class NodePerformanceProfileState(State[EventCategoryEnum.MutatesNodePerformanceState]):
    node_profiles: Mapping[NodeId, NodePerformanceProfile]


class TopologyState(State[EventCategoryEnum.MutatesTopologyState]):
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = (
        EventCategoryEnum.MutatesTopologyState
    )
    topology: Topology = Topology(
        edge_base=TypeAdapter(TopologyEdge), vertex_base=TypeAdapter(TopologyNode)
    )
    history: Sequence[OrphanedPartOfTopology] = []

    def delete_edge(self, edge_id: TopologyEdgeId) -> None: ...
    def add_edge(self, edge: TopologyEdge) -> None: ...


class MasterState(SharedState):
    topology_state: TopologyState = TopologyState()
    task_inbox: Queue[Task] = Queue()
    task_outbox: Queue[Task] = Queue()
    cache_policy: CachePolicy[CachePolicyType] = CachePolicy[CachePolicyType](
        policy_type=CachePolicyType.KeepAll
    )
