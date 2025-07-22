from collections.abc import Mapping, Sequence
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.graphs.topology import (
    OrphanedPartOfTopology,
    Topology,
    TopologyEdge,
    TopologyNode,
)
from shared.types.profiling.common import NodePerformanceProfile
from shared.types.tasks.common import Task, TaskId, TaskSagaEntry
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import BaseInstance
from shared.types.worker.runners import RunnerId, RunnerStatus


class ExternalCommand(BaseModel): ...


class CachePolicy(str, Enum):
    KeepAll = "KeepAll"


class State(BaseModel):
    node_status: Mapping[NodeId, NodeStatus] = {}
    instances: Mapping[InstanceId, BaseInstance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    tasks: Mapping[TaskId, Task] = {}
    task_sagas: Mapping[TaskId, Sequence[TaskSagaEntry]] = {}
    node_profiles: Mapping[NodeId, NodePerformanceProfile] = {}
    topology: Topology = Topology(
        edge_base=TypeAdapter(TopologyEdge), vertex_base=TypeAdapter(TopologyNode)
    )
    history: Sequence[OrphanedPartOfTopology] = []
    task_inbox: List[Task] = Field(default_factory=list)
    task_outbox: List[Task] = Field(default_factory=list)
    cache_policy: CachePolicy = CachePolicy.KeepAll

    # TODO: implement / use this? 
    last_event_applied_idx: int = Field(default=0, ge=0)
