from collections.abc import Mapping, Sequence
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from shared.topology import Topology
from shared.types.common import NodeId
from shared.types.profiling import NodePerformanceProfile
from shared.types.tasks import Task, TaskId, TaskSagaEntry
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import BaseInstance
from shared.types.worker.runners import RunnerId, RunnerStatus


class ExternalCommand(BaseModel): ...


class CachePolicy(str, Enum):
    KEEP_ALL = "KEEP_ALL"


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_status: Mapping[NodeId, NodeStatus] = {}
    instances: Mapping[InstanceId, BaseInstance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    tasks: Mapping[TaskId, Task] = {}
    task_sagas: Mapping[TaskId, Sequence[TaskSagaEntry]] = {}
    node_profiles: Mapping[NodeId, NodePerformanceProfile] = {}
    topology: Topology = Topology()
    history: Sequence[Topology] = []
    task_inbox: List[Task] = Field(default_factory=list)
    task_outbox: List[Task] = Field(default_factory=list)
    cache_policy: CachePolicy = CachePolicy.KEEP_ALL
    last_event_applied_idx: int = Field(default=0, ge=0)
