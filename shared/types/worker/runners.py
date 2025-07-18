from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from shared.types.common import NodeId
from shared.types.models.common import ModelId
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import DownloadProgress
from shared.types.worker.shards import ShardMetadata


class RunnerStatusType(str, Enum):
    Assigned = "Assigned"
    Downloading = "Downloading"
    Ready = "Ready"
    Starting = "Starting"
    Loaded = "Loaded"
    Running = "Running"
    Failed = "Failed"


RunnerStatusTypeT = TypeVar("RunnerStatusTypeT", bound=RunnerStatusType, covariant=True)


class BaseRunnerStatus(BaseModel, Generic[RunnerStatusTypeT]):
    runner_status: RunnerStatusTypeT


# Emitted by the Master
class AssignedRunnerStatus(BaseRunnerStatus[RunnerStatusType.Assigned]):
    runner_status: Literal[RunnerStatusType.Assigned] = Field(default=RunnerStatusType.Assigned)

# Emitted by the Worker
class DownloadingRunnerStatus(BaseRunnerStatus[RunnerStatusType.Downloading]):
    runner_status: Literal[RunnerStatusType.Downloading] = Field(default=RunnerStatusType.Downloading)
    download_progress: DownloadProgress

# Emitted by the Worker
class ReadyRunnerStatus(BaseRunnerStatus[RunnerStatusType.Ready]):
    runner_status: Literal[RunnerStatusType.Ready] = Field(default=RunnerStatusType.Ready)

# Emitted by the Master
class StartingRunnerStatus(BaseRunnerStatus[RunnerStatusType.Starting]):
    runner_status: Literal[RunnerStatusType.Starting] = Field(default=RunnerStatusType.Starting)

# Emitted by the Worker
class LoadedRunnerStatus(BaseRunnerStatus[RunnerStatusType.Loaded]):
    runner_status: Literal[RunnerStatusType.Loaded] = Field(default=RunnerStatusType.Loaded)

# Emitted by the Worker
class RunningRunnerStatus(BaseRunnerStatus[RunnerStatusType.Running]):
    runner_status: Literal[RunnerStatusType.Running] = Field(default=RunnerStatusType.Running)

# Emitted by the Worker
class FailedRunnerStatus(BaseRunnerStatus[RunnerStatusType.Failed]):
    runner_status: Literal[RunnerStatusType.Failed] = Field(default=RunnerStatusType.Failed)
    error_message: str | None = None


RunnerStatus = Annotated[
    AssignedRunnerStatus
    | DownloadingRunnerStatus
    | ReadyRunnerStatus
    | StartingRunnerStatus
    | LoadedRunnerStatus
    | RunningRunnerStatus
    | FailedRunnerStatus,
    Field,
]
RunnerStatusParser: TypeAdapter[RunnerStatus] = TypeAdapter(
    RunnerStatus
)


class ShardAssignments(BaseModel):
    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata]
    node_to_runner: Mapping[NodeId, RunnerId]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        for runner_id in self.node_to_runner.values():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                )
        return self
