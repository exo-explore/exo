from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from exo.shared.types.common import NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.shards import ShardMetadata


class RunnerStatusType(str, Enum):
    Downloading = "Downloading"
    Inactive = "Inactive"
    Starting = "Starting"
    Loaded = "Loaded"
    Running = "Running"
    Failed = "Failed"


class BaseRunnerStatus[T: RunnerStatusType](BaseModel):
    runner_status: T


class DownloadingRunnerStatus(BaseRunnerStatus[RunnerStatusType.Downloading]):
    runner_status: Literal[RunnerStatusType.Downloading] = Field(
        default=RunnerStatusType.Downloading
    )
    download_progress: DownloadProgress


class InactiveRunnerStatus(BaseRunnerStatus[RunnerStatusType.Inactive]):
    runner_status: Literal[RunnerStatusType.Inactive] = Field(
        default=RunnerStatusType.Inactive
    )


class StartingRunnerStatus(BaseRunnerStatus[RunnerStatusType.Starting]):
    runner_status: Literal[RunnerStatusType.Starting] = Field(
        default=RunnerStatusType.Starting
    )


class LoadedRunnerStatus(BaseRunnerStatus[RunnerStatusType.Loaded]):
    runner_status: Literal[RunnerStatusType.Loaded] = Field(
        default=RunnerStatusType.Loaded
    )


class RunningRunnerStatus(BaseRunnerStatus[RunnerStatusType.Running]):
    runner_status: Literal[RunnerStatusType.Running] = Field(
        default=RunnerStatusType.Running
    )


class FailedRunnerStatus(BaseRunnerStatus[RunnerStatusType.Failed]):
    runner_status: Literal[RunnerStatusType.Failed] = Field(
        default=RunnerStatusType.Failed
    )
    error_message: str | None = None


RunnerStatus = Annotated[
    DownloadingRunnerStatus
    | InactiveRunnerStatus
    | StartingRunnerStatus
    | LoadedRunnerStatus
    | RunningRunnerStatus
    | FailedRunnerStatus,
    Field,
]
RunnerStatusParser: TypeAdapter[RunnerStatus] = TypeAdapter(RunnerStatus)


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
