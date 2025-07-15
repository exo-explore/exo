from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from shared.types.common import NodeId
from shared.types.models.common import ModelId
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import BaseDownloadProgress, DownloadStatus
from shared.types.worker.shards import PartitionStrategy, ShardMetadata


class RunnerStatusType(str, Enum):
    Rejected = "Rejected"
    Starting = "Starting"
    Downloading = "Downloading"
    Running = "Running"
    Failed = "Failed"


RunnerStatusTypeT = TypeVar("RunnerStatusTypeT", bound=RunnerStatusType)


class RunnerStatus(BaseModel, Generic[RunnerStatusTypeT]):
    runner_status: RunnerStatusTypeT


class RejectedRunnerStatus(RunnerStatus[RunnerStatusType.Rejected]):
    runner_status: Literal[RunnerStatusType.Rejected]


class StartingRunnerStatus(RunnerStatus[RunnerStatusType.Starting]):
    runner_status: Literal[RunnerStatusType.Starting]


class DownloadingRunnerStatus(RunnerStatus[RunnerStatusType.Downloading]):
    runner_status: Literal[RunnerStatusType.Downloading]
    download_progress: BaseDownloadProgress[DownloadStatus]


class RunningRunnerStatus(RunnerStatus[RunnerStatusType.Running]):
    runner_status: Literal[RunnerStatusType.Running]


class FailedRunnerStatus(RunnerStatus[RunnerStatusType.Failed]):
    runner_status: Literal[RunnerStatusType.Failed]
    error_message: str | None = None


_RunnerStatus = Annotated[
    RejectedRunnerStatus
    | StartingRunnerStatus
    | DownloadingRunnerStatus
    | RunningRunnerStatus
    | FailedRunnerStatus,
    Field,
]
RunnerStatusParser: TypeAdapter[RunnerStatus[RunnerStatusType]] = TypeAdapter(
    _RunnerStatus
)


class ShardAssignments(BaseModel):
    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata[PartitionStrategy]]
    node_to_runner: Mapping[NodeId, Sequence[RunnerId]]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        for runners in self.node_to_runner.values():
            for runner_id in runners:
                if runner_id not in self.runner_to_shard:
                    raise ValueError(
                        f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                    )
        return self
