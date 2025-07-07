from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Generic, Literal, TypeVar, Self

from pydantic import BaseModel, model_validator

from shared.types.common import NodeId
from shared.types.models.common import ModelId
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import BaseDownloadProgress, DownloadStatus
from shared.types.worker.shards import BaseModelShardMeta, PartitionStrategyT


class RunnerStateType(str, Enum):
    Rejected = "Rejected"
    Starting = "Starting"
    Downloading = "Downloading"
    Running = "Running"
    Failed = "Failed"


RunnerStateTypeT = TypeVar("RunnerStateTypeT", bound=RunnerStateType)


class RunnerState(BaseModel, Generic[RunnerStateTypeT]):
    runner_state: RunnerStateTypeT


class RejectedRunnerState(RunnerState[RunnerStateType.Rejected]):
    runner_state: Literal[RunnerStateType.Rejected]


class StartingRunnerState(RunnerState[RunnerStateType.Starting]):
    runner_state: Literal[RunnerStateType.Starting]


class DownloadingRunnerState(RunnerState[RunnerStateType.Downloading]):
    runner_state: Literal[RunnerStateType.Downloading]
    download_progress: BaseDownloadProgress[DownloadStatus]


class RunningRunnerState(RunnerState[RunnerStateType.Running]):
    runner_state: Literal[RunnerStateType.Running]


class FailedRunnerState(RunnerState[RunnerStateType.Failed]):
    runner_state: Literal[RunnerStateType.Failed]
    error_message: str | None = None


class RunnerData(BaseModel):
    runner_id: RunnerId
    runner_state: RunnerState[RunnerStateType] = RunnerState(
        runner_state=RunnerStateType.Starting
    )


# Runner placement must be consistent in its partitioning strategy across all shards.
# Using a generic type parameter enforces this constraint at type-checking time.


class RunnerPlacement(BaseModel, Generic[PartitionStrategyT]):
    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, BaseModelShardMeta[PartitionStrategyT]]
    node_to_runner: Mapping[NodeId, Sequence[RunnerId]]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> Self:
        for runners in self.node_to_runner.values():
            for runner_id in runners:
                if runner_id not in self.runner_to_shard:
                    raise ValueError(
                        f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                    )
        return self
