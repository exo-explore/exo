from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.types.common import NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class BaseRunnerStatus(TaggedModel):
    pass


class DownloadingRunnerStatus(BaseRunnerStatus):
    download_progress: DownloadProgress


class InactiveRunnerStatus(BaseRunnerStatus):
    pass


class StartingRunnerStatus(BaseRunnerStatus):
    pass


class LoadedRunnerStatus(BaseRunnerStatus):
    pass


class RunningRunnerStatus(BaseRunnerStatus):
    pass


class FailedRunnerStatus(BaseRunnerStatus):
    error_message: str | None = None


RunnerStatus = (
    DownloadingRunnerStatus
    | InactiveRunnerStatus
    | StartingRunnerStatus
    | LoadedRunnerStatus
    | RunningRunnerStatus
    | FailedRunnerStatus
)


class ShardAssignments(CamelCaseModel):
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
