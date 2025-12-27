from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.types.common import Id, NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class RunnerId(Id):
    pass


class RunnerError(Exception):
    pass


class BaseRunnerStatus(TaggedModel):
    def is_running(self):
        return isinstance(self, RunnerRunning)


class RunnerIdle(BaseRunnerStatus):
    pass


class RunnerConnecting(BaseRunnerStatus):
    pass


class RunnerConnected(BaseRunnerStatus):
    pass


class RunnerLoading(BaseRunnerStatus):
    pass


class RunnerLoaded(BaseRunnerStatus):
    pass


class RunnerWarmingUp(BaseRunnerStatus):
    pass


class RunnerReady(BaseRunnerStatus):
    pass


class RunnerRunning(BaseRunnerStatus):
    pass


class RunnerShutdown(BaseRunnerStatus):
    pass


class RunnerFailed(BaseRunnerStatus):
    error_message: str | None = None


RunnerStatus = (
    RunnerIdle
    | RunnerConnecting
    | RunnerConnected
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerShutdown
    | RunnerFailed
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
