from collections.abc import Sequence
from typing import NamedTuple

from pydantic import model_validator

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Id, NodeId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel
from exo.worker.runner.diagnostics import KnownRunnerDiagnostic


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
    layers_loaded: int = 0
    total_layers: int = 0


class RunnerLoaded(BaseRunnerStatus):
    pass


class RunnerWarmingUp(BaseRunnerStatus):
    pass


class RunnerReady(BaseRunnerStatus):
    prefill_server_port: int | None = None


class RunnerRunning(BaseRunnerStatus):
    pass


class RunnerShuttingDown(BaseRunnerStatus):
    pass


class RunnerShutdown(BaseRunnerStatus):
    pass


class RunnerFailed(BaseRunnerStatus):
    error_message: str | None = None
    diagnostics: list[KnownRunnerDiagnostic]


RunnerStatus = (
    RunnerIdle
    | RunnerConnecting
    | RunnerConnected
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerShuttingDown
    | RunnerShutdown
    | RunnerFailed
)


class ShardWithId(NamedTuple):
    node_id: NodeId
    runner_id: RunnerId
    shard: ShardMetadata


class ShardAssignments(FrozenModel):
    model_id: ModelId
    shards: Sequence[ShardWithId]
    # this node needs to be connected to the API node for the stream to be considered ready
    # (this is a device rank)
    primary_output_node: int

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        for position, shard in enumerate(self.shards):
            if shard.shard.device_rank != position:
                raise ValueError("shard position does not correspond to device rank")

        if not self.shards[self.primary_output_node].shard.is_primary_output():
            raise ValueError("primary output node does not correspond to primary shard")

        return self
