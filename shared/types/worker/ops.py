from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field

from shared.types.events.events import InstanceId
from shared.types.tasks import Task
from shared.types.worker.common import RunnerId
from shared.types.worker.mlx import Host
from shared.types.worker.shards import ShardMetadata


class RunnerOpType(str, Enum):
    ASSIGN_RUNNER = "assign_runner"
    UNASSIGN_RUNNER = "unassign_runner"
    RUNNER_UP = "runner_up"
    RUNNER_DOWN = "runner_down"
    DOWNLOAD = "download"
    CHAT_COMPLETION = "chat_completion"

RunnerOpT = TypeVar("RunnerOpT", bound=RunnerOpType)

class BaseRunnerOp(BaseModel, Generic[RunnerOpT]):
    op_type: RunnerOpT

class AssignRunnerOp(BaseRunnerOp[Literal[RunnerOpType.ASSIGN_RUNNER]]):
    op_type: Literal[RunnerOpType.ASSIGN_RUNNER] = Field(default=RunnerOpType.ASSIGN_RUNNER, frozen=True)
    instance_id: InstanceId
    runner_id: RunnerId
    shard_metadata: ShardMetadata
    hosts: list[Host]

class UnassignRunnerOp(BaseRunnerOp[Literal[RunnerOpType.UNASSIGN_RUNNER]]):
    op_type: Literal[RunnerOpType.UNASSIGN_RUNNER] = Field(default=RunnerOpType.UNASSIGN_RUNNER, frozen=True)
    runner_id: RunnerId

class RunnerUpOp(BaseRunnerOp[Literal[RunnerOpType.RUNNER_UP]]):
    op_type: Literal[RunnerOpType.RUNNER_UP] = Field(default=RunnerOpType.RUNNER_UP, frozen=True)
    runner_id: RunnerId

class RunnerDownOp(BaseRunnerOp[Literal[RunnerOpType.RUNNER_DOWN]]):
    op_type: Literal[RunnerOpType.RUNNER_DOWN] = Field(default=RunnerOpType.RUNNER_DOWN, frozen=True)
    runner_id: RunnerId

class DownloadOp(BaseRunnerOp[Literal[RunnerOpType.DOWNLOAD]]):
    op_type: Literal[RunnerOpType.DOWNLOAD] = Field(default=RunnerOpType.DOWNLOAD, frozen=True)
    instance_id: InstanceId
    runner_id: RunnerId
    shard_metadata: ShardMetadata
    hosts: list[Host]

class ExecuteTaskOp(BaseRunnerOp[Literal[RunnerOpType.CHAT_COMPLETION]]):
    op_type: Literal[RunnerOpType.CHAT_COMPLETION] = Field(default=RunnerOpType.CHAT_COMPLETION, frozen=True)
    runner_id: RunnerId
    task: Task


# Aggregate all runner operations into a single, strictly-typed union for dispatching.
RunnerOp = Annotated[
    Union[
        AssignRunnerOp,
        UnassignRunnerOp,
        RunnerUpOp,
        RunnerDownOp,
        DownloadOp,
        ExecuteTaskOp,
    ],
    Field(discriminator="op_type")
]