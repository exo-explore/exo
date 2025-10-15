from exo.shared.types.common import Host
from exo.shared.types.events import InstanceId
from exo.shared.types.tasks import Task
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerOp(TaggedModel):
    pass


class AssignRunnerOp(BaseRunnerOp):
    instance_id: InstanceId
    runner_id: RunnerId
    shard_metadata: ShardMetadata
    hosts: list[Host]


class UnassignRunnerOp(BaseRunnerOp):
    runner_id: RunnerId


class RunnerUpOp(BaseRunnerOp):
    runner_id: RunnerId


class RunnerDownOp(BaseRunnerOp):
    runner_id: RunnerId


class RunnerFailedOp(BaseRunnerOp):
    runner_id: RunnerId


class ExecuteTaskOp(BaseRunnerOp):
    runner_id: RunnerId
    task: Task


# Aggregate all runner operations into a single, strictly-typed union for dispatching.
RunnerOp = (
    AssignRunnerOp
    | UnassignRunnerOp
    | RunnerUpOp
    | RunnerDownOp
    | RunnerFailedOp
    | ExecuteTaskOp
)
