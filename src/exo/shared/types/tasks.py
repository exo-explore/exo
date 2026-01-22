from enum import Enum

from pydantic import Field

from exo.plugins.type_registry import task_registry
from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, Id
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    pass


class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"


class BaseTask(TaggedModel):
    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


@task_registry.register
class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


@task_registry.register
class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata


@task_registry.register
class LoadModel(BaseTask):  # emitted by Worker
    pass


@task_registry.register
class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


@task_registry.register
class StartWarmup(BaseTask):  # emitted by Worker
    pass


@task_registry.register
class ChatCompletion(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ChatCompletionTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


@task_registry.register
class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


# Type alias for backward compatibility - use BaseTask for type hints
# Actual deserialization uses task_registry
Task = BaseTask
