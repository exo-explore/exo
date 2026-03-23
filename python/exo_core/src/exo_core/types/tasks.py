from enum import Enum

from pydantic import Field

from exo_core.models import TaggedModel

from .common import CommandId, Id
from .image_generation import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from .instances import BoundInstance, InstanceId
from .runners import RunnerId
from .shards import ShardMetadata
from .text_generation import TextGenerationTaskParams


class TaskId(Id):
    pass


CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")


class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"
    Cancelled = "Cancelled"


class BaseTask(TaggedModel):
    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata


class LoadModel(BaseTask):  # emitted by Worker
    pass


class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


class StartWarmup(BaseTask):  # emitted by Worker
    pass


class TextGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: TextGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class CancelTask(BaseTask):
    cancelled_task_id: TaskId
    runner_id: RunnerId


class ImageGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageEditsTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


Task = (
    CreateRunner
    | DownloadModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | TextGeneration
    | CancelTask
    | ImageGeneration
    | ImageEdits
    | Shutdown
)
