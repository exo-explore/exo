from enum import Enum

from pydantic import Field

from exo.shared.types.api import (
    ChatCompletionTaskParams,
    ImageEditsInternalParams,
    ImageGenerationTaskParams,
)
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


class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata


class DownloadDraftModel(BaseTask):  # emitted by Worker
    """Download a draft model for speculative decoding (rank 0 only)."""

    model_id: str  # HuggingFace model ID


class LoadModel(BaseTask):  # emitted by Worker
    pass


class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


class StartWarmup(BaseTask):  # emitted by Worker
    pass


class ChatCompletion(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ChatCompletionTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageEditsInternalParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


class SetDraftModel(BaseTask):  # emitted by Worker
    """Load or clear a draft model on an already-running instance."""

    model_id: str | None  # HuggingFace model ID, or None to clear
    num_draft_tokens: int = 4


Task = (
    CreateRunner
    | DownloadModel
    | DownloadDraftModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | ChatCompletion
    | ImageGeneration
    | ImageEdits
    | Shutdown
    | SetDraftModel
)
