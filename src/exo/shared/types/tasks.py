from enum import Enum

from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, Id
from exo.shared.types.worker.common import InstanceId
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    pass
    

class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    Failed = "Failed"


class ChatCompletionTask(TaggedModel):
    task_id: TaskId
    command_id: CommandId
    instance_id: InstanceId
    task_status: TaskStatus
    task_params: ChatCompletionTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


Task = ChatCompletionTask
