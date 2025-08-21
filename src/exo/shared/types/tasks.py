from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import ID, CommandId
from exo.shared.types.worker.common import InstanceId


class TaskId(ID):
    pass


class TaskType(str, Enum):
    CHAT_COMPLETION = "CHAT_COMPLETION"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ChatCompletionTask(BaseModel):
    task_type: Literal[TaskType.CHAT_COMPLETION] = TaskType.CHAT_COMPLETION
    task_id: TaskId
    command_id: CommandId
    instance_id: InstanceId
    task_status: TaskStatus
    task_params: ChatCompletionTaskParams

    error_type: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


Task = Annotated[ChatCompletionTask, Field(discriminator="task_type")]
