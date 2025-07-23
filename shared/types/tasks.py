from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from shared.types.api import ChatCompletionTaskParams
from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId


class TaskId(NewUUID):
    pass


class TaskType(str, Enum):
    CHAT_COMPLETION = "CHAT_COMPLETION"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ChatCompletionTask(BaseModel):
    task_type: TaskType
    task_id: TaskId
    instance_id: InstanceId
    task_status: TaskStatus
    task_params: ChatCompletionTaskParams

Task = Annotated[ChatCompletionTask, Field(discriminator="task_type")]
