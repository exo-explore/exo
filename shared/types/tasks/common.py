from enum import Enum

from pydantic import BaseModel

from shared.types.api import ChatCompletionTaskParams
from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId


class TaskId(NewUUID):
    pass

class TaskType(str, Enum):
    ChatCompletion = "ChatCompletion"

class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    Failed = "Failed"


class Task(BaseModel):
    task_id: TaskId
    task_type: TaskType # redundant atm as we only have 1 task type.
    instance_id: InstanceId
    task_status: TaskStatus
    task_params: ChatCompletionTaskParams


class TaskSagaEntry(BaseModel):
    task_id: TaskId
    instance_id: InstanceId
