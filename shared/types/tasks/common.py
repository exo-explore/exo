from collections.abc import Mapping
from enum import Enum
from typing import Any, Generic, Literal, TypeVar, Union

import openai.types.chat as openai
from pydantic import BaseModel

from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId, RunnerId


class TaskId(NewUUID):
    pass


class TaskType(str, Enum):
    ChatCompletionNonStreaming = "ChatCompletionNonStreaming"
    ChatCompletionStreaming = "ChatCompletionStreaming"


TaskTypeT = TypeVar("TaskTypeT", bound=TaskType)


class TaskData(BaseModel, Generic[TaskTypeT]):
    task_type: TaskTypeT
    task_data: Any


class ChatCompletionNonStreamingTask(TaskData[TaskType.ChatCompletionNonStreaming]):
    task_type: Literal[TaskType.ChatCompletionNonStreaming] = (
        TaskType.ChatCompletionNonStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams


class ChatCompletionStreamingTask(TaskData[TaskType.ChatCompletionStreaming]):
    task_type: Literal[TaskType.ChatCompletionStreaming] = (
        TaskType.ChatCompletionStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams


class TaskStatusType(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Failed = "Failed"
    Complete = "Complete"


TaskStatusTypeT = TypeVar(
    "TaskStatusTypeT", bound=Union[TaskStatusType, Literal["Complete"]]
)


class TaskUpdate(BaseModel, Generic[TaskStatusTypeT]):
    task_status: TaskStatusTypeT


class PendingTask(TaskUpdate[TaskStatusType.Pending]):
    task_status: Literal[TaskStatusType.Pending]


class RunningTask(TaskUpdate[TaskStatusType.Running]):
    task_status: Literal[TaskStatusType.Running]


class CompletedTask(TaskUpdate[TaskStatusType.Complete]):
    task_status: Literal[TaskStatusType.Complete]
    task_artifact: bytes


class FailedTask(TaskUpdate[TaskStatusType.Failed]):
    task_status: Literal[TaskStatusType.Failed]
    error_message: Mapping[RunnerId, str]


class BaseTask(BaseModel):
    task_data: TaskData[TaskType]
    task_status: TaskUpdate[TaskStatusType]
    on_instance: InstanceId


class Task(BaseTask):
    task_id: TaskId
