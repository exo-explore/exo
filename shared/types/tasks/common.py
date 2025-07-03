from collections.abc import Mapping
from enum import Enum
from typing import Generic, Literal, TypeVar, Union

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


class TaskArtifact(BaseModel, Generic[TaskTypeT]): ...


class TaskUpdate(BaseModel, Generic[TaskStatusTypeT, TaskTypeT]):
    task_status: TaskStatusTypeT


class PendingTask(TaskUpdate[TaskStatusType.Pending, TaskTypeT]):
    task_status: Literal[TaskStatusType.Pending]


class RunningTask(TaskUpdate[TaskStatusType.Running, TaskTypeT]):
    task_status: Literal[TaskStatusType.Running]


class CompletedTask(TaskUpdate[TaskStatusType.Complete, TaskTypeT]):
    task_status: Literal[TaskStatusType.Complete]
    task_artifact: TaskArtifact[TaskTypeT]


class FailedTask(TaskUpdate[TaskStatusType.Failed, TaskTypeT]):
    task_status: Literal[TaskStatusType.Failed]
    error_message: Mapping[RunnerId, str]


class BaseTask(BaseModel, Generic[TaskTypeT]):
    task_data: TaskData[TaskTypeT]
    task_status: TaskUpdate[TaskStatusType, TaskTypeT]
    on_instance: InstanceId


class Task(BaseTask[TaskTypeT]):
    task_id: TaskId
