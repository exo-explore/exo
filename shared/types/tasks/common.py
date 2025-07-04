from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar, Union

import openai.types.chat as openai
from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId, RunnerId


class TaskId(NewUUID):
    pass


class TaskType(str, Enum):
    ChatCompletionNonStreaming = "ChatCompletionNonStreaming"
    ChatCompletionStreaming = "ChatCompletionStreaming"


TaskTypeT = TypeVar("TaskTypeT", bound=TaskType, covariant=True)


class TaskData(BaseModel, Generic[TaskTypeT]): ...


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


class TaskStatusIncompleteType(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Failed = "Failed"


class TaskStatusCompleteType(str, Enum):
    Complete = "Complete"


TaskStatusType = Union[TaskStatusIncompleteType, TaskStatusCompleteType]


TaskStatusTypeT = TypeVar("TaskStatusTypeT", bound=TaskStatusType, covariant=True)


class TaskArtifact[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](BaseModel): ...


class IncompleteTaskArtifact[TaskTypeT: TaskType](
    TaskArtifact[TaskTypeT, TaskStatusIncompleteType]
):
    pass


class TaskStatusUpdate[TaskStatusTypeT: TaskStatusType](BaseModel):
    task_status: TaskStatusTypeT


class PendingTaskStatus(TaskStatusUpdate[TaskStatusIncompleteType.Pending]):
    task_status: Literal[TaskStatusIncompleteType.Pending] = (
        TaskStatusIncompleteType.Pending
    )


class RunningTaskStatus(TaskStatusUpdate[TaskStatusIncompleteType.Running]):
    task_status: Literal[TaskStatusIncompleteType.Running] = (
        TaskStatusIncompleteType.Running
    )


class CompletedTaskStatus(TaskStatusUpdate[TaskStatusCompleteType.Complete]):
    task_status: Literal[TaskStatusCompleteType.Complete] = (
        TaskStatusCompleteType.Complete
    )


class FailedTaskStatus(TaskStatusUpdate[TaskStatusIncompleteType.Failed]):
    task_status: Literal[TaskStatusIncompleteType.Failed] = (
        TaskStatusIncompleteType.Failed
    )
    error_message: Mapping[RunnerId, str]


class TaskState(BaseModel, Generic[TaskTypeT, TaskStatusTypeT]):
    task_status: TaskStatusUpdate[TaskStatusTypeT]
    task_artifact: TaskArtifact[TaskTypeT, TaskStatusTypeT]


class BaseTask(BaseModel, Generic[TaskTypeT, TaskStatusTypeT]):
    task_type: TaskTypeT
    task_data: TaskData[TaskTypeT]
    task_state: TaskState[TaskTypeT, TaskStatusTypeT]
    on_instance: InstanceId


BaseTaskAnnotated = Annotated[
    Union[
        BaseTask[Literal[TaskType.ChatCompletionNonStreaming], TaskStatusType],
        BaseTask[Literal[TaskType.ChatCompletionStreaming], TaskStatusType],
    ],
    Field(discriminator="task_type"),
]

BaseTaskValidator: TypeAdapter[BaseTask[TaskType, TaskStatusType]] = TypeAdapter(
    BaseTaskAnnotated
)


class Task(BaseTask[TaskTypeT, TaskStatusTypeT]):
    task_id: TaskId
