from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar, Union, final

import openai.types.chat as openai
from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId


class TaskId(NewUUID):
    pass


@final
class TaskType(str, Enum):
    ChatCompletionNonStreaming = "ChatCompletionNonStreaming"
    ChatCompletionStreaming = "ChatCompletionStreaming"


TaskTypeT = TypeVar("TaskTypeT", bound=TaskType, covariant=True)


class TaskParams(BaseModel, Generic[TaskTypeT]): ...


@final
class ChatCompletionNonStreamingTask(TaskParams[TaskType.ChatCompletionNonStreaming]):
    task_type: Literal[TaskType.ChatCompletionNonStreaming] = (
        TaskType.ChatCompletionNonStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams


@final
class ChatCompletionStreamingTask(TaskParams[TaskType.ChatCompletionStreaming]):
    task_type: Literal[TaskType.ChatCompletionStreaming] = (
        TaskType.ChatCompletionStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams


@final
class TaskStatusFailedType(str, Enum):
    Failed = "Failed"


@final
class TaskStatusCompleteType(str, Enum):
    Complete = "Complete"


@final
class TaskStatusOtherType(str, Enum):
    Pending = "Pending"
    Running = "Running"


TaskStatusType = TaskStatusCompleteType | TaskStatusFailedType | TaskStatusOtherType


class TaskArtifact[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](BaseModel): ...


@final
class NoTaskArtifact[TaskTypeT: TaskType](TaskArtifact[TaskTypeT, TaskStatusOtherType]):
    pass


@final
class FailedTaskArtifact[TaskTypeT: TaskType](
    TaskArtifact[TaskTypeT, TaskStatusFailedType]
):
    error_message: str


@final
class TaskState[TaskStatusTypeT: TaskStatusType, TaskTypeT: TaskType](BaseModel):
    task_status: TaskStatusTypeT
    task_artifact: TaskArtifact[TaskTypeT, TaskStatusTypeT]


class BaseTask[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](BaseModel):
    task_type: TaskTypeT
    task_params: TaskParams[TaskTypeT]
    task_stats: TaskState[TaskStatusTypeT, TaskTypeT]
    on_instance: InstanceId


BaseTaskAnnotated = Annotated[
    Union[
        BaseTask[Literal[TaskType.ChatCompletionNonStreaming], TaskStatusType],
        BaseTask[Literal[TaskType.ChatCompletionStreaming], TaskStatusType],
    ],
    Field(discriminator="task_type"),
]

BaseTaskParser: TypeAdapter[BaseTask[TaskType, TaskStatusType]] = TypeAdapter(
    BaseTaskAnnotated
)


class TaskSagaEntry(BaseModel):
    task_id: TaskId
    instance_id: InstanceId


@final
class Task[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](
    BaseTask[TaskTypeT, TaskStatusTypeT]
):
    task_id: TaskId
