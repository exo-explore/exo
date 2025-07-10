from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId, RunnerId


class TaskId(NewUUID):
    pass


class TaskType(str, Enum):
    ChatCompletionNonStreaming = "ChatCompletionNonStreaming"
    ChatCompletionStreaming = "ChatCompletionStreaming"


TaskTypeT = TypeVar("TaskTypeT", bound=TaskType, covariant=True)


class BaseTaskData(BaseModel, Generic[TaskTypeT]):
    task_type: TaskTypeT


# Custom message types that mirror OpenAI's but are designed for serialization
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionParams(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None

class ChatCompletionNonStreamingTask(BaseTaskData[TaskType.ChatCompletionNonStreaming]):
    task_type: Literal[TaskType.ChatCompletionNonStreaming] = (
        TaskType.ChatCompletionNonStreaming
    )
    task_data: ChatCompletionParams


class ChatCompletionStreamingTask(BaseTaskData[TaskType.ChatCompletionStreaming]):
    task_type: Literal[TaskType.ChatCompletionStreaming] = (
        TaskType.ChatCompletionStreaming
    )
    task_data: ChatCompletionParams


TaskData = Annotated[
    ChatCompletionNonStreamingTask | ChatCompletionStreamingTask,
    Field(discriminator="task_type"),
]

TaskDataValidator: TypeAdapter[TaskData] = TypeAdapter(TaskData)


class TaskStatusIncompleteType(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Failed = "Failed"


class TaskStatusCompleteType(str, Enum):
    Complete = "Complete"


TaskStatusType = Union[TaskStatusIncompleteType, TaskStatusCompleteType]


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


class TaskState[TaskStatusTypeT: TaskStatusType, TaskTypeT: TaskType](BaseModel):
    task_status: TaskStatusUpdate[TaskStatusTypeT]
    task_artifact: TaskArtifact[TaskTypeT, TaskStatusTypeT]


class BaseTask[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](BaseModel):
    task_type: TaskTypeT
    task_data: TaskData
    task_state: TaskState[TaskStatusTypeT, TaskTypeT]
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


class Task[TaskTypeT: TaskType, TaskStatusTypeT: TaskStatusType](
    BaseTask[TaskTypeT, TaskStatusTypeT]
):
    task_id: TaskId
