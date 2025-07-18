from enum import Enum
from typing import (  # noqa: E402
    Annotated,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    final,
)

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NewUUID
from shared.types.worker.common import InstanceId


class TaskId(NewUUID):
    pass


## TASK TYPES
@final
class TaskType(str, Enum):
    ChatCompletion = "ChatCompletion"

TaskTypeT = TypeVar("TaskTypeT", bound=TaskType, covariant=True)

## TASK STATUSES
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
TaskStatusTypeT = TypeVar("TaskStatusTypeT", bound=TaskStatusType)#, covariant=True)


## Peripherals
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None

class CompletionCreateParams(BaseModel):
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


## Task Data is stored in task, one-to-one with task type

class BaseTaskData(BaseModel, Generic[TaskTypeT]): ...

@final
class ChatCompletionTaskData(BaseTaskData[TaskType.ChatCompletion]):
    task_type: Literal[TaskType.ChatCompletion] = (
        TaskType.ChatCompletion
    )
    task_params: CompletionCreateParams

TaskData: TypeAlias = ChatCompletionTaskData


## TASKS

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
    task_data: TaskData # Really this should be BaseTaskData[TaskTypeT], but this causes a bunch of errors that I don't know how to fix yet.
    task_state: TaskState[TaskStatusTypeT, TaskTypeT]
    on_instance: InstanceId


BaseTaskAnnotated = Annotated[
    Union[
        BaseTask[Literal[TaskType.ChatCompletion], TaskStatusType],
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