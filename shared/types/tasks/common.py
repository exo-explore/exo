from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

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


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionTaskParams(BaseModel):
    task_type: Literal[TaskType.ChatCompletion] = TaskType.ChatCompletion
    model: str
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
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


class Task(BaseModel):
    task_id: TaskId
    instance_id: InstanceId
    task_type: TaskType
    task_status: TaskStatus
    task_params: ChatCompletionTaskParams


class TaskSagaEntry(BaseModel):
    task_id: TaskId
    instance_id: InstanceId
