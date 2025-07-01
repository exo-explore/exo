from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

import openai.types.chat as openai
from pydantic import BaseModel, TypeAdapter
from pydantic.types import UuidVersion

_TaskId = Annotated[UUID, UuidVersion(4)]
TaskId = type("TaskId", (UUID,), {})
TaskIdParser: TypeAdapter[TaskId] = TypeAdapter(_TaskId)


class TaskType(str, Enum):
    ChatCompletionNonStreaming = "ChatCompletionNonStreaming"
    ChatCompletionStreaming = "ChatCompletionStreaming"


TaskTypeT = TypeVar("TaskTypeT", bound=TaskType)


class Task(BaseModel, Generic[TaskTypeT]):
    task_id: TaskId
    task_type: TaskTypeT
    task_data: Any


class ChatCompletionNonStreamingTask(Task[TaskType.ChatCompletionNonStreaming]):
    task_type: Literal[TaskType.ChatCompletionNonStreaming] = (
        TaskType.ChatCompletionNonStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams


class ChatCompletionStreamingTask(Task[TaskType.ChatCompletionStreaming]):
    task_type: Literal[TaskType.ChatCompletionStreaming] = (
        TaskType.ChatCompletionStreaming
    )
    task_data: openai.completion_create_params.CompletionCreateParams
