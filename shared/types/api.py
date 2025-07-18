from typing import Literal

from pydantic import BaseModel

from shared.types.tasks.common import CompletionCreateParams, TaskId


class ChatTask(BaseModel):
    task_id: TaskId
    kind: Literal["chat"] = "chat"
    task_data: CompletionCreateParams
