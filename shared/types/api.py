from typing import Literal
from pydantic import BaseModel
from openai.types.chat.completion_create_params import CompletionCreateParams

from shared.types.tasks.common import TaskId

class ChatTask(BaseModel):
  task_id: TaskId
  kind: Literal["chat"] = "chat"
  task_data: CompletionCreateParams