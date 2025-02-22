from typing import Optional, List, Literal, Any

from pydantic import BaseModel, TypeAdapter
from typing import Optional


class TextResponseFormat(BaseModel):
  type: Literal["text"]


class JsonObjectResponseFormat(BaseModel):
  type: Literal["json_object"]


class JsonSchemaResponseFormat(BaseModel):
  type: Literal["json_schema"]
  json_schema: Any


ResponseFormat = TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat
ResponseFormatAdapter = TypeAdapter(ResponseFormat)

class GenerationOptions:
  max_completion_tokens: Optional[int] = None

  # Textual stop sequences that will halt generation when encountered
  stop: Optional[List[str]] = None
  temperature: Optional[float] = None
  response_format: Optional[ResponseFormat] = None

  def __init__(
    self,
    max_completion_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    response_format: Optional[ResponseFormat] = None
  ):
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop
    self.temperature = temperature
    self.response_format = response_format
