from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

from exo.api.inference_result_manager import InferenceResultChunk


class UnplacedToolCall(BaseModel):
  name: str
  arguments: str


class ToolParser(ABC):
  @abstractmethod
  def is_start_of_tool(self, chunk: InferenceResultChunk):
    ...

  @abstractmethod
  def parse_complete(self, text: str) -> list[UnplacedToolCall]:
    """
    Parse
    """
    ...

  @abstractmethod
  def to_grammar(self, tools: list[Any], required: bool) -> str:
    ...


def get_tool_parser_by_name(name: str) -> ToolParser:
  ...
