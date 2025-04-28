from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

from exo.api.inference_result_manager import InferenceResultChunk


class UnplacedToolCall(BaseModel):
  name: str
  arguments: str


class ToolParser(ABC):
  @abstractmethod
  def is_start_of_tool_section(self, chunk: InferenceResultChunk):
    ...

  @abstractmethod
  def parse_complete(self, text: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
    """
    Parse
    """
    ...

  @abstractmethod
  def to_grammar(self, tools: list[Any], required: bool, parallel_tool_calling: bool) -> str:
    ...


def get_tool_parser_by_name(name: str) -> ToolParser:
  if name == "llama_python_tag":
    from exo.tools.llama_python_tag_tool_parser import LlamaPythonTag
    return LlamaPythonTag()
  elif name == "watt":
    from exo.tools.watt_tool_parser import WattToolParser

    return WattToolParser()
  else:
    raise ValueError(f"Unknown tool parser name: {name}")
