from typing import List, Literal, Union, Any, Optional
from pydantic import BaseModel, TypeAdapter


class SpecificToolChoice(BaseModel):
  class SpecificToolChoiceInner(BaseModel):
    name: str

  type: Literal["function"] = "function"
  function: SpecificToolChoiceInner


ToolChoice = Union[
  # none => ignore tools,
  # auto => model decides whether to use tools,
  # required => model must use tools,
  # specific => model must use specific tools
  Literal["none", "auto", "required"],
  SpecificToolChoice
]

ToolChoiceModel = TypeAdapter(ToolChoice)


def choose_tools(tool_choice: Optional[ToolChoice], tools: List[Any]) -> List[Any]:
  if tool_choice == "none":
    return []
  elif tool_choice == "auto":
    return tools
  elif tool_choice == "required":
    return tools
  elif isinstance(tool_choice, SpecificToolChoice):
    return [tool for tool in tools if tool.name == tool_choice.function.name]
  elif tool_choice is None:
    return tools
  else:
    raise ValueError(f"Invalid tool choice: {tool_choice}")
