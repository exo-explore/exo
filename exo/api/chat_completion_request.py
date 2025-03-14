from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel

import time

from exo import VERSION
from exo.tools import ToolChoice, ToolChoiceModel, choose_tools, SpecificToolChoice
from exo.tools.tool_parser import ToolParser, get_tool_parser_by_name
from exo.models import get_default_tool_format
from exo.inference.generation_options import GenerationOptions

from exo.api.response_formats import ResponseFormat, ResponseFormatAdapter

class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]], tools: Optional[List[Dict]] = None):
    self.role = role
    self.content = content
    self.tools = tools

  def to_dict(self):
    data = {"role": self.role, "content": self.content}
    if self.tools:
      data["tools"] = self.tools
    return data

class ToolBehaviour(BaseModel):
  format: str
  guided: bool = True
  parsed: bool = True

class ToolDefinition(BaseModel):
  """
  This model maps to elements of the tools array in the request body.
  """
  class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]]
    strict: Optional[bool] = False

  type: Literal["function"]
  function: FunctionDefinition


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float, tools: Optional[List[ToolDefinition]] = None,
               max_completion_tokens: Optional[int] = None, stop: Optional[Union[str, List[str]]] = None, response_format: Optional[ResponseFormat] = None,
               tool_choice: Optional[ToolChoice] = None, tool_behaviour: Optional[ToolBehaviour] = None, parallel_tool_calling: Optional[bool] = None):
    self.model = model
    self.messages = messages
    self.temperature = temperature
    self.tools = tools
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else None
    self.response_format = response_format
    self.tool_choice = tool_choice
    self.tool_behaviour = tool_behaviour
    self.parallel_tool_calling = parallel_tool_calling

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature, "tools": self.tools, "max_completion_tokens": self.max_completion_tokens,
            "stop": self.stop, "response_format": self.response_format.model_dump() if self.response_format else None, "tool_choice": self.tool_choice, "tool_behaviour": self.tool_behaviour.model_dump() if self.tool_behaviour else None,
            "parallel_tool_calling": self.parallel_tool_calling}

  def to_generation_options(self) -> GenerationOptions:
    grammar_definition = None
    tool_parser = self.get_tool_parser()

    if self.response_format and tool_parser:
      raise ValueError("Cannot use response_format and tools at the same time")

    if self.response_format is not None:
      grammar_definition = self.response_format.to_grammar()
    elif tool_parser:
      grammar_definition = tool_parser.to_grammar(self.get_tools(), self.tool_choice == "required" or isinstance(self.tool_choice, SpecificToolChoice), self.parallel_tool_calling)

    return GenerationOptions(max_completion_tokens=self.max_completion_tokens, stop=self.stop, grammar_definition=grammar_definition)

  def get_tools(self):
    ##
    return choose_tools(self.tool_choice, self.tools) or []

  def get_tool_parser(self) -> Optional[ToolParser]:
    if self.tool_behaviour and not self.tool_behaviour.guided:
      return None

    if self.tool_choice == "none":
      return None

    if self.tools is None or len(self.tools) == 0:
      return None

    tool_format = get_default_tool_format(self.model)
    if self.tool_behaviour:
      tool_format = self.tool_behaviour.format or tool_format

    return get_tool_parser_by_name(tool_format)

  def generate_completion(
    self,
    tokenizer,
    prompt: str,
    request_id: str,
    tokens: List[int],
    decoded_tokens: str,
    stream: bool,
    finish_reason: Union[Literal["length", "stop"], None],
    object_type: Literal["chat.completion", "text_completion"],
    ) -> dict:
    completion = {
        "id": f"chatcmpl-{request_id}",
        "object": object_type,
        "created": int(time.time()),
        "model": self.model,
        "system_fingerprint": f"exo_{VERSION}",
        "choices": [{
        "index": 0,
        "logprobs": None,
        "finish_reason": finish_reason,
        }],
    }

    if not stream:
        completion["usage"] = {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "completion_tokens": len(tokens),
        "total_tokens": len(tokenizer.encode(prompt)) + len(tokens),
        }

    choice = completion["choices"][0]
    if object_type.startswith("chat.completion"):
      if stream:
        choice["delta"] = {"role": "assistant", "content": decoded_tokens} if len(decoded_tokens) > 0 else {}
      else:
        choice["message"] = {"role": "assistant", "content": decoded_tokens}
    elif object_type == "text_completion":
        choice["text"] = decoded_tokens
    else:
        ValueError(f"Unsupported response type: {object_type}")

    return completion

  @staticmethod
  def from_chat_request_dict(data: dict, default_model: str):
    return ChatCompletionRequest(
        data.get("model", default_model),
        data["messages"],
        data.get("temperature", 0.0),
        [ToolDefinition.model_validate(tool) for tool in data["tools"]] if "tools" in data else None,
        # The max_tokens field is deprecated, but some clients may still use it, fall back to that value if
        # max_completion_tokens is not provided.
        data.get("max_completion_tokens", data.get("max_tokens", None)),
        data.get("stop", None),
        response_format=ResponseFormatAdapter.validate_python(data.get("response_format")) if "response_format" in data else None,
        tool_choice=ToolChoiceModel.validate_python(data.get("tool_choice")) if "tool_choice" in data else None,
        tool_behaviour=ToolBehaviour.model_validate(data.get("tool_behaviour")) if "tool_behaviour" in data else None,
        parallel_tool_calling=data.get("parallel_tool_calls", False),
    )
