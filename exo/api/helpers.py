import time
from typing import Union, List, Dict, Optional, Literal, Any

from pydantic import BaseModel

from exo import VERSION, DEBUG
from exo.api.response_formats import ResponseFormat, ResponseFormatAdapter
from exo.inference.generation_options import GenerationOptions
from exo.models import get_default_tool_format
from exo.tools import ToolChoice, ToolChoiceModel, choose_tools, SpecificToolChoice
from exo.tools.tool_parser import ToolParser, get_tool_parser_by_name


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
               tool_choice: Optional[ToolChoice] = None, tool_behaviour: Optional[ToolBehaviour] = None):
    self.model = model
    self.messages = messages
    self.temperature = temperature
    self.tools = tools
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else None
    self.response_format = response_format
    self.tool_choice = tool_choice
    self.tool_behaviour = tool_behaviour

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature, "tools": self.tools, "max_completion_tokens": self.max_completion_tokens,
            "stop": self.stop, "response_format": self.response_format.model_dump() if self.response_format else None, "tool_choice": self.tool_choice, "tool_behaviour": self.tool_behaviour.model_dump() if self.tool_behaviour else None}

  def to_generation_options(self) -> GenerationOptions:
    grammar_definition = None
    tool_parser = self.get_tool_parser()

    if self.response_format and tool_parser:
      raise ValueError("Cannot use response_format and tools at the same time")

    if self.response_format is not None:
      grammar_definition = self.response_format.to_grammar()
    elif tool_parser:
      grammar_definition = tool_parser.to_grammar(self.get_tools(), self.tool_choice == "required" or isinstance(self.tool_choice, SpecificToolChoice))

    return GenerationOptions(max_completion_tokens=self.max_completion_tokens, stop=self.stop, grammar_definition=grammar_definition)

  def get_tools(self):
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
  chat_request: ChatCompletionRequest,
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
    "model": chat_request.model,
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


def remap_messages(messages: List[Message]) -> List[Message]:
  remapped_messages = []
  last_image = None
  for message in messages:
    if not isinstance(message.content, list):
      remapped_messages.append(message)
      continue

    remapped_content = []
    for content in message.content:
      if isinstance(content, dict):
        if content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if image_url:
            last_image = {"type": "image", "image": image_url}
            remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
        else:
          remapped_content.append(content)
      else:
        remapped_content.append(content)
    remapped_messages.append(Message(role=message.role, content=remapped_content))

  if last_image:
    # Replace the last image placeholder with the actual image content
    for message in reversed(remapped_messages):
      for i, content in enumerate(message.content):
        if isinstance(content, dict):
          if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
            message.content[i] = last_image
            return remapped_messages

  return remapped_messages


def build_prompt(tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None):
  messages = remap_messages(_messages)
  chat_template_args = {"conversation": [m.to_dict() for m in messages], "tokenize": False, "add_generation_prompt": True}
  if tools:
    chat_template_args["tools"] = tools

  try:
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt: {prompt}")
    return prompt
  except UnicodeEncodeError:
    # Handle Unicode encoding by ensuring everything is UTF-8
    chat_template_args["conversation"] = [
      {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v
       for k, v in m.to_dict().items()}
      for m in messages
    ]
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt (UTF-8 encoded): {prompt}")
    return prompt


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"], data.get("tools"))


def parse_chat_request(data: dict, default_model: str):
  return ChatCompletionRequest(
    data.get("model", default_model),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
    [ToolDefinition.model_validate(tool) for tool in data["tools"]] if "tools" in data else None,
    # The max_tokens field is deprecated, but some clients may still use it, fall back to that value if
    # max_completion_tokens is not provided.
    data.get("max_completion_tokens", data.get("max_tokens", None)),
    data.get("stop", None),
    response_format=ResponseFormatAdapter.validate_python(data.get("response_format")) if "response_format" in data else None,
    tool_choice=ToolChoiceModel.validate_python(data.get("tool_choice")) if "tool_choice" in data else None,
    tool_behaviour=ToolBehaviour.model_validate(data.get("tool_behaviour")) if "tool_behaviour" in data else None,
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt
