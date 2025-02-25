from typing import Optional, List, Literal, Any

from pydantic import BaseModel, TypeAdapter
from typing import Optional
import json
from exo.inference.grammars import JSON_GRAMMAR

class ResponseFormat(BaseModel):
  type: Literal["text", "json_object", "json_schema"]

  def to_grammar(self) -> Optional[str]:
    raise NotImplementedError()

  def is_guided(self):
    """
    If the response format requires guided generation. By default, this is true. If this returns true you must return
    a grammar from to_grammar.
    """

    return True

class TextResponseFormat(ResponseFormat):
  type: Literal["text"]

  def is_guided(self):
    return False

  def to_grammar(self) -> Optional[str]:
    return None


class JsonObjectResponseFormat(BaseModel):
  type: Literal["json_object"]

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": JSON_GRAMMAR}]
    })


class JsonSchemaResponseFormat(BaseModel):
  type: Literal["json_schema"]
  json_schema: Any

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"json_schema": self.json_schema}]
    })

class GenerationOptions:
  max_completion_tokens: Optional[int] = None

  # Textual stop sequences that will halt generation when encountered
  stop: Optional[List[str]] = None
  temperature: Optional[float] = None
  grammar_definition: Optional[str] = None

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
    self.grammar_definition = response_format.to_grammar() if response_format else None
