from typing import Literal, Optional, Any
import json

from pydantic import BaseModel, TypeAdapter
from abc import ABC
from exo.inference.grammars import JSON_LARK_GRAMMAR, json_object_grammar, json_schema_grammar


class ResponseFormatBase(ABC, BaseModel):
  type: str

  def to_grammar(self) -> Optional[str]:
    raise NotImplementedError()


class TextResponseFormat(ResponseFormatBase):
  type: Literal["text"]

  def to_grammar(self) -> Optional[str]:
    return None


class JsonObjectResponseFormat(ResponseFormatBase):
  type: Literal["json_object"]

  def to_grammar(self) -> Optional[str]:
    return json_object_grammar()


class JsonSchemaResponseFormat(ResponseFormatBase):
  type: Literal["json_schema"]
  json_schema: Any

  def to_grammar(self) -> Optional[str]:
    return json_schema_grammar(self.json_schema)


# Aligns with https://github.com/guidance-ai/llgtrt
class LarkGrammarResponseFormat(ResponseFormatBase):
  type: Literal["lark_grammar"]
  lark_grammar: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": self.lark_grammar}]
    })


class RegexResponseFormat(ResponseFormatBase):
  type: Literal["regex"]
  regex: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": f"start: /{self.regex}/"}]
    })


ResponseFormat = (TextResponseFormat |
                  JsonObjectResponseFormat | JsonSchemaResponseFormat |
                  LarkGrammarResponseFormat | RegexResponseFormat)

ResponseFormatAdapter = TypeAdapter(ResponseFormat)
