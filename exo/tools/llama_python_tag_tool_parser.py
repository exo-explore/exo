import json
import re
from typing import Any

from exo.api.chat_completion_request import ToolDefinition
from exo.api.inference_result_manager import InferenceResultChunk
from exo.helpers import DEBUG
from exo.inference.grammars import lark_grammar
from exo.tools.tool_parser import ToolParser, UnplacedToolCall


class LlamaPythonTag(ToolParser):
  def is_start_of_tool_section(self, chunk: InferenceResultChunk):
    return chunk.tokens[0] == 128010

  def to_grammar(self, tools: list[ToolDefinition], required: bool, parallel_tool_calling: bool) -> str:
    tool_call_schema = generate_tool_call_json_schema(tools, "parameters")

    # This implements a parallel format of <|python_tag|> json array of tool calls <|eom_id|>.
    # I think this is the best option as the other obvious alternative is <|python_tag|> tool call 1 <|eom_id|><|python_tag|> tool call 2 <|eom_id|>.
    # However, the model has been observed to repeat the same tool call multiple times in a row when not stopped at <|eom_id|> so this is not ideal.
    parallel_tool_call_schema = {
      "type": "array",
      "items": tool_call_schema
    }

    function_call_production = "parallel_fun_call" if parallel_tool_calling else "single_fun_call"
    entry_production = function_call_production if required else "TEXT | single_fun_call"

    # This is lifted from https://github.com/guidance-ai/llguidance/blob/cc83715f/docs/syntax.md#special-tokens
    return lark_grammar(f"""
%llguidance {{}}

start: {entry_production}
TEXT: /[^{{](.|\n)*/
single_fun_call: <|python_tag|> json_body <|eom_id|>
parallel_fun_call: <|python_tag|> parallel_json_body <|eom_id|>
json_body: %json{json.dumps(tool_call_schema)}
parallel_json_body: %json{json.dumps(parallel_tool_call_schema)}
    """.strip())

  def parse_complete(self, content: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
    tool_calls = []

    for m in re.finditer(r"<\|python_tag\|>(.+)<\|eom_id\|>", content, re.DOTALL):
      try:
        if parallel_tool_calling:
          array = json.loads(m.group(1))

          for raw_tool_call in array:
            tool_calls.append(UnplacedToolCall(
              name=raw_tool_call["name"],
              arguments=json.dumps(raw_tool_call["parameters"])
            ))
        else:
          raw_tool_call = json.loads(m.group(1))

          # Rename "parameters" to "arguments" as that is the expected format
          tool_calls.append(UnplacedToolCall(
            name=raw_tool_call["name"],
            arguments=json.dumps(raw_tool_call["parameters"])
          ))
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse python_tag tool calls: {e}")

    return tool_calls


def generate_tool_call_json_schema(tools: list[ToolDefinition], parameter_key: str = "arguments") -> dict[str, Any]:
  """
  Generate a JSON schema for tool calling. For a given tool name, the schema should have the rough form of:

  type ValidToolCall[name] = {
    "name": name,
    "arguments": tools[name].strict ? tools[name].parameters : object
  }

  With the overall schema looking like:

  // For each tool in the list
  type ValidToolCall = ValidToolCall[name] | ...;

  Ie it should be a union of all the tool calls, disjoint from each other by the unqiue "name" field.
  """
  if len(tools) == 0:
    raise ValueError("No tools provided")

  schema_variants = []

  for tool in tools:
    # Create a schema variant for this tool
    tool_schema = {
      "type": "object",
      "properties": {
        "name": { "const": tool.function.name },
        parameter_key: tool.function.parameters if getattr(tool.function, "strict", False) else {
          "type": "object"
        }
      },
      "required": ["name", parameter_key],
      "additionalProperties": False
    }
    schema_variants.append(tool_schema)

  # Combine all tool schemas into a oneOf union
  if len(schema_variants) == 1:
    # Just return the single schema if only one tool
    return schema_variants[0]
  else:
    # Return a union of all tool schemas
    return {"oneOf": schema_variants}
