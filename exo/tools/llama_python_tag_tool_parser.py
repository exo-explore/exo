import json
import re
from typing import Any

from exo.api.helpers import ToolDefinition
from exo.api.inference_result_manager import InferenceResultChunk
from exo.helpers import DEBUG
from exo.inference.grammars import lark_grammar
from exo.tools.tool_parser import ToolParser, UnplacedToolCall


class LlamaPythonTag(ToolParser):
  def is_start_of_tool(self, chunk: InferenceResultChunk):
    return chunk.tokens[0] == 128010

  def to_grammar(self, tools: list[ToolDefinition], required: bool) -> str:
    # This is lifted from https://github.com/guidance-ai/llguidance/blob/cc83715f/docs/syntax.md#special-tokens
    return lark_grammar(f"""
%llguidance {{}}

start: {"fun_call" if required else "TEXT | fun_call"}
TEXT: /[^{{](.|\n)*/
fun_call: <|python_tag|> json_body <|eom_id|>
json_body: %json{json.dumps(generate_tool_call_json_schema(tools, "parameters"))}
    """.strip())

  def parse_complete(self, content: str) -> list[UnplacedToolCall]:
    tool_calls = []

    for m in re.finditer(r"<\|python_tag\|>(.+)<\|eom_id\|>", content, re.DOTALL):
      try:
        remapped = json.loads(m.group(1))

        # Rename "parameters" to "arguments" as that is the expected format
        tool_calls.append(UnplacedToolCall(
          name=remapped["name"],
          arguments=json.dumps(remapped["parameters"])
        ))
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse python_tag tool calls: {e}")

    return tool_calls

def generate_tool_call_json_schema(tools: list[ToolDefinition], parameter_key: str = "arguments") -> dict[str, Any]:
  """
  Generate a JSON schema for tool calling. For a given tool name, the schema should have the rough form of:

  type ValidToolCall[name] = {
    "name": name,
    "arguments": tools[name].parameters
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
        # TODO: The LLama example on LLGuidance uses "name": { "const": "get_weather" } which might be easier?
        "name": {
          "type": "string",
          "enum": [tool.function.name]
        },
        parameter_key: tool.function.parameters
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
