import json
import re

from exo import DEBUG
from exo.api.inference_result_manager import InferenceResultChunk
from exo.inference.grammars import lark_grammar
from exo.tools.tool_parser import ToolParser, UnplacedToolCall


class WattToolParser(ToolParser):
  def is_start_of_tool_section(self, chunk: InferenceResultChunk):
    return chunk.text[0] == "["

  def to_grammar(self, tools, required: bool, parallel_tool_calling: bool) -> str:
    import os

    with open(os.path.join(os.path.dirname(__file__), "watt_grammar.lark"), "r") as f:
      return lark_grammar(
        f.read().replace("%%FUNCTION_NAME%%", self._generate_function_names(tools)).replace("%%ENTRY_PRODUCTION%%",
                                                                                            "function_call_expression" if required else "TEXT | function_call_expression").strip())

  def _generate_function_names(self, tools) -> str:
    """Generate a grammar rule for function names based on available tools."""
    function_names = [tool.function.name for tool in tools]
    if not function_names:
      return '""'  # Empty string if no functions available
    return ' | '.join([f'"{name}"' for name in function_names])

  def parse_complete(self, text: str, parallel_tool_calling: bool) -> list[UnplacedToolCall]:
    tool_calls = []

    # Match patterns like [func_name(param1=value1, param2=value2)]
    for i, m in enumerate(re.finditer(r'\[([\w_\-]+)\((.*?)\)\]', text, re.DOTALL)):
      try:
        func_name = m.group(1)
        params_str = m.group(2).strip()

        # Parse parameters more robustly
        params = {}
        if params_str:
          # Use a state machine approach to parse parameters
          current_param = ""
          current_name = ""
          in_string = False
          in_value = False
          string_delimiter = None
          brace_level = 0
          bracket_level = 0

          for char in params_str + ",":  # Add trailing comma to process the last parameter
            if not in_value:
              if char == "=":
                current_name = current_param.strip()
                current_param = ""
                in_value = True
              else:
                current_param += char
            else:
              if in_string:
                current_param += char
                if char == string_delimiter and params_str[i - 1:i] != "\\":
                  in_string = False
              elif char == '"' or char == "'":
                current_param += char
                in_string = True
                string_delimiter = char
              elif char == "{":
                current_param += char
                brace_level += 1
              elif char == "}":
                current_param += char
                brace_level -= 1
              elif char == "[":
                current_param += char
                bracket_level += 1
              elif char == "]":
                current_param += char
                bracket_level -= 1
              elif char == "," and brace_level == 0 and bracket_level == 0:
                # End of parameter
                param_value = current_param.strip()

                # Try to parse as JSON
                try:
                  # Handle quoted strings
                  if (param_value.startswith('"') and param_value.endswith('"')) or \
                    (param_value.startswith("'") and param_value.endswith("'")):
                    param_value = param_value[1:-1]
                  # Try parsing as JSON for complex structures
                  elif param_value.startswith("{") or param_value.startswith("[") or \
                    param_value.lower() in ["true", "false", "null"] or \
                    re.match(r'^-?\d+(\.\d+)?$', param_value):
                    param_value = json.loads(param_value)
                except json.JSONDecodeError:
                  # Keep as string if JSON parsing fails
                  pass

                params[current_name] = param_value
                current_param = ""
                in_value = False
              else:
                current_param += char

        # Create the tool call object
        tool_calls.append(UnplacedToolCall(name=func_name, arguments=json.dumps(params)))
      except Exception as e:
        if DEBUG >= 2: print(f"Failed to parse Watt tool calls: {e}")

    return tool_calls
