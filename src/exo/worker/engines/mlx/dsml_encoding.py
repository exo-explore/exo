import copy
import json
import re
from typing import Any

from exo.shared.types.api import ToolCallItem

# We vendor the DSML encoding file here, taken from HuggingFace.

# ── Special tokens ────────────────────────────────────────────────
DSML_TOKEN = "\uff5cDSML\uff5c"  # ｜DSML｜ (fullwidth vertical bars)
BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
THINKING_START = "<think>"
THINKING_END = "</think>"

TOOL_CALLS_START = f"<{DSML_TOKEN}function_calls>"
TOOL_CALLS_END = f"</{DSML_TOKEN}function_calls>"

# ── Prompt templates ──────────────────────────────────────────────
_TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml}function_calls>" block like the following as part of your reply to the user:
<{dsml}function_calls>
<{dsml}invoke name="$FUNCTION_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$FUNCTION_NAME2">
...
</{dsml}invoke>
</{dsml}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml}function_calls>
...
</{dsml}function_calls>

<function_results>
...
</function_results>

{think_start}...thinking about results{think_end}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""


# ── JSON helpers ──────────────────────────────────────────────────
def _to_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


# ── Encoding: arguments → DSML ───────────────────────────────────
def _encode_arguments_to_dsml(arguments: str | dict[str, Any]) -> str:
    """Convert JSON arguments to DSML parameter tags."""
    if isinstance(arguments, str):
        args_dict: dict[str, Any] = json.loads(arguments)  # pyright: ignore[reportAny]
    else:
        args_dict = arguments

    parts: list[str] = []
    for key, value in args_dict.items():  # pyright: ignore[reportAny]
        is_str = isinstance(value, str)
        rendered_value: str = value if is_str else _to_json(value)  # pyright: ignore[reportAny]
        parts.append(
            f'<{DSML_TOKEN}parameter name="{key}" string="{"true" if is_str else "false"}">'
            f"{rendered_value}"
            f"</{DSML_TOKEN}parameter>"
        )
    return "\n".join(parts)


def _encode_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Encode a list of tool calls into DSML format."""
    invocations: list[str] = []
    for tc in tool_calls:
        func: dict[str, Any] = tc.get("function", tc)  # pyright: ignore[reportAny]
        name: str = func.get("name", "")  # pyright: ignore[reportAny]
        arguments: str | dict[str, Any] = func.get("arguments", "{}")  # pyright: ignore[reportAny]
        dsml_args = _encode_arguments_to_dsml(arguments)
        invocations.append(
            f'<{DSML_TOKEN}invoke name="{name}">\n{dsml_args}\n</{DSML_TOKEN}invoke>'
        )
    joined = "\n".join(invocations)
    return f"\n\n<{DSML_TOKEN}function_calls>\n{joined}\n</{DSML_TOKEN}function_calls>"


# ── Encoding: tools → system prompt section ───────────────────────
def _render_tools_section(tools: list[dict[str, Any]]) -> str:
    """Render tool definitions into the DSML system prompt section."""
    # Extract function defs from OpenAI format
    tool_schemas: list[str] = []
    for tool in tools:
        func: dict[str, Any] = tool.get("function", tool)  # pyright: ignore[reportAny]
        tool_schemas.append(_to_json(func))

    return _TOOLS_SYSTEM_TEMPLATE.format(
        dsml=DSML_TOKEN,
        think_start=THINKING_START,
        think_end=THINKING_END,
        tool_schemas="\n".join(tool_schemas),
    )


# ── Finding last user/developer message index ────────────────────
def _find_last_user_index(messages: list[dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ("user", "developer"):
            return idx
    return -1


# ── Drop thinking from old turns ─────────────────────────────────
def _drop_old_thinking(
    messages: list[dict[str, Any]], last_user_idx: int
) -> list[dict[str, Any]]:
    """Strip reasoning_content from assistant messages before the last user message."""
    result: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant" and idx < last_user_idx:
            msg_copy = copy.copy(msg)
            msg_copy.pop("reasoning_content", None)
            result.append(msg_copy)
        else:
            result.append(msg)
    return result


# ── Main encoding function ────────────────────────────────────────
def encode_dsml_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Encode messages into a DeepSeek V3.2 prompt string with DSML tool support.

    Args:
        messages: OpenAI-format messages (system, user, assistant, tool roles).
        tools: OpenAI-format tool definitions, or None.
        enable_thinking: Whether thinking mode is enabled.

    Returns:
        Complete prompt string ready for tokenization.
    """
    thinking_mode = "thinking" if enable_thinking else "chat"
    last_user_idx = _find_last_user_index(messages)

    if thinking_mode == "thinking":
        messages = _drop_old_thinking(messages, last_user_idx)

    prompt = BOS_TOKEN

    for idx, _msg in enumerate(messages):
        prompt += _render_message(idx, messages, thinking_mode, tools, last_user_idx)

    return prompt


def _render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    tools: list[dict[str, Any]] | None,
    last_user_idx: int,
) -> str:
    """Render a single message into the prompt format."""
    msg = messages[index]
    role = str(msg.get("role", ""))  # pyright: ignore[reportAny]
    content = str(msg.get("content") or "")
    tool_calls: list[dict[str, Any]] | None = msg.get("tool_calls")
    reasoning_content: str | None = msg.get("reasoning_content")

    result = ""

    if role == "system":
        result += content
        if tools:
            result += "\n\n" + _render_tools_section(tools)

    elif role in ("user", "developer"):
        user_content = content
        # Developer messages with tools get special formatting
        if role == "developer" and tools:
            user_content = (
                _render_tools_section(tools) + "\n\n# The user's message is: " + content
            )

        result += f"{USER_TOKEN}{user_content}{ASSISTANT_TOKEN}"

        if index == last_user_idx and thinking_mode == "thinking":
            result += THINKING_START
        else:
            result += THINKING_END

    elif role == "tool":
        # Find the preceding assistant message to count tool_call order
        prev_idx = index - 1
        while prev_idx >= 0 and messages[prev_idx].get("role") == "tool":
            prev_idx -= 1

        tool_call_order = index - prev_idx
        assistant_tool_calls: list[dict[str, Any]] | None = (
            messages[prev_idx].get("tool_calls") if prev_idx >= 0 else None
        )

        if tool_call_order == 1:
            result += "\n\n<function_results>"

        result += f"\n<result>{content}</result>"

        if assistant_tool_calls and tool_call_order == len(assistant_tool_calls):
            result += "\n</function_results>"
            if index >= last_user_idx and thinking_mode == "thinking":
                result += "\n\n" + THINKING_START
            else:
                result += "\n\n" + THINKING_END

    elif role == "assistant":
        thinking_part = ""
        tool_calls_part = ""

        if tool_calls:
            tool_calls_part = _encode_tool_calls(tool_calls)

        if thinking_mode == "thinking" and index > last_user_idx:
            thinking_part = (reasoning_content or "") + THINKING_END

        result += f"{thinking_part}{content}{tool_calls_part}{EOS_TOKEN}"

    return result


# ── Output parsing: DSML → ToolCallItem ──────────────────────────

_INVOKE_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\s+name=\"([^\"]+)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)

_PARAM_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}parameter>",
    re.DOTALL,
)


def parse_dsml_output(text: str) -> list[ToolCallItem] | None:
    """Parse DSML function_calls block from model output text.

    Args:
        text: The text containing the DSML function_calls block
              (including the start/end markers).

    Returns:
        List of ToolCallItem, or None if parsing fails.
    """
    tool_calls: list[ToolCallItem] = []

    for invoke_match in _INVOKE_PATTERN.finditer(text):
        func_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)

        args: dict[str, Any] = {}
        for param_match in _PARAM_PATTERN.finditer(invoke_body):
            param_name = param_match.group(1)
            is_string = param_match.group(2) == "true"
            param_value = param_match.group(3)

            if is_string:
                args[param_name] = param_value
            else:
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value

        tool_calls.append(
            ToolCallItem(
                name=func_name,
                arguments=json.dumps(args),
            )
        )

    return tool_calls if tool_calls else None
