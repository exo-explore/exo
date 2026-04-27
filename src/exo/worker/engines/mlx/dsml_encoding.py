import json
import re
from typing import Any

from mlx_lm.chat_templates import deepseek_v32

from exo.api.types import ToolCallItem

BOS_TOKEN: str = deepseek_v32.bos_token
EOS_TOKEN: str = deepseek_v32.eos_token
DSML_TOKEN: str = deepseek_v32.dsml_token
THINKING_START: str = deepseek_v32.thinking_start_token
THINKING_END: str = deepseek_v32.thinking_end_token
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
TOOL_CALLS_START = f"<{DSML_TOKEN}function_calls>"
TOOL_CALLS_END = f"</{DSML_TOKEN}function_calls>"
_ORPHAN_THINK_END = ASSISTANT_TOKEN + THINKING_END
_FIXED_THINK_BLOCK = ASSISTANT_TOKEN + THINKING_START + "\n" + THINKING_END
_FUNCTION_RESULTS_CLOSE = "</function_results>"
_ORPHAN_TOOL_RESULT_SUFFIX = _FUNCTION_RESULTS_CLOSE + "\n\n" + THINKING_END
_EMPTY_THINK_BLOCKS = (
    THINKING_START + "\n\n" + THINKING_END,
    THINKING_START + "\n" + THINKING_END,
    THINKING_START + THINKING_END,
)


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str = "thinking",
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    tools: Any = None,  # pyright: ignore[reportAny]
) -> str:
    # V3.2 (like V4) is `tool_conditional`: when tools are in play, prior-turn
    # reasoning_content must be retained so multi-step tool chains stay
    # coherent.
    effective_drop_thinking = drop_thinking
    if tools:
        effective_drop_thinking = False
    prompt: str = deepseek_v32.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=effective_drop_thinking,
        add_default_bos_token=add_default_bos_token,
        tools=tools,
    )
    prompt = prompt.replace(_ORPHAN_TOOL_RESULT_SUFFIX, _FUNCTION_RESULTS_CLOSE)
    prompt = prompt.replace(_ORPHAN_THINK_END, _FIXED_THINK_BLOCK)
    for empty in _EMPTY_THINK_BLOCKS:
        prompt = prompt.replace(empty, "")
    return prompt


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
