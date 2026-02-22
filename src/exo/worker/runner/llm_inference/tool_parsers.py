import json
from dataclasses import dataclass
from typing import Any, Callable

from exo.shared.types.api import ToolCallItem


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    parse_tool_calls: Callable[[str], list[ToolCallItem] | None]


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
        try:
            text = text.removeprefix(tool_call_start)
            text = text.removesuffix(tool_call_end)
            parsed = tool_parser(text)
            if isinstance(parsed, list):
                return [ToolCallItem.model_validate(_flatten(p)) for p in parsed]
            else:
                return [ToolCallItem.model_validate(_flatten(parsed))]

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        parse_tool_calls=parse_tool_calls,
    )


# TODO / example code:
def _parse_json_calls(text: str) -> list[ToolCallItem] | None:
    try:
        text = text.removeprefix("<tool_call>")
        text = text.removesuffix("</tool_call>")
        top_level = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in json.loads(text).items()  # pyright: ignore[reportAny]
        }
        return [ToolCallItem.model_validate(top_level)]
    except Exception:
        return None


def _flatten(p: dict[str, Any]) -> dict[str, str]:
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
        for k, v in p.items()  # pyright: ignore[reportAny]
    }


json_tool_parser = ToolParser(
    start_parsing="<tool_call>",
    end_parsing="</tool_call>",
    parse_tool_calls=_parse_json_calls,
)


# ── Llama 3 instruct tool parser ──


# Tokens that Llama 3 instruct models use in their chat template.
# These should never appear as visible text in generation output.
LLAMA3_SPECIAL_TOKENS = frozenset({
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|python_tag|>",
    "<|finetune_right_pad_id|>",
})

# The token Llama 3 uses to signal the start of a tool/function call.
LLAMA3_TOOL_START = "<|python_tag|>"
# The token Llama 3 uses to signal end-of-message (used as EOS during tool calls).
LLAMA3_EOM = "<|eom_id|>"
# The token Llama 3 uses to signal end-of-turn.
LLAMA3_EOT = "<|eot_id|>"


def _flatten_llama3(p: dict[str, Any]) -> dict[str, str]:
    """Flatten a Llama 3 tool call dict into ToolCallItem format.

    Llama 3 uses {"name": ..., "parameters": {...}}.
    ToolCallItem expects {"name": ..., "arguments": "..."}.
    """
    result: dict[str, str] = {}
    if "name" in p:
        result["name"] = str(p["name"])  # pyright: ignore[reportAny]
    if "parameters" in p:
        v = p["parameters"]  # pyright: ignore[reportAny]
        result["arguments"] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
    elif "arguments" in p:
        v = p["arguments"]  # pyright: ignore[reportAny]
        result["arguments"] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
    return result


def _parse_llama3_tool_calls(text: str) -> list[ToolCallItem] | None:
    """Parse Llama 3 instruct tool calls.

    Llama 3 emits tool calls after <|python_tag|> as JSON:
      {"name": "function_name", "parameters": {"key": "value"}}

    Multiple calls may be newline-separated or wrapped in an array.
    """
    try:
        text = text.removeprefix(LLAMA3_TOOL_START)
        text = text.removesuffix(LLAMA3_EOM)
        text = text.strip()
        if not text:
            return None

        # Try as a JSON array first
        if text.startswith("["):
            calls = json.loads(text)  # pyright: ignore[reportAny]
            if isinstance(calls, list):
                return [
                    ToolCallItem.model_validate(_flatten_llama3(c))
                    for c in calls  # pyright: ignore[reportAny]
                ]

        # Try individual JSON objects (possibly newline-separated)
        results: list[ToolCallItem] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)  # pyright: ignore[reportAny]
            results.append(ToolCallItem.model_validate(_flatten_llama3(parsed)))

        return results if results else None
    except Exception:
        return None


llama3_tool_parser = ToolParser(
    start_parsing=LLAMA3_TOOL_START,
    end_parsing=LLAMA3_EOM,
    parse_tool_calls=_parse_llama3_tool_calls,
)


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return json_tool_parser
    return None
