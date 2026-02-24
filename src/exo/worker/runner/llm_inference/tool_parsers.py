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


def _try_parse_json(v: str) -> str | dict[str, Any] | list[Any]:
    stripped = v.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    ):
        try:
            parsed: dict[str, Any] | list[Any] = json.loads(stripped)  # pyright: ignore[reportAny]
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return v


def _flatten(p: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in p.items():  # pyright: ignore[reportAny]
        if isinstance(v, dict):
            resolved: dict[str, Any] = {
                str(ik): _try_parse_json(str(iv)) if isinstance(iv, str) else iv  # pyright: ignore[reportUnknownArgumentType]
                for ik, iv in v.items()  # pyright: ignore[reportUnknownVariableType]
            }
            result[k] = json.dumps(resolved)
        elif isinstance(v, list):
            result[k] = json.dumps(v)
        else:
            result[k] = str(v)  # pyright: ignore[reportAny]
    return result


json_tool_parser = ToolParser(
    start_parsing="<tool_call>",
    end_parsing="</tool_call>",
    parse_tool_calls=_parse_json_calls,
)


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return json_tool_parser
    return None
