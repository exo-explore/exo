import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from exo.api.types import ToolCallItem


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    _inner_parser: Callable[[str], list[ToolCallItem] | None]

    def parse(
        self, text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        parsed = self._inner_parser(text)
        if parsed is None:
            return None
        if tools is not None:
            parsed = _coerce_tool_calls_to_schema(parsed, tools)
        return parsed


def _json_type_matches(value: Any, expected_type: str) -> bool:  # pyright: ignore[reportAny]
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _coerce_tool_arg_with_schema(value: Any, schema: dict[str, Any]) -> Any:  # pyright: ignore[reportAny]
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        for candidate in schema_type:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(candidate, str):
                continue
            if candidate == "null" and value is None:
                return None
            candidate_schema = {**schema, "type": candidate}
            coerced = _coerce_tool_arg_with_schema(value, candidate_schema)  # pyright: ignore[reportAny]
            if _json_type_matches(coerced, candidate):
                return coerced  # pyright: ignore[reportAny]
        return value  # pyright: ignore[reportAny]

    if not isinstance(schema_type, str):
        return value  # pyright: ignore[reportAny]

    if schema_type == "object":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, dict):
            return value  # pyright: ignore[reportAny]
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return {
            key: (
                _coerce_tool_arg_with_schema(prop_value, prop_schema)  # pyright: ignore[reportUnknownArgumentType]
                if isinstance(prop_schema, dict)
                else prop_value
            )
            for key, prop_value in parsed.items()  # pyright: ignore[reportUnknownVariableType]
            for prop_schema in [properties.get(key)]  # type: ignore
        }

    if schema_type == "array":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, list):
            return value  # pyright: ignore[reportAny]
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return [_coerce_tool_arg_with_schema(item, item_schema) for item in parsed]  # type: ignore

    if schema_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value

    if schema_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                num = float(value.strip())
                if math.isfinite(num):
                    return num
            except ValueError:
                return value
        return value

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return value

    return value  # pyright: ignore[reportAny]


def _coerce_tool_calls_to_schema(
    tool_calls: list[ToolCallItem], tools: list[dict[str, Any]]
) -> list[ToolCallItem]:
    schema_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")  # type: ignore
        parameters = function.get("parameters")  # type: ignore
        if isinstance(name, str) and isinstance(parameters, dict):
            schema_by_name[name] = parameters

    if not schema_by_name:
        return tool_calls

    coerced_calls: list[ToolCallItem] = []
    for tool_call in tool_calls:
        schema = schema_by_name.get(tool_call.name)
        if schema is None:
            coerced_calls.append(tool_call)
            continue

        try:
            parsed_args = json.loads(tool_call.arguments)  # pyright: ignore[reportAny]
        except Exception:
            coerced_calls.append(tool_call)
            continue

        if not isinstance(parsed_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_args = _coerce_tool_arg_with_schema(parsed_args, schema)  # pyright: ignore[reportAny]
        if not isinstance(coerced_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_calls.append(
            tool_call.model_copy(update={"arguments": json.dumps(coerced_args)})
        )
    return coerced_calls


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
            raw_calls = parsed if isinstance(parsed, list) else [parsed]
            items: list[ToolCallItem] = []
            for call in raw_calls:
                if _has_residual_markup(call):
                    # The upstream parser hit malformed/partial markup and fell
                    # back to a lenient path that left tool-call protocol
                    # fragments (e.g. <arg_key> / <arg_value>) embedded in the
                    # call name or arguments. Dispatching it would forward
                    # corrupted arguments downstream, so drop it and let the
                    # model regenerate the call.
                    logger.warning(
                        f"Dropping tool call with residual markup (parser={tool_call_start!r})"
                    )
                    continue
                items.append(ToolCallItem.model_validate(_flatten(call)))
            return items or None

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        _inner_parser=parse_tool_calls,
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


# GLM-style argument delimiters that the GLM format parser consumes during a
# clean parse and so must never survive into a parsed call's name or arguments.
# When the parser meets malformed or partial markup it can fall back to a
# lenient path that leaves these fragments embedded in argument keys or values
# (observed: a key like "file<arg_key" and a value like
# "page</arg_key><arg_value>5"); detecting them lets the caller drop the
# corrupted call instead of forwarding garbage arguments downstream. Kept
# narrow to the arg_key/arg_value family so other formats' legitimate
# delimiters (e.g. qwen3_coder's <function=>/<parameter=>) are never flagged.
_MARKUP_FRAGMENTS: tuple[str, ...] = (
    "<arg_key",
    "</arg_key>",
    "<arg_value",
    "</arg_value>",
)


def _contains_markup(text: str) -> bool:
    return any(fragment in text for fragment in _MARKUP_FRAGMENTS)


def _value_has_markup(value: Any) -> bool:  # pyright: ignore[reportAny]
    if isinstance(value, dict):
        return any(
            (isinstance(k, str) and _contains_markup(k)) or _value_has_markup(v)  # pyright: ignore[reportAny]
            for k, v in value.items()  # pyright: ignore[reportAny]
        )
    if isinstance(value, list):
        return any(_value_has_markup(item) for item in value)  # pyright: ignore[reportAny]
    if isinstance(value, str):
        return _contains_markup(value)
    return False


def _has_residual_markup(parsed_call: dict[str, Any]) -> bool:
    if not isinstance(parsed_call, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return False
    name = parsed_call.get("name")  # pyright: ignore[reportAny]
    if isinstance(name, str) and _contains_markup(name):
        return True
    return _value_has_markup(parsed_call.get("arguments"))


def make_json_parser() -> ToolParser:
    return ToolParser(
        start_parsing="<tool_call>",
        end_parsing="</tool_call>",
        _inner_parser=_parse_json_calls,
    )


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return make_json_parser()
    return None
