import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, cast

from exo.api.types import ToolCallItem


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    _inner_parser: Callable[
        [str, list[dict[str, Any]] | None], list[ToolCallItem] | None
    ]

    def parse(
        self, text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        parsed = self._inner_parser(text, tools)
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


def _as_json_text(value: Any) -> str:  # pyright: ignore[reportAny]
    """A value a parser decoded from JSON (or a Python literal), written back as JSON text:
    non-ASCII characters as they are, json's default separators. Whitespace, key order
    within the text, escapes and number spelling may differ from what the model wrote."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)  # pyright: ignore[reportAny]


def _coerce_tool_arg_with_schema(value: Any, schema: dict[str, Any]) -> Any:  # pyright: ignore[reportAny]
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        candidates = [c for c in schema_type if isinstance(c, str)]  # pyright: ignore[reportUnknownVariableType]
        for candidate in candidates:
            if candidate == "null" and value is None:
                return None
            if candidate == "string":
                # Only a value that already is a string here: turning another value into
                # text is the last resort, once no other allowed type takes it.
                if isinstance(value, str):
                    return value
                continue
            candidate_schema = {**schema, "type": candidate}
            coerced = _coerce_tool_arg_with_schema(value, candidate_schema)  # pyright: ignore[reportAny]
            if _json_type_matches(coerced, candidate):
                return coerced  # pyright: ignore[reportAny]
        if "string" in candidates and value is not None:
            return _as_json_text(value)
        return value  # pyright: ignore[reportAny]

    if not isinstance(schema_type, str):
        return value  # pyright: ignore[reportAny]

    if schema_type == "string":
        # Parsers decode argument values that look like JSON (an object, a number, true), so a
        # string argument whose text is JSON arrives decoded. JSON null stays null.
        if isinstance(value, str) or value is None:
            return value
        return _as_json_text(value)

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


def _accepts_tools(
    tool_parser: Callable[..., dict[str, Any] | list[dict[str, Any]]],
) -> bool:
    """Whether a parser takes the tools as a second argument, as mlx-lm's parsers do."""
    try:
        parameters = list(inspect.signature(tool_parser).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in parameters
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    return len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in positional)


def _string_arguments(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    """For each tool, the top-level arguments its schema declares as strings (or strings or
    null)."""
    by_tool: dict[str, set[str]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")  # type: ignore
        parameters = function.get("parameters")  # type: ignore
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")  # type: ignore
        if not isinstance(properties, dict):
            continue
        names: set[str] = set()
        for key, prop in properties.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(key, str) or not isinstance(prop, dict):
                continue
            prop_type = prop.get("type")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            types = prop_type if isinstance(prop_type, list) else [prop_type]  # pyright: ignore[reportUnknownVariableType]
            if "string" in types and all(t in ("string", "null") for t in types):  # pyright: ignore[reportUnknownVariableType]
                names.add(key)
        by_tool[name] = names
    return by_tool


def _with_string_arguments_as_written(
    calls: list[dict[str, Any]],
    tool_parser: Callable[..., dict[str, Any] | list[dict[str, Any]]],
    text: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the model's text for string arguments.

    Without the tools, parsers decode every argument value that reads as JSON, so a string
    argument holding a JSON document (a file's contents, say) comes back as an object. Given
    the tools, some parsers keep the arguments the schema declares as strings exactly as the
    model wrote them. Take those from a second parse with the tools, for the arguments the
    first parse decoded into something other than a string; everything else stays the first
    parse's."""
    if not _accepts_tools(tool_parser):
        return calls
    try:
        with_tools = tool_parser(text, tools)
    except Exception:
        return calls
    calls_with_tools = with_tools if isinstance(with_tools, list) else [with_tools]
    if len(calls_with_tools) != len(calls):
        return calls
    string_arguments = _string_arguments(tools)
    kept: list[dict[str, Any]] = []
    for call, call_with_tools in zip(calls, calls_with_tools, strict=True):
        name = call.get("name")
        arguments = call.get("arguments")
        arguments_with_tools = call_with_tools.get("arguments")
        if (
            not isinstance(name, str)
            or call_with_tools.get("name") != name
            or not isinstance(arguments, dict)
            or not isinstance(arguments_with_tools, dict)
        ):
            kept.append(call)
            continue
        strings = string_arguments.get(name, set())
        decoded = cast(dict[str, Any], arguments)
        as_written = cast(dict[str, Any], arguments_with_tools)
        merged: dict[str, Any] = {}
        for key, value in decoded.items():  # pyright: ignore[reportAny]
            written = as_written.get(key)
            if (
                key in strings
                and value is not None
                and not isinstance(value, str)
                and isinstance(written, str)
            ):
                merged[key] = written
            else:
                merged[key] = value
        kept.append({**call, "arguments": merged})
    return kept


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[..., dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(
        text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        try:
            text = text.removeprefix(tool_call_start)
            text = text.removesuffix(tool_call_end)
            parsed = tool_parser(text)
            calls = parsed if isinstance(parsed, list) else [parsed]
            if tools:
                calls = _with_string_arguments_as_written(
                    calls, tool_parser, text, tools
                )
            return [ToolCallItem.model_validate(_flatten(p)) for p in calls]

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        _inner_parser=parse_tool_calls,
    )


# TODO / example code:
def _parse_json_calls(
    text: str, _tools: list[dict[str, Any]] | None = None
) -> list[ToolCallItem] | None:
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
