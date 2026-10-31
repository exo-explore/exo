import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from exo.api.types import ToolCallItem


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    _inner_parser: Callable[[str], list[ToolCallItem] | None]

    def parse(
        self, text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        parsed: list[ToolCallItem] | None = None
        if tools is not None and "<function=" in text:
            parsed = _parse_xml_with_schema(
                text.strip()
                .removeprefix(self.start_parsing)
                .removesuffix(self.end_parsing),
                tools,
            )
        if parsed is None:
            parsed = self._inner_parser(text)
        if parsed is None:
            return None
        if tools is not None:
            parsed = _coerce_tool_calls_to_schema(parsed, tools)
            if any(_missing_required_arguments(call, tools) for call in parsed):
                # Better to hand back the raw text than an empty call: an empty
                # call always fails, and once it is in the history the model
                # copies it and loops.
                return None
        return parsed


def _missing_required_arguments(
    tool_call: ToolCallItem, tools: list[dict[str, Any]]
) -> bool:
    try:
        arguments = json.loads(tool_call.arguments)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return False
    if arguments:
        return False
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == tool_call.name:  # pyright: ignore[reportUnknownMemberType]
            parameters = function.get("parameters")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return isinstance(parameters, dict) and bool(parameters.get("required"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return False


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


_xml_function_regex = re.compile(
    r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL
)
_xml_parameter_regex = re.compile(
    r'<parameter(?:=|\s+name=)"?([^">\n]+)"?>\n?(.*?)\n?(?=</parameter>|<parameter[=\s]|$)',
    re.DOTALL,
)


def _parse_xml_lenient(text: str) -> dict[str, Any] | None:
    """Recover Qwen3-Coder style calls the strict mlx_lm parser turns into `{}`.

    Qwen sometimes drops `</parameter>` / `</function>`, writes
    `<parameter name="x">`, or writes a JSON object inside `<function=name>`
    (often with raw newlines inside its strings, which strict JSON rejects).
    The strict parser returns empty arguments for all of those, and the agent
    then loops resending a call it thinks it made.
    """
    function_match = _xml_function_regex.search(text)
    if function_match is None:
        return None
    name = function_match.group(1).strip()
    body = function_match.group(2).strip()
    if body.startswith("{"):
        try:
            arguments = json.loads(body, strict=False)  # pyright: ignore[reportAny]
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        return {"name": name, "arguments": arguments}
    arguments = {
        parameter_match.group(1).strip(): parameter_match.group(2)
        for parameter_match in _xml_parameter_regex.finditer(body)
    }
    if not arguments:
        return None
    return {"name": name, "arguments": arguments}


_xml_function_open_regex = re.compile(r"<function=([^>\n]+)>")
_xml_parameter_close_regex = re.compile(r"\n?</param[a-z]*>\s*$")


def _parse_xml_with_schema(
    text: str, tools: list[dict[str, Any]]
) -> list[ToolCallItem] | None:
    """Parse a Qwen3-Coder style call by splitting on the tool's own parameter names.

    The strict mlx_lm parser pairs each `<parameter=` with the first
    `</parameter>` after it, so a dropped close tag loses that parameter or folds
    the next one into it, and a value that itself contains `</parameter>` (an
    edit to a prompt or XML file) is cut short and can apply a wrong edit. Here a
    value runs until the next opening tag for one of the tool's parameters, and
    only the close tag at its very end is removed.
    """
    function_matches = list(_xml_function_open_regex.finditer(text))
    if len(function_matches) != 1:
        return None
    name = function_matches[0].group(1).strip()
    body = text[function_matches[0].end() :]
    function_close = body.rfind("</function>")
    if function_close != -1:
        body = body[:function_close]
    if body.strip().startswith("{"):
        return None

    parameter_names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:  # pyright: ignore[reportUnknownMemberType]
            parameters = function.get("parameters")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(parameters, dict):
                properties = parameters.get("properties")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if isinstance(properties, dict):
                    parameter_names = [
                        key
                        for key in properties  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(key, str)
                    ]
    if not parameter_names:
        return None

    names_pattern = "|".join(
        re.escape(parameter_name)
        for parameter_name in sorted(parameter_names, key=len, reverse=True)
    )
    opening_tags = list(
        re.finditer(rf'<parameter(?:=|\s+name=)"?({names_pattern})"?>', body)
    )
    if not opening_tags:
        return None

    arguments: dict[str, str] = {}
    for index, opening_tag in enumerate(opening_tags):
        value_end = (
            opening_tags[index + 1].start()
            if index + 1 < len(opening_tags)
            else len(body)
        )
        value = body[opening_tag.end() : value_end]
        closed_value = _xml_parameter_close_regex.sub("", value, count=1)
        if closed_value == value:
            # No close tag: the newline before the next tag is not part of the value
            closed_value = value.rstrip(" \t").removesuffix("\n")
        arguments[opening_tag.group(1)] = closed_value.removeprefix("\n")
    return [
        ToolCallItem.model_validate({"name": name, "arguments": json.dumps(arguments)})
    ]


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
        text = text.removeprefix(tool_call_start)
        text = text.removesuffix(tool_call_end)
        try:
            parsed = tool_parser(text)
        except Exception:
            parsed = None
        if "<function=" in text and (
            parsed is None or (isinstance(parsed, dict) and not parsed.get("arguments"))
        ):
            parsed = _parse_xml_lenient(text) or parsed
        if parsed is None:
            return None
        try:
            if isinstance(parsed, list):
                return [ToolCallItem.model_validate(_flatten(p)) for p in parsed]
            else:
                return [ToolCallItem.model_validate(_flatten(parsed))]

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
