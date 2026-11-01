"""MiniMax M3 / M3.1 output: reasoning and tool calls.

The official chat template renders an assistant turn as
``<mm:think>reasoning</mm:think>answer`` or, for a turn without reasoning, as
``</mm:think>answer``: a bare closing tag. The model writes its turns the same way.

Tool calls are XML in a namespace of their own: every tag starts with the
``]<]minimax[>[`` token, so text inside a value (code, markup) can't be mistaken for a tag::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="get_weather">]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Objects nest as child elements and arrays as ``<item>`` children. Scalars are written as
text (booleans as ``true``/``false``), so their JSON types come from the tool's schema: exo
coerces the arguments to the request's tool schemas after parsing.
"""

import json
import re
from collections.abc import Generator
from dataclasses import dataclass, field

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_dsml_stream

THINK_START = "<mm:think>"
THINK_END = "</mm:think>"
NAMESPACE = "]<]minimax[>["
TOOL_CALLS_START = f"{NAMESPACE}<tool_call>"
TOOL_CALLS_END = f"{NAMESPACE}</tool_call>"

_THINK_TAG = re.compile(f"({re.escape(THINK_START)}|{re.escape(THINK_END)})")
_INVOKE = re.compile(r'<invoke name="([^"]*)">')
_OPEN = re.compile(r"<([^/<>\s][^<>]*)>")
_CLOSE = re.compile(r"</([^<>]+)>")

JsonValue = str | list["JsonValue"] | dict[str, "JsonValue"]


def parse_thinking(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | None]:
    """Marks reasoning with is_thinking and drops the tags. ``<mm:think>`` opens reasoning
    and ``</mm:think>`` closes it, including the bare one a turn without reasoning starts
    with. Each tag is a single token, so it arrives whole (at most next to a space the
    detokenizer held back)."""
    is_thinking = False
    for response in responses:
        if response is None:
            yield None
            continue
        if response.finish_reason is not None:
            text = _THINK_TAG.sub("", response.text)
            yield response.model_copy(update={"text": text, "is_thinking": False})
            continue
        parts = _THINK_TAG.split(response.text)
        if len(parts) == 1:
            yield response.model_copy(update={"is_thinking": is_thinking})
            continue
        for part in parts:
            if part == THINK_START:
                is_thinking = True
            elif part == THINK_END:
                is_thinking = False
            elif part:
                yield response.model_copy(
                    update={"text": part, "is_thinking": is_thinking}
                )


@dataclass
class _Element:
    name: str
    text: str = ""
    children: list["_Element"] = field(default_factory=list)

    def value(self) -> JsonValue:
        if not self.children:
            return self.text
        if all(child.name == "item" for child in self.children):
            return [child.value() for child in self.children]
        return {child.name: child.value() for child in self.children}


def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
    """The calls in one ``<tool_call>`` block, with string leaves (see the module
    docstring); None if the text isn't a well-formed block."""
    segments = text.split(NAMESPACE)
    if segments[0].strip():
        return None
    calls: list[ToolCallItem] = []
    invoke_name: str | None = None
    arguments: dict[str, JsonValue] = {}
    stack: list[_Element] = []
    opened_block = False
    closed_block = False
    for segment in segments[1:]:
        if closed_block:
            return None
        if segment.startswith("<tool_call>") and not opened_block:
            opened_block = True
            continue
        if not opened_block:
            return None
        if segment.startswith("</tool_call>"):
            if invoke_name is not None:
                return None
            closed_block = True
            continue
        if (invoke := _INVOKE.match(segment)) is not None:
            if invoke_name is not None:
                return None
            invoke_name = invoke.group(1)
            arguments = {}
            continue
        if segment.startswith("</invoke>"):
            if invoke_name is None or stack:
                return None
            calls.append(
                ToolCallItem(
                    name=invoke_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )
            invoke_name = None
            continue
        if invoke_name is None:
            return None
        if (close := _CLOSE.match(segment)) is not None:
            if not stack or stack[-1].name != close.group(1):
                return None
            element = stack.pop()
            if stack:
                stack[-1].children.append(element)
            else:
                arguments[element.name] = element.value()
            continue
        if (open_tag := _OPEN.match(segment)) is not None:
            stack.append(
                _Element(name=open_tag.group(1), text=segment[open_tag.end() :])
            )
            continue
        return None
    if not closed_block or not calls:
        return None
    return calls


def parse_minimax_m3(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    return parse_dsml_stream(
        parse_thinking(responses), TOOL_CALLS_START, TOOL_CALLS_END, parse_tool_calls
    )
