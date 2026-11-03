"""Llama 3.1, 3.2 and 3.3 tool calls.

Given tools, the official chat templates open the system turn with ``Environment: ipython``
and ask for a JSON function call. An assistant turn that calls a tool renders as::

    {"name": "get_weather", "parameters": {"city": "Paris"}}<|eot_id|>

The models also write the call after ``<|python_tag|>``, the token that starts code for the
ipython environment, and end it with ``<|eom_id|>`` (end of message: the turn waits for the
tool's result). Llama 3.1's built-in tools are called that way too, in Python::

    <|python_tag|>brave_search.call(query="weather in Paris")<|eom_id|>

``<|eom_id|>`` is one of the models' EOS tokens (``spec.py``), so generation stops there.

Small models also write a sentence first and the JSON call on a line of its own.

This parser turns such a turn into tool calls, with any text around them as content. A
turn that isn't one stays content, without the ``<|python_tag|>`` token: an answer streams
as usual, and what the model writes after ``<|python_tag|>`` that isn't a call (e.g. code
for the code interpreter) comes back as text. It runs only on prompts in the ipython
environment, so a JSON answer to a request without tools stays text.
"""

import ast
import json
from collections.abc import Generator
from typing import Final, cast

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.bootstrap import logger

PYTHON_TAG: Final = "<|python_tag|>"
# The tokens that end a message or the text, or start the next one. The first three are
# EOS tokens, so they normally end generation before they reach the parser; for a
# checkpoint whose EOS ids don't list them (one named unlike Llama 3), the parser ends the
# message there itself.
END_OF_MESSAGE: Final = (
    "<|eom_id|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    # A new message's header.
    "<|start_header_id|>",
)

_SYSTEM_HEADER: Final = "<|start_header_id|>system<|end_header_id|>\n\n"
_IPYTHON: Final = "Environment: ipython"
# Between calls written one after another: `{...}; {...}` or one per line.
_CALL_SEPARATORS: Final = " \t\r\n;"

_decoder = json.JSONDecoder()


def in_ipython_environment(prompt: str) -> bool:
    """Whether the prompt's system turn sets up the ipython environment, as the Llama
    3.1-3.3 chat templates do when the request has tools."""
    start = prompt.find(_SYSTEM_HEADER)
    if start < 0:
        return False
    start += len(_SYSTEM_HEADER)
    end = prompt.find("<|eot_id|>", start)
    system = prompt[start:] if end < 0 else prompt[start:end]
    return any(line.strip() == _IPYTHON for line in system.splitlines())


def _json_call(value: object) -> ToolCallItem | None:
    """``{"name": ..., "parameters": {...}}`` (the templates' format), also with
    ``arguments`` for ``parameters``, as a JSON string, or inside ``"function"``."""
    if not isinstance(value, dict):
        return None
    call = cast(dict[str, object], value)
    function = call.get("function")
    if isinstance(function, dict):
        call = cast(dict[str, object], function)
    name = call.get("name")
    arguments = call.get("parameters", call.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = cast(object, json.loads(arguments))
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return ToolCallItem(name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def _json_calls(text: str) -> tuple[list[ToolCallItem], str]:
    """The JSON calls at the start of the text (one or more objects, or a list of them),
    and the text after them (all of it when there are none)."""
    calls: list[ToolCallItem] = []
    position = 0
    while True:
        while position < len(text) and text[position] in _CALL_SEPARATORS:
            position += 1
        if position == len(text) or text[position] not in "{[":
            break
        try:
            value, end = cast(tuple[object, int], _decoder.raw_decode(text, position))
        except json.JSONDecodeError:
            break
        items = cast(list[object], value) if isinstance(value, list) else [value]
        parsed = [_json_call(item) for item in items]
        if not items or any(call is None for call in parsed):
            break
        calls.extend(call for call in parsed if call is not None)
        position = end
    return calls, text[position:] if calls else text


def _python_call(text: str) -> ToolCallItem | None:
    """A built-in tool call, ``name.call(argument="value", ...)``, with literal arguments."""
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if (
        not isinstance(expression, ast.Call)
        or expression.args
        or not isinstance(expression.func, ast.Attribute)
        or expression.func.attr != "call"
        or not isinstance(expression.func.value, ast.Name)
    ):
        return None
    arguments: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = cast(object, ast.literal_eval(keyword.value))
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            return None
    try:
        encoded = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    return ToolCallItem(name=expression.func.value.id, arguments=encoded)


def parse_tool_calls(text: str) -> tuple[list[ToolCallItem], str]:
    """The tool calls a message starts with and the text after them; no calls (and all of
    the text) when it doesn't start with one."""
    calls, rest = _json_calls(text)
    if calls:
        return calls, rest
    call = _python_call(text)
    if call is not None:
        return [call], ""
    return [], text


def _end_of_message(text: str) -> int | None:
    positions = [
        index for marker in END_OF_MESSAGE if (index := text.find(marker)) >= 0
    ]
    return min(positions) if positions else None


def _ends_at_line_start(text: str, at_line_start: bool) -> bool:
    """Whether a line starts after the text (only spaces after its last newline), given
    whether one started before it."""
    for char in text:
        if char == "\n":
            at_line_start = True
        elif not char.isspace():
            at_line_start = False
    return at_line_start


def _call_start(text: str, at_line_start: bool) -> int | None:
    """Where a call starts in the text: at ``<|python_tag|>``, or at a ``{`` that opens a
    line (the models write a JSON call on its own line, also after a sentence)."""
    tag = text.find(PYTHON_TAG)
    for index, char in enumerate(text if tag < 0 else text[:tag]):
        if char == "{" and at_line_start:
            return index
        at_line_start = _ends_at_line_start(char, at_line_start)
    return tag if tag >= 0 else None


def parse_llama3(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Content is passed on as it comes, but for whitespace at its end, held back until
    more content follows (so none is left before a call). From where a call starts to the
    end of the message, the text is held back: tool calls if it parses as them, otherwise
    content."""
    in_call = False
    held: str = ""
    at_line_start = True
    gap: str = ""  # the whitespace held back when the call started
    call: str = ""
    ended = False
    # Whether any content was passed on, for the text after a call to follow with a gap.
    said = False

    for response in responses:
        if response is None:
            yield None
            continue
        if response.is_thinking:
            yield response
            continue

        text: str = ""
        if not ended:
            text = response.text
            end = _end_of_message(text)
            if end is not None:
                logger.warning(
                    "Llama 3 output went on past the end of its message; dropped "
                    "what followed"
                )
                text, ended = text[:end], True

        content: str = ""
        if in_call:
            call += text.replace(PYTHON_TAG, "")
        else:
            start = _call_start(text, at_line_start)
            before = text if start is None else text[:start]
            pending = held + before
            content = pending.rstrip()
            held = pending[len(content) :]
            at_line_start = _ends_at_line_start(before, at_line_start)
            if start is not None:
                in_call = True
                gap, held = held, ""
                call = text[start:].replace(PYTHON_TAG, "")

        if response.finish_reason is None:
            if content:
                said = True
                yield response.model_copy(update={"text": content})
            continue

        # The message's last response, with the finish reason, usage and stats.
        if in_call:
            calls, rest = parse_tool_calls(call)
            if calls:
                # Text the model wrote after its calls stays content.
                if rest.strip():
                    content += (gap if said or content else "") + rest
                if content:
                    yield response.model_copy(
                        update={
                            "text": content,
                            "finish_reason": None,
                            "usage": None,
                            "stats": None,
                        }
                    )
                yield ToolCallResponse(
                    tool_calls=calls, usage=response.usage, stats=response.stats
                )
                continue
            # Not a call we can read: give it back as text, as other OpenAI-compatible
            # servers do.
            logger.warning(f"Llama 3 tool call parsing failed for: {call!r}")
            content += gap + call
        yield response.model_copy(update={"text": content + held})
