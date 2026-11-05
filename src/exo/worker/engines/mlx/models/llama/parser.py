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
for the code interpreter) comes back as text. Only a call to one of the request's tools is
taken as a call, and a JSON object in a code fence never is, so an answer that shows a call
as an example stays text. It runs only on prompts in the ipython environment, so a JSON
answer to a request without tools stays text.
"""

import ast
import json
from collections.abc import Generator
from typing import Final, Literal, cast, final

from exo.api.types import ToolCallItem
from exo.shared.request_content import describe_text
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
# A line that starts with it (after its indentation) opens or closes a code fence.
_FENCE: Final = "```"

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


def _arguments_json(arguments: dict[str, object]) -> str | None:
    """The arguments as JSON; None for NaN or an infinity (also a number too big for a
    float), which JSON can't hold."""
    try:
        return json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None


def _json_call(value: object, tool_names: frozenset[str] | None) -> ToolCallItem | None:
    """``{"name": ..., "parameters": {...}}`` (the templates' format), also with
    ``arguments`` for ``parameters``, as a JSON string, or inside ``"function"``, calling
    one of the tools (any name when ``tool_names`` is None)."""
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
    if tool_names is not None and name not in tool_names:
        return None
    encoded = _arguments_json(cast(dict[str, object], arguments))
    if encoded is None:
        return None
    return ToolCallItem(name=name, arguments=encoded)


def _json_calls(
    text: str, tool_names: frozenset[str] | None
) -> tuple[list[ToolCallItem], str]:
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
        parsed = [_json_call(item, tool_names) for item in items]
        if not items or any(call is None for call in parsed):
            break
        calls.extend(call for call in parsed if call is not None)
        position = end
    return calls, text[position:] if calls else text


def _python_call(text: str, tool_names: frozenset[str] | None) -> ToolCallItem | None:
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
    name = expression.func.value.id
    if tool_names is not None and name not in tool_names:
        return None
    arguments: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = cast(object, ast.literal_eval(keyword.value))
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            return None
    encoded = _arguments_json(arguments)
    if encoded is None:
        return None
    return ToolCallItem(name=name, arguments=encoded)


def parse_tool_calls(
    text: str, tool_names: frozenset[str] | None = None
) -> tuple[list[ToolCallItem], str]:
    """The tool calls a message starts with and the text after them; no calls (and all of
    the text) when it doesn't start with one."""
    calls, rest = _json_calls(text, tool_names)
    if calls:
        return calls, rest
    call = _python_call(text, tool_names)
    if call is not None:
        return [call], ""
    return [], text


def _end_of_message(text: str) -> int | None:
    positions = [
        index for marker in END_OF_MESSAGE if (index := text.find(marker)) >= 0
    ]
    return min(positions) if positions else None


_JSON_WHITESPACE: Final = " \t\n\r"
_DIGITS: Final = "0123456789"
_HEX_DIGITS: Final = "0123456789abcdefABCDEF"
_ESCAPES: Final = '"\\/bfnrt'
_LITERALS: Final = {"t": "rue", "f": "alse", "n": "ull"}
# The states of a JSON number in which it can end.
_NUMBER_ENDS: Final = ("zero", "integer", "fraction", "exponent")

_Expected = Literal["value", "value_or_close", "key", "key_or_close", "colon", "next"]
_Outcome = Literal["open", "ended", "invalid"]


def _number_step(state: str, char: str) -> str | None:
    """The next state of a JSON number, ``-?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?``,
    after the character; None when the character can't come next."""
    if char in _DIGITS:
        if state in ("start", "minus"):
            return "zero" if char == "0" else "integer"
        if state == "integer":
            return "integer"
        if state in ("dot", "fraction"):
            return "fraction"
        if state in ("e", "sign", "exponent"):
            return "exponent"
        return None
    if char == "-" and state == "start":
        return "minus"
    if char == "." and state in ("zero", "integer"):
        return "dot"
    if char in "eE" and state in ("zero", "integer", "fraction"):
        return "e"
    if char in "+-" and state == "e":
        return "sign"
    return None


@final
class _JsonContainer:
    """Follows a JSON object or array as it is written, to tell as soon as the text can't
    be one, and where it ends. Strict JSON, as the parser reads a call: no NaN or Infinity,
    no control characters in strings."""

    def __init__(self) -> None:
        self._open: list[str] = []  # the containers not yet closed, "{" or "["
        self._expected: _Expected = "value"
        self._in_string = False
        self._string_is_key = False
        # In a string: -1 after a backslash, the hex digits left of a \u escape, or 0.
        self._escape = 0
        self._number: str | None = None  # its state, in a number
        self._literal = ""  # what is left of true, false or null, in one

    def feed(self, text: str) -> tuple[_Outcome, int]:
        """Reads the text: ``("open", len(text))`` when all of it can be the container so
        far, ``("ended", index)`` when it ends just before the index, and
        ``("invalid", index)`` when the character at the index can't come next."""
        for index, char in enumerate(text):
            outcome = self._step(char)
            if outcome == "ended":
                return "ended", index + 1
            if outcome == "invalid":
                return "invalid", index
        return "open", len(text)

    def _step(self, char: str) -> _Outcome:
        if self._in_string:
            return self._string_step(char)
        if self._literal:
            if char != self._literal[0]:
                return "invalid"
            self._literal = self._literal[1:]
            return "open" if self._literal else self._value_done()
        if self._number is not None:
            step = _number_step(self._number, char)
            if step is not None:
                self._number = step
                return "open"
            if self._number not in _NUMBER_ENDS:
                return "invalid"
            # The number ends before the character, inside a container.
            self._number = None
            self._value_done()
        if char in _JSON_WHITESPACE:
            return "open"
        match self._expected:
            case "value" | "value_or_close":
                if char == "]" and self._expected == "value_or_close":
                    return self._close("[")
                return self._value_start(char)
            case "key" | "key_or_close":
                if char == "}" and self._expected == "key_or_close":
                    return self._close("{")
                if char != '"':
                    return "invalid"
                self._in_string, self._string_is_key = True, True
                return "open"
            case "colon":
                if char != ":":
                    return "invalid"
                self._expected = "value"
                return "open"
            case "next":
                if char == ",":
                    self._expected = "key" if self._open[-1] == "{" else "value"
                    return "open"
                if char == "}":
                    return self._close("{")
                if char == "]":
                    return self._close("[")
                return "invalid"

    def _string_step(self, char: str) -> _Outcome:
        if self._escape == -1:
            if char == "u":
                self._escape = 4
            elif char in _ESCAPES:
                self._escape = 0
            else:
                return "invalid"
        elif self._escape > 0:
            if char not in _HEX_DIGITS:
                return "invalid"
            self._escape -= 1
        elif char == "\\":
            self._escape = -1
        elif char == '"':
            self._in_string = False
            if self._string_is_key:
                self._expected = "colon"
            else:
                return self._value_done()
        elif char < " ":
            return "invalid"
        return "open"

    def _value_start(self, char: str) -> _Outcome:
        if char in "{[":
            self._open.append(char)
            self._expected = "key_or_close" if char == "{" else "value_or_close"
            return "open"
        if not self._open:
            # Only an object or an array.
            return "invalid"
        if char == '"':
            self._in_string, self._string_is_key = True, False
            return "open"
        if char in _LITERALS:
            self._literal = _LITERALS[char]
            return "open"
        self._number = _number_step("start", char)
        return "invalid" if self._number is None else "open"

    def _close(self, opener: str) -> _Outcome:
        if self._open[-1] != opener:
            return "invalid"
        self._open.pop()
        return self._value_done()

    def _value_done(self) -> _Outcome:
        if not self._open:
            return "ended"
        self._expected = "next"
        return "open"


@final
class _Llama3Stream:
    """What the parser has read of the message so far.

    In ``text``, content is passed on as it comes, but for whitespace at its end, held
    back until more content follows (so none is left before a call). A ``{`` that opens a
    line outside a code fence may start a call (``maybe_call``): the text is held back from
    there while it can still be a JSON call to one of the tools, and given back as content
    as soon as it can't. Once it is one, or after ``<|python_tag|>`` (``call``), the text
    is held back to the end of the message: tool calls if it parses as them, otherwise
    content.
    """

    def __init__(self, tool_names: frozenset[str] | None) -> None:
        self._tool_names = tool_names
        self._mode: Literal["text", "maybe_call", "call"] = "text"
        self._held = ""
        # Only whitespace since the line started.
        self._at_line_start = True
        # The line's first characters after its indentation, up to the length of a fence.
        self._line_head = ""
        self._in_fence = False
        self._gap = ""  # the whitespace held back when the (possible) call started
        self._call = ""  # the text from where it started
        self._json = _JsonContainer()
        # Whether any content was passed on, for the text after a call to follow with a gap.
        self.said = False

    def feed(self, text: str) -> str:
        """Reads the text; returns the content to pass on now."""
        content: list[str] = []
        while text:
            if self._mode == "call":
                self._call += text.replace(PYTHON_TAG, "")
                break
            if self._mode == "maybe_call":
                tag = text.find(PYTHON_TAG)
                upto = len(text) if tag < 0 else tag
                outcome, index = self._json.feed(text[:upto])
                self._call, text = self._call + text[:index], text[index:]
                if outcome == "open" and tag < 0:
                    break
                if outcome == "ended" and self._is_call(self._call):
                    self._mode = "call"
                    continue
                # Not a call: invalid JSON, JSON that is no call, or text before a
                # <|python_tag|>. What follows is read again as text.
                content.append(self._give_back())
                continue
            start = self._call_start(text)
            content.append(self._content(text if start is None else text[:start]))
            if start is None:
                break
            self._gap, self._held, self._call = self._held, "", ""
            if text.startswith(PYTHON_TAG, start):
                self._mode = "call"
                text = text[start + len(PYTHON_TAG) :]
            else:
                self._mode, self._json = "maybe_call", _JsonContainer()
                text = text[start:]
        return "".join(content)

    def finish(self, content: str) -> tuple[str, list[ToolCallItem]]:
        """The rest of the content, after the content of the last piece of text, and the
        tool calls, at the end of the message."""
        if self._mode == "call":
            calls, rest = parse_tool_calls(self._call, self._tool_names)
            if calls:
                # Text the model wrote after its calls stays content.
                if not rest.strip():
                    return "", calls
                return (self._gap if self.said or content else "") + rest, calls
            # Not a call we can read: give it back as text, as other OpenAI-compatible
            # servers do.
            logger.warning(
                f"Llama 3 tool call parsing failed for: {describe_text(self._call)}"
            )
        if self._mode == "text":
            return self._held, []
        return self._gap + self._call, []

    def _is_call(self, text: str) -> bool:
        try:
            value = cast(object, _decoder.decode(text))
        except json.JSONDecodeError:
            return False
        return _json_call(value, self._tool_names) is not None

    def _give_back(self) -> str:
        text = self._gap + self._call
        for char in self._call:
            self._see(char)
        self._mode, self._gap, self._call = "text", "", ""
        return self._content(text)

    def _content(self, text: str) -> str:
        pending = self._held + text
        content = pending.rstrip()
        self._held = pending[len(content) :]
        return content

    def _call_start(self, text: str) -> int | None:
        """Where a call may start in the text: at ``<|python_tag|>``, or at a ``{`` that
        opens a line outside a code fence (the models write a JSON call on its own line,
        also after a sentence). Follows the lines up to there."""
        tag = text.find(PYTHON_TAG)
        for index, char in enumerate(text if tag < 0 else text[:tag]):
            if char == "{" and self._at_line_start and not self._in_fence:
                return index
            self._see(char)
        return tag if tag >= 0 else None

    def _see(self, char: str) -> None:
        if char == "\n":
            self._at_line_start, self._line_head = True, ""
            return
        if self._at_line_start and char.isspace():
            return
        self._at_line_start = False
        if len(self._line_head) < len(_FENCE):
            self._line_head += char
            if self._line_head == _FENCE:
                self._in_fence = not self._in_fence


def parse_llama3(
    responses: Generator[GenerationResponse | None],
    tool_names: frozenset[str] | None = None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """The message's content and tool calls (``_Llama3Stream``). ``tool_names``: the
    request's tools; a call to any other name stays text (any name is a call when None)."""
    stream = _Llama3Stream(tool_names)
    ended = False

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

        content = stream.feed(text)

        if response.finish_reason is None:
            if content:
                stream.said = True
                yield response.model_copy(update={"text": content})
            continue

        # The message's last response, with the finish reason, usage and stats.
        rest, calls = stream.finish(content)
        content += rest
        if not calls:
            yield response.model_copy(update={"text": content})
            continue
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
