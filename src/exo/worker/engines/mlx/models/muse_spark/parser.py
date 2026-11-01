"""Muse Spark output (the ATEM format): reasoning, answers and tool calls.

An assistant turn is a run of messages, each with a header naming its recipient::

    <|start|>assistant to=self<|message|>reasoning<|eom|>
    <|start|>assistant to=commentary<|message|>a preamble<|eom|>
    <|start|>assistant to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Paris</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

The generation prompt ends with ``<|start|>assistant``, so the model starts with the first
header's `` to=…``. ``self`` is reasoning; ``user`` (the answer) and ``commentary``
(user-visible updates) are content; any other recipient is a tool, and the message holds
one ``<atem:function_calls>`` block (parallel calls are consecutive messages). A turn ends
with ``<|eot|>``, an end-of-sequence token that never reaches the parser.

Parameter values are written as they are for strings and scalars and as JSON for lists and
objects, so their types come from the tool's schema: the parser keeps every value as its
text and exo coerces the arguments to the request's tool schemas. Each marker is a single
special token, so it arrives whole in a detokenized segment.
"""

import html
import json
import re
from collections.abc import Generator
from typing import Literal

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse

START = "<|start|>"
MESSAGE = "<|message|>"
END_OF_MESSAGE = "<|eom|>"
END_OF_TURN = "<|eot|>"
END_OF_TEXT = "<|end_of_text|>"
TOOL_CALLS_START = "<atem:function_calls>"
TOOL_CALLS_END = "</atem:function_calls>"

# Longer than any header the model writes (`assistant to=<tool name>`): text that runs on
# this long without a <|message|> isn't a header, so it is given back as content.
MAX_HEADER_LENGTH = 256

_MARKER = re.compile(
    "("
    + "|".join(
        re.escape(m) for m in (START, MESSAGE, END_OF_MESSAGE, END_OF_TURN, END_OF_TEXT)
    )
    + ")"
)
_RECIPIENT = re.compile(r"\bto=(\S+)")
_INVOKE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="([^"]*)"[^>]*>(.*?)</atem:invoke>', re.DOTALL
)
_PARAMETER = re.compile(
    r'<atem:parameter\b[^>]*?\bname="([^"]*)"[^>]*>(.*?)</atem:parameter>', re.DOTALL
)

Channel = Literal["reasoning", "content", "tool"]


def recipient_of(header: str) -> str:
    """The recipient a header names (``assistant to=user`` -> ``user``); a header without
    one is a message to the user, as the chat template renders it."""
    match = _RECIPIENT.search(header)
    return "user" if match is None else match.group(1)


def channel_of(recipient: str) -> Channel:
    if recipient == "self":
        return "reasoning"
    if recipient in ("user", "commentary", "assistant"):
        return "content"
    return "tool"


def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
    """The calls in one ``<atem:function_calls>`` block, with every argument as its text;
    None if the text isn't a well-formed block."""
    body = text.strip()
    if not body.startswith(TOOL_CALLS_START) or not body.endswith(TOOL_CALLS_END):
        return None
    body = body[len(TOOL_CALLS_START) : -len(TOOL_CALLS_END)]
    calls: list[ToolCallItem] = []
    position = 0
    for invoke in _INVOKE.finditer(body):
        if body[position : invoke.start()].strip():
            return None
        position = invoke.end()
        parameters = invoke.group(2)
        arguments: dict[str, str] = {}
        parameter_position = 0
        for parameter in _PARAMETER.finditer(parameters):
            if parameters[parameter_position : parameter.start()].strip():
                return None
            parameter_position = parameter.end()
            arguments[html.unescape(parameter.group(1))] = parameter.group(2)
        if parameters[parameter_position:].strip():
            return None
        calls.append(
            ToolCallItem(
                name=html.unescape(invoke.group(1)),
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )
    if body[position:].strip() or not calls:
        return None
    return calls


class _Turn:
    """Where the stream is in the turn: a header, or the body of a message to a channel."""

    def __init__(self) -> None:
        # None while reading a header (up to its <|message|>), else the message's channel.
        self.channel: Channel | None = None
        self.header = ""
        self.tool_text = ""
        self.tool_calls: list[ToolCallItem] = []

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """The (text, is_thinking) pieces of the reasoning and content in ``text``."""
        pieces: list[tuple[str, bool]] = []
        for part in _MARKER.split(text):
            if not part:
                continue
            if part == MESSAGE:
                if self.channel is None:
                    self.channel = channel_of(recipient_of(self.header))
                    self.header = ""
            elif part in (START, END_OF_MESSAGE, END_OF_TURN, END_OF_TEXT):
                pieces.extend(self.end_message())
                self.channel = None
                self.header = ""
            elif self.channel is None:
                self.header += part
                if len(self.header) > MAX_HEADER_LENGTH:
                    pieces.append((self.header, False))
                    self.channel = "content"
                    self.header = ""
            elif self.channel == "tool":
                self.tool_text += part
            else:
                pieces.append((part, self.channel == "reasoning"))
        return pieces

    def end_message(self) -> list[tuple[str, bool]]:
        """Ends a tool message: its calls, or its text when exo can't read them (as the
        generic tool parser does)."""
        if self.channel != "tool":
            return []
        text, self.tool_text = self.tool_text, ""
        parsed = parse_tool_calls(text)
        if parsed is None:
            return [(text, False)]
        self.tool_calls.extend(parsed)
        return []

    def end_turn(self) -> list[tuple[str, bool]]:
        """At the end of generation, whose end-of-turn token isn't in the text: the open
        tool message, or a header that never reached its <|message|>, as content."""
        if self.channel is None and self.header.strip():
            header, self.header = self.header, ""
            return [(header, False)]
        return self.end_message()


def parse_muse_spark(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    turn = _Turn()
    for response in responses:
        if response is None:
            yield None
            continue

        pieces = turn.feed(response.text)
        if response.finish_reason is None:
            for text, is_thinking in pieces:
                yield response.model_copy(
                    update={"text": text, "is_thinking": is_thinking}
                )
            continue

        pieces.extend(turn.end_turn())
        last_text, last_thinking = pieces.pop() if pieces else ("", False)
        if turn.tool_calls and last_text:
            pieces.append((last_text, last_thinking))
            last_text = ""
        for text, is_thinking in pieces:
            yield response.model_copy(
                update={
                    "text": text,
                    "is_thinking": is_thinking,
                    "finish_reason": None,
                    "stats": None,
                    "usage": None,
                }
            )
        if turn.tool_calls:
            yield ToolCallResponse(
                tool_calls=turn.tool_calls, usage=response.usage, stats=response.stats
            )
        else:
            yield response.model_copy(
                update={"text": last_text, "is_thinking": last_thinking}
            )
        return
