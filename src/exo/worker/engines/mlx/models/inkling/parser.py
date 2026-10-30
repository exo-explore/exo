"""Inkling's output: a sequence of messages, each opened by ``<|message_model|>`` and closed
by ``<|end_message|>``, after the ``<|message_model|>`` that ends the prompt. The token
after the opener says what the message holds:

- ``<|content_thinking|>`` reasoning;
- ``<|content_text|>`` the answer;
- a tool name, then ``<|content_invoke_tool_json|>`` and
  ``{"name": ..., "args": {...}}``: a tool call.

``<|content_model_end_sampling|>`` (the EOS token) ends the turn. This is the format the
chat template renders assistant turns in; e.g.
``<|content_thinking|>..<|end_message|><|message_model|><|content_text|>..<|end_message|>``.
The markers are special tokens, so they are matched by id; the text that comes with a
marker's token (bytes the detokenizer held back) belongs to what precedes the marker.
"""

import json
import re
from collections.abc import Generator
from typing import Final, Literal, cast

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.bootstrap import logger

MESSAGE_MODEL: Final = 200001
CONTENT_TEXT: Final = 200004
END_SAMPLING: Final = 200006
CONTENT_THINKING: Final = 200008
END_MESSAGE: Final = 200010
INVOKE_TOOL_JSON: Final = 200049
INVOKE_TOOL_TEXT: Final = 200057

MARKERS: Final[dict[int, str]] = {
    MESSAGE_MODEL: "<|message_model|>",
    CONTENT_TEXT: "<|content_text|>",
    END_SAMPLING: "<|content_model_end_sampling|>",
    CONTENT_THINKING: "<|content_thinking|>",
    END_MESSAGE: "<|end_message|>",
    INVOKE_TOOL_JSON: "<|content_invoke_tool_json|>",
    INVOKE_TOOL_TEXT: "<|content_invoke_tool_text|>",
}

# A message header longer than this, or with characters no tool name has (OpenAI function
# names are [A-Za-z0-9_-], up to 64), is not a tool name: it is text (e.g. the continuation
# of an assistant prefill, which the prompt already opened as a text message).
_MAX_HEADER_TOKENS: Final = 64
_TOOL_NAME: Final = re.compile(r"[A-Za-z0-9_.:-]*")

_State = Literal["header", "thinking", "text", "tool", "between"]


def parse_tool_call(name: str, body: str) -> ToolCallItem | None:
    """``{"name": ..., "args": {...}}`` (the template's rendering) as a tool call."""
    try:
        parsed = cast(object, json.loads(body))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    call = cast(dict[str, object], parsed)
    called = call.get("name", name)
    arguments = call.get("args", call.get("arguments", {}))
    if not isinstance(called, str) or not isinstance(arguments, dict):
        return None
    return ToolCallItem(
        name=called, arguments=json.dumps(arguments, ensure_ascii=False)
    )


def parse_inkling(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    state: _State = "header"
    header: list[GenerationResponse] = []
    tool_name = ""
    tool_body: list[str] = []
    tool_calls: list[ToolCallItem] = []

    def content(
        response: GenerationResponse, text: str, in_state: _State
    ) -> GenerationResponse | None:
        if not text:
            return None
        return response.model_copy(
            update={"text": text, "is_thinking": in_state == "thinking"}
        )

    def flush_header() -> Generator[GenerationResponse]:
        for buffered in header:
            yield buffered.model_copy(update={"is_thinking": False})
        header.clear()

    for response in responses:
        if response is None:
            yield None
            continue

        token = response.token
        marker = MARKERS.get(token)
        text = (
            response.text.replace(marker, "") if marker is not None else response.text
        )

        if response.finish_reason is not None:
            if state == "header":
                yield from flush_header()
            elif state == "tool":
                # Cut off inside a tool call: give back what it wrote.
                text = MARKERS[INVOKE_TOOL_JSON].join(
                    [tool_name, "".join(tool_body) + text]
                )
                state = "text"
            if tool_calls:
                if text and state in ("text", "thinking"):
                    leftover = content(response, text, state)
                    if leftover is not None:
                        yield leftover.model_copy(
                            update={"finish_reason": None, "stats": None, "usage": None}
                        )
                yield ToolCallResponse(
                    tool_calls=list(tool_calls),
                    usage=response.usage,
                    stats=response.stats,
                )
                tool_calls.clear()
                continue
            yield response.model_copy(
                update={"text": text, "is_thinking": state == "thinking"}
            )
            continue

        match state:
            case "header":
                if token == CONTENT_THINKING:
                    state = "thinking"
                elif token == CONTENT_TEXT:
                    state = "text"
                elif token in (INVOKE_TOOL_JSON, INVOKE_TOOL_TEXT):
                    tool_name = "".join(r.text for r in header).strip()
                    header.clear()
                    tool_body = [text]
                    state = "tool"
                elif token == END_MESSAGE:
                    yield from flush_header()
                    state = "between"
                elif marker is not None:
                    # Another marker where a header should be: keep going as text.
                    yield from flush_header()
                else:
                    header.append(response)
                    name = "".join(r.text for r in header).strip()
                    if len(header) > _MAX_HEADER_TOKENS or not _TOOL_NAME.fullmatch(
                        name
                    ):
                        yield from flush_header()
                        state = "text"
            case "thinking" | "text":
                if marker is not None:
                    leftover = content(response, text, state)
                    if leftover is not None:
                        yield leftover
                    if token == END_MESSAGE:
                        state = "between"
                    elif token == MESSAGE_MODEL:
                        state = "header"
                else:
                    yield response.model_copy(
                        update={"is_thinking": state == "thinking"}
                    )
            case "tool":
                if token == END_MESSAGE:
                    tool_body.append(text)
                    body = "".join(tool_body).strip()
                    parsed = parse_tool_call(tool_name, body)
                    if parsed is None:
                        # Not a tool call we can read: give it back as text, as other
                        # OpenAI-compatible servers do.
                        logger.warning(f"Inkling tool call parsing failed: {body!r}")
                        yield response.model_copy(
                            update={
                                "text": MARKERS[INVOKE_TOOL_JSON].join(
                                    [tool_name, body]
                                ),
                                "is_thinking": False,
                            }
                        )
                    else:
                        tool_calls.append(parsed)
                    tool_body = []
                    state = "between"
                else:
                    tool_body.append(text)
            case "between":
                if token == MESSAGE_MODEL:
                    state = "header"
                elif marker is None:
                    # Text outside any message: pass it on.
                    yield response.model_copy(update={"is_thinking": False})
