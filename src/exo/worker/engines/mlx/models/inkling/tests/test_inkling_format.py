"""Inkling's stop token, reasoning effort levels and output parsing.

The token streams are what the official chat template renders for an assistant turn
(after the ``<|message_model|>`` that ends the prompt): reasoning, answer and tool calls are
messages marked by special tokens.
"""

import json
from collections.abc import Generator

import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.inkling import reasoning
from exo.worker.engines.mlx.models.inkling.parser import (
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    END_SAMPLING,
    INVOKE_TOOL_JSON,
    MARKERS,
    MESSAGE_MODEL,
    parse_inkling,
)
from exo.worker.engines.mlx.models.spec_registry import eos_token_ids_for_model_id

# The names the official chat template's effort_map accepts.
_TEMPLATE_EFFORTS = {"none", "minimal", "low", "medium", "high", "max"}


@pytest.mark.parametrize(
    "model_id",
    [
        "thinkingmachines/Inkling-Small",
        "thinkingmachines/Inkling",
        "mlx-community/Inkling-Small-mlx-4bit",
    ],
)
def test_inkling_stops_at_the_end_of_its_turn(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) == [200006]


def _task(effort: ReasoningEffort | None) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/Inkling-Small-mlx-4bit"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=effort,
    )


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (None, None),
        ("none", "none"),
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "max"),
    ],
)
def test_every_reasoning_effort_reaches_the_template_as_a_level_it_knows(
    effort: ReasoningEffort | None, expected: str | None
) -> None:
    got = reasoning.template_effort(_task(effort))
    assert got == expected
    assert got is None or got in _TEMPLATE_EFFORTS


def test_the_reasoning_hook_is_only_for_inkling() -> None:
    assert reasoning.applies_to(_task(None))
    other = _task(None).model_copy(update={"model": ModelId("someone/Other-Model")})
    assert not reasoning.applies_to(other)


_Piece = tuple[int, str]
_WORD = 1000  # an ordinary token id


def _responses(
    pieces: list[_Piece], finish: _Piece = (END_SAMPLING, "")
) -> Generator[GenerationResponse | None]:
    for token, text in pieces:
        yield GenerationResponse(text=text, token=token, usage=None)
    token, text = finish
    yield GenerationResponse(text=text, token=token, finish_reason="stop", usage=None)


def _marker(token: int) -> _Piece:
    return (token, MARKERS[token])


def _words(text: str) -> list[_Piece]:
    return [(_WORD, part) for part in text.split("|")]


def _run(
    pieces: list[_Piece], finish: _Piece = (END_SAMPLING, "")
) -> list[GenerationResponse | ToolCallResponse]:
    return [r for r in parse_inkling(_responses(pieces, finish)) if r is not None]


def _text(out: list[GenerationResponse | ToolCallResponse], thinking: bool) -> str:
    return "".join(
        r.text
        for r in out
        if isinstance(r, GenerationResponse) and r.is_thinking == thinking
    )


def test_reasoning_then_answer() -> None:
    out = _run(
        [
            _marker(CONTENT_THINKING),
            *_words("17 x 23 |is 391."),
            _marker(END_MESSAGE),
            _marker(MESSAGE_MODEL),
            _marker(CONTENT_TEXT),
            *_words("It is |391."),
            _marker(END_MESSAGE),
        ]
    )
    assert _text(out, thinking=True) == "17 x 23 is 391."
    assert _text(out, thinking=False) == "It is 391."
    last = out[-1]
    assert isinstance(last, GenerationResponse) and last.finish_reason == "stop"


def test_an_answer_without_reasoning() -> None:
    out = _run([_marker(CONTENT_TEXT), *_words("Paris."), _marker(END_MESSAGE)])
    assert _text(out, thinking=False) == "Paris."
    assert _text(out, thinking=True) == ""


def _tool_call(name: str, args: dict[str, object]) -> list[_Piece]:
    body = json.dumps({"name": name, "args": args}, separators=(",", ":"))
    return [
        _marker(MESSAGE_MODEL),
        *_words(name),
        _marker(INVOKE_TOOL_JSON),
        *[(_WORD, body[i : i + 7]) for i in range(0, len(body), 7)],
        _marker(END_MESSAGE),
    ]


def test_tool_calls_come_back_together_with_typed_arguments() -> None:
    out = _run(
        [
            _marker(CONTENT_THINKING),
            *_words("I need the weather."),
            _marker(END_MESSAGE),
            *_tool_call("get_weather", {"city": "Paris", "days": 3}),
            *_tool_call("write_file", {"path": "package.json", "content": '{"a": 1}'}),
        ]
    )
    calls = [r for r in out if isinstance(r, ToolCallResponse)]
    assert len(calls) == 1 and out[-1] is calls[0]
    names = [c.name for c in calls[0].tool_calls]
    assert names == ["get_weather", "write_file"]
    assert json.loads(calls[0].tool_calls[0].arguments) == {"city": "Paris", "days": 3}
    # A string argument that looks like JSON stays a string.
    assert json.loads(calls[0].tool_calls[1].arguments)["content"] == '{"a": 1}'
    assert _text(out, thinking=True) == "I need the weather."


def test_a_tool_call_without_arguments() -> None:
    out = _run(
        _tool_call("list_files", {})[1:]
    )  # the prompt already opened the message
    calls = [r for r in out if isinstance(r, ToolCallResponse)]
    assert [(c.name, json.loads(c.arguments)) for c in calls[0].tool_calls] == [
        ("list_files", {})
    ]


def test_an_unreadable_tool_call_comes_back_as_text() -> None:
    out = _run(
        [
            *_words("lookup"),
            _marker(INVOKE_TOOL_JSON),
            *_words("{not json"),
            _marker(END_MESSAGE),
        ]
    )
    assert not any(isinstance(r, ToolCallResponse) for r in out)
    assert "{not json" in _text(out, thinking=False)


def test_a_tool_call_cut_off_by_the_length_limit_comes_back_as_text() -> None:
    out = _run(
        [*_words("lookup"), _marker(INVOKE_TOOL_JSON), *_words('{"name":"lookup"')],
        finish=(_WORD, ',"args"'),
    )
    assert not any(isinstance(r, ToolCallResponse) for r in out)
    assert _text(out, thinking=False).endswith('{"name":"lookup","args"')


def test_text_the_detokenizer_held_back_stays_with_its_message() -> None:
    # The bytes of a character split across tokens arrive with the next (marker) token.
    out = _run(
        [
            _marker(CONTENT_THINKING),
            (_WORD, "caf"),
            (END_MESSAGE, "é" + MARKERS[END_MESSAGE]),
            _marker(MESSAGE_MODEL),
            _marker(CONTENT_TEXT),
            (_WORD, "ok"),
            _marker(END_MESSAGE),
        ]
    )
    assert _text(out, thinking=True) == "café"
    assert _text(out, thinking=False) == "ok"


def test_no_marker_text_reaches_the_client() -> None:
    out = _run(
        [
            _marker(CONTENT_THINKING),
            *_words("a"),
            _marker(END_MESSAGE),
            _marker(MESSAGE_MODEL),
            _marker(CONTENT_TEXT),
            *_words("b"),
            _marker(END_MESSAGE),
        ],
        finish=_marker(END_SAMPLING),
    )
    text = "".join(r.text for r in out if isinstance(r, GenerationResponse))
    assert "<|" not in text


def test_text_without_a_content_marker_streams_as_text() -> None:
    # e.g. the continuation of an assistant prefill: the prompt already opened the text
    # message, so the model writes text straight away.
    out = _run(
        [(_WORD, "Soft"), (_WORD, " rain"), (_WORD, " falls."), _marker(END_MESSAGE)]
    )
    assert _text(out, thinking=False) == "Soft rain falls."
    assert not any(isinstance(r, ToolCallResponse) for r in out)
    # It is passed on as soon as it can't be a tool name, not at the end of the message.
    first = next(r for r in out if isinstance(r, GenerationResponse))
    assert first.text == "Soft"
