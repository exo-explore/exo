"""MiniMax M3's output format (reasoning, tool calls) and reasoning effort.

The tool-call and reasoning strings are what the official MiniMax-M3.1-Flash chat template
renders for an assistant turn.
"""

import json
import re
from collections.abc import Generator, Iterator
from typing import Any, cast, final

import jinja2
import pytest
from mlx_lm.models.minimax_m3_vl import Model as MiniMaxM3ModelClass

from exo.shared.types.chunks import GenerationChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.models.minimax.parser import parse_tool_calls
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.llm_inference.model_output_parsers import apply_all_parsers

_NS = "]<]minimax[>["
# apply_all_parsers finds the output parser by the loaded model's class.
MiniMaxM3Model = cast(type[Model], MiniMaxM3ModelClass)

# The official template's rendering of two calls (a get_weather and a write_file call
# with code, a nested object, a list, an empty list and a float).
_TOOL_CALLS = (
    f"{_NS}<tool_call>\n"
    f'{_NS}<invoke name="get_weather">{_NS}<city>Paris{_NS}</city>{_NS}<days>3{_NS}</days>'
    f"{_NS}<metric>true{_NS}</metric>{_NS}</invoke>\n"
    f'{_NS}<invoke name="write_file">{_NS}<path>a/b.py{_NS}</path>'
    f'{_NS}<content>if x < 3:\n    print("<b>")\n{_NS}</content>'
    f"{_NS}<meta>{_NS}<tags>{_NS}<item>x{_NS}</item>{_NS}<item>y{_NS}</item>{_NS}</tags>"
    f"{_NS}<mode>420{_NS}</mode>{_NS}<nested>{_NS}<k>v{_NS}</k>{_NS}</nested>{_NS}</meta>"
    f"{_NS}<empty>{_NS}</empty>{_NS}<ratio>0.5{_NS}</ratio>{_NS}</invoke>\n"
    f"{_NS}</tool_call>"
)

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "meta": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "mode": {"type": "integer"},
                            "nested": {"type": "object"},
                        },
                    },
                    "empty": {"type": "array"},
                    "ratio": {"type": "number"},
                },
            },
        },
    },
]

# Special tokens are whole tokens; split the rest into small pieces, as a tokenizer would.
_SPECIAL = re.compile(
    r"(\]<\]minimax\[>\[|<mm:think>|</mm:think>|<tool_call>|</tool_call>)"
)


def _pieces(text: str) -> list[str]:
    pieces: list[str] = []
    for part in _SPECIAL.split(text):
        if _SPECIAL.fullmatch(part):
            pieces.append(part)
        else:
            pieces.extend(part[i : i + 3] for i in range(0, len(part), 3))
    return pieces


def _responses(text: str) -> Generator[GenerationResponse | None]:
    pieces = _pieces(text)
    for index, piece in enumerate(pieces):
        yield GenerationResponse(text=piece, token=index, usage=None)
    yield GenerationResponse(
        text="", token=len(pieces), usage=None, finish_reason="stop"
    )


@final
class _Tokenizer:
    # mlx-lm finds <think> in MiniMax's vocabulary, which the model doesn't use.
    has_thinking = True
    think_start = "<think>"
    think_end = "</think>"


def _chunks(
    text: str, tools: list[dict[str, Any]] | None = None
) -> list[GenerationChunk]:
    chunks: Iterator[GenerationChunk | None] = apply_all_parsers(
        _responses(text),
        "]~b]ai\n",
        None,
        _Tokenizer(),  # pyright: ignore[reportArgumentType]
        MiniMaxM3Model,
        ModelId("MiniMaxAI/MiniMax-M3.1-Flash"),
        tools,
    )
    return [chunk for chunk in chunks if chunk is not None]


def _text(chunks: list[GenerationChunk], thinking: bool) -> str:
    return "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and chunk.is_thinking == thinking
    )


def test_reasoning_goes_to_its_own_channel() -> None:
    chunks = _chunks("<mm:think>Two plus two.</mm:think>It is 4.")
    assert _text(chunks, thinking=True) == "Two plus two."
    assert _text(chunks, thinking=False) == "It is 4."


def test_a_turn_without_reasoning_drops_its_bare_closing_tag() -> None:
    chunks = _chunks("</mm:think>Hello there.")
    assert _text(chunks, thinking=True) == ""
    assert _text(chunks, thinking=False) == "Hello there."


def test_a_tag_next_to_a_held_back_space_is_still_dropped() -> None:
    responses = [
        GenerationResponse(text="<mm:think>", token=0, usage=None),
        GenerationResponse(text="Hm.", token=1, usage=None),
        GenerationResponse(text=" </mm:think>", token=2, usage=None),
        GenerationResponse(text="Yes.", token=3, usage=None),
        GenerationResponse(text="", token=4, usage=None, finish_reason="stop"),
    ]
    chunks = [
        chunk
        for chunk in apply_all_parsers(
            (response for response in responses),
            "",
            None,
            _Tokenizer(),  # pyright: ignore[reportArgumentType]
            MiniMaxM3Model,
            ModelId("MiniMaxAI/MiniMax-M3.1-Flash"),
            None,
        )
        if chunk is not None
    ]
    assert _text(chunks, thinking=True) == "Hm. "
    assert _text(chunks, thinking=False) == "Yes."


def test_tool_calls_parse_back_to_typed_arguments() -> None:
    chunks = _chunks(
        f"<mm:think>Need the weather.</mm:think>Let me check.{_TOOL_CALLS}", _TOOLS
    )
    assert _text(chunks, thinking=True) == "Need the weather."
    assert _text(chunks, thinking=False) == "Let me check."
    calls = [
        call
        for chunk in chunks
        if isinstance(chunk, ToolCallChunk)
        for call in chunk.tool_calls
    ]
    assert [call.name for call in calls] == ["get_weather", "write_file"]
    assert json.loads(calls[0].arguments) == {
        "city": "Paris",
        "days": 3,
        "metric": True,
    }
    assert json.loads(calls[1].arguments) == {
        "path": "a/b.py",
        "content": 'if x < 3:\n    print("<b>")\n',
        "meta": {"tags": ["x", "y"], "mode": 420, "nested": {"k": "v"}},
        "empty": [],
        "ratio": 0.5,
    }


def test_without_a_schema_values_stay_text() -> None:
    calls = parse_tool_calls(_TOOL_CALLS)
    assert calls is not None
    assert json.loads(calls[0].arguments) == {
        "city": "Paris",
        "days": "3",
        "metric": "true",
    }


def test_a_string_parameter_that_looks_like_json_stays_a_string() -> None:
    text = (
        f'{_NS}<tool_call>\n{_NS}<invoke name="write_file">{_NS}<path>package.json'
        f'{_NS}</path>{_NS}<content>{{"a": 1}}{_NS}</content>{_NS}</invoke>\n'
        f"{_NS}</tool_call>"
    )
    chunks = _chunks(text, _TOOLS)
    calls = [
        c
        for chunk in chunks
        if isinstance(chunk, ToolCallChunk)
        for c in chunk.tool_calls
    ]
    assert json.loads(calls[0].arguments) == {
        "path": "package.json",
        "content": '{"a": 1}',
    }


@pytest.mark.parametrize(
    "text",
    [
        f'{_NS}<tool_call>\n{_NS}<invoke name="f">{_NS}<a>1{_NS}</b>{_NS}</invoke>\n{_NS}</tool_call>',
        f'{_NS}<tool_call>\n{_NS}<invoke name="f">{_NS}<a>1{_NS}</a>\n{_NS}</tool_call>',
        f"{_NS}<tool_call>\n{_NS}</tool_call>",
    ],
)
def test_a_malformed_block_comes_back_as_text(text: str) -> None:
    assert parse_tool_calls(text) is None
    chunks = _chunks(text, _TOOLS)
    assert not any(isinstance(chunk, ToolCallChunk) for chunk in chunks)
    assert _text(chunks, thinking=False) == text


# From MiniMaxAI/MiniMax-M3.1-Flash chat_template.jinja: the effort line of the system
# prompt and the generation prompt.
_M31_TEMPLATE = (
    "{%- if reasoning_effort is defined and reasoning_effort is not none -%}"
    "{{- '<effort>' ~ reasoning_effort ~ '</effort>' }}{%- endif -%}"
    "{%- for m in messages -%}{{- ']~b]' ~ m.role ~ '\\n' ~ m.content ~ '[e~[\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- ']~b]ai' ~ '\\n' }}{%- endif -%}"
)


@final
class _TemplateTokenizer:
    def __init__(self) -> None:
        self.chat_template = _M31_TEMPLATE
        self.has_thinking = True
        self.think_start = "<think>"
        self.think_end = "</think>"
        self._tokenizer = self
        self._chat_template = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny]
    ) -> str:
        assert not tokenize
        assert tools is None
        kwargs.pop("return_dict", None)
        return (
            jinja2.Environment()
            .from_string(self.chat_template)
            .render(
                messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
            )
        )


def _render(
    model: str, reasoning_effort: ReasoningEffort | None, enable_thinking: bool | None
) -> str:
    params = TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    return apply_chat_template(_TemplateTokenizer(), params)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("effort", "thinking", "expected"),
    [
        (None, None, ""),
        (None, True, "<effort>medium</effort>"),  # thinking on without a level
        ("medium", True, "<effort>medium</effort>"),
        ("high", True, "<effort>high</effort>"),
        ("xhigh", True, "<effort>xhigh</effort>"),
        ("low", True, "<effort>low</effort>"),
        ("minimal", True, "<effort>low</effort>"),
    ],
)
def test_effort_uses_the_templates_vocabulary(
    effort: ReasoningEffort | None, thinking: bool | None, expected: str
) -> None:
    prompt = _render("MiniMaxAI/MiniMax-M3.1-Flash", effort, thinking)
    assert prompt == f"{expected}]~b]user\nhi[e~[\n]~b]ai\n"


def test_thinking_off_starts_the_turn_without_reasoning() -> None:
    prompt = _render("mlx-community/MiniMax-M3.1-Flash-Text-3bit", "none", False)
    assert prompt == "]~b]user\nhi[e~[\n]~b]ai\n</mm:think>"


# From MiniMaxAI/MiniMax-M3 chat_template.jinja: the thinking-mode line of the system
# prompt and the generation prompt.
_M3_TEMPLATE = (
    "{%- if thinking_mode is defined and thinking_mode == 'disabled' -%}"
    "{{- 'Current thinking mode: disabled.' }}{%- else -%}"
    "{{- 'Current thinking mode: adaptive.' }}{%- endif -%}"
    "{%- for m in messages -%}{{- ']~b]' ~ m.role ~ '\\n' ~ m.content ~ '[e~[\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- ']~b]ai' ~ '\\n' }}"
    "{%- if thinking_mode is defined and thinking_mode == 'disabled' -%}"
    "{{- '</mm:think>' }}{%- endif -%}{%- endif -%}"
)


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (
            False,
            "Current thinking mode: disabled.]~b]user\nhi[e~[\n]~b]ai\n</mm:think>",
        ),
        (True, "Current thinking mode: adaptive.]~b]user\nhi[e~[\n]~b]ai\n"),
        (None, "Current thinking mode: adaptive.]~b]user\nhi[e~[\n]~b]ai\n"),
    ],
)
def test_m3_turns_thinking_off_with_its_own_switch(
    thinking: bool | None, expected: str
) -> None:
    tokenizer = _TemplateTokenizer()
    tokenizer.chat_template = _M3_TEMPLATE
    params = TextGenerationTaskParams(
        model=ModelId("mlx-community/MiniMax-M3-4bit"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        enable_thinking=thinking,
    )
    assert apply_chat_template(tokenizer, params) == expected  # pyright: ignore[reportArgumentType]


def test_other_models_receive_the_effort_unchanged() -> None:
    prompt = _render("mlx-community/MiniMax-M2.7-4bit", "minimal", True)
    assert prompt.startswith("<effort>minimal</effort>")
