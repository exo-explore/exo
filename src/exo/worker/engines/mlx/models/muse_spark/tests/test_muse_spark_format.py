"""Muse Spark's output format (messages with recipients, ATEM tool calls) and its
reasoning effort.

The tool calls are rendered with the official chat template's ``render_atem`` macro
(meta-models/Muse-Spark-Open-816B-A42B, chat_template.jinja), and parsed back.
"""

import json
from collections.abc import Generator, Iterator
from typing import Any, cast, final

import jinja2
import pytest
from mlx_lm.models.muse_spark import Model as MuseSparkModelClass

from exo.shared.types.chunks import GenerationChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.models.muse_spark.parser import (
    parse_tool_calls,
    recipient_of,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.llm_inference.model_output_parsers import apply_all_parsers

# apply_all_parsers finds the output parser by the loaded model's class.
MuseSparkModel = cast(type[Model], MuseSparkModelClass)
_MODEL_ID = ModelId("mlx-community/Muse-Spark-Open-816B-A42B-4bit")

# The official template's macro for one tool call (its escaping macros inlined).
_RENDER_ATEM = """
{%- macro esc(value) -%}{{- value | string | replace('<|', '< |') -}}{%- endmacro -%}
{%- macro esc_attr(value) -%}
    {{- esc(value) | replace('&', '&amp;') | replace('"', '&quot;') | replace("'", '&#39;')
        | replace('<', '&lt;') | replace('>', '&gt;') -}}
{%- endmacro -%}
{%- macro render_atem(tc) -%}
    {%- set args = tc.function.arguments -%}
    {{- '<atem:function_calls>\\n<atem:invoke name="' -}}{{- esc_attr(tc.function.name) -}}{{- '">\\n' -}}
    {%- for k, v in args.items() -%}
        {{- '<atem:parameter name="' -}}{{- esc_attr(k) -}}{{- '">' -}}
        {%- if v is boolean -%}
            {%- if v -%}true{%- else -%}false{%- endif -%}
        {%- elif v is none -%}
            null
        {%- elif v is mapping or (v is iterable and v is not string) -%}
            {{- esc(v | tojson) -}}
        {%- else -%}
            {{- esc(v) -}}
        {%- endif -%}
        {{- '</atem:parameter>\\n' -}}
    {%- endfor -%}
    {{- '</atem:invoke>\\n</atem:function_calls>' -}}
{%- endmacro -%}
{{- render_atem(call) -}}
"""

_WEATHER_ARGUMENTS: dict[str, Any] = {
    "city": "Paris",
    "days": 3,
    "metric": True,
    "note": "  spaces stay  ",
}
_WRITE_ARGUMENTS: dict[str, Any] = {
    "path": "a/b.py",
    "content": 'if x < 3:\n    print("<b>")\n',
    "meta": {"tags": ["x", "y"], "mode": 420},
    "empty": [],
    "ratio": 0.5,
    "flag": "true",
    "count": "42",
}

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
                    "note": {"type": "string"},
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
                        },
                    },
                    "empty": {"type": "array"},
                    "ratio": {"type": "number"},
                    "flag": {"type": "string"},
                    "count": {"type": "string"},
                },
            },
        },
    },
]


def _call(name: str, arguments: dict[str, Any]) -> str:
    return (
        jinja2.Environment()
        .from_string(_RENDER_ATEM)
        .render(call={"function": {"name": name, "arguments": arguments}})
    )


def _pieces(text: str) -> list[str]:
    """Detokenized segments: each special token whole, other text in 3-character bits."""
    pieces: list[str] = []
    rest = text
    while rest:
        if rest.startswith("<|"):
            end = rest.index("|>") + 2
            pieces.append(rest[:end])
            rest = rest[end:]
            continue
        cut = rest.find("<|")
        part = rest if cut < 0 else rest[:cut]
        pieces.extend(part[i : i + 3] for i in range(0, len(part), 3))
        rest = rest[len(part) :]
    return pieces


def _responses(text: str) -> Generator[GenerationResponse | None]:
    pieces = _pieces(text)
    for index, piece in enumerate(pieces):
        yield GenerationResponse(text=piece, token=index, usage=None)
    # The turn's <|eot|> is an end-of-sequence token: it doesn't reach the text.
    yield GenerationResponse(
        text="", token=len(pieces), usage=None, finish_reason="stop"
    )


@final
class _Tokenizer:
    has_thinking = False
    think_start = None
    think_end = None


def _chunks(
    text: str, tools: list[dict[str, Any]] | None = None
) -> list[GenerationChunk]:
    chunks: Iterator[GenerationChunk | None] = apply_all_parsers(
        _responses(text),
        "<|start|>assistant",
        None,
        _Tokenizer(),  # pyright: ignore[reportArgumentType]
        MuseSparkModel,
        _MODEL_ID,
        tools,
    )
    return [chunk for chunk in chunks if chunk is not None]


def _text(chunks: list[GenerationChunk], thinking: bool) -> str:
    return "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and chunk.is_thinking == thinking
    )


def _calls(chunks: list[GenerationChunk]) -> list[tuple[str, Any]]:
    return [
        (call.name, json.loads(call.arguments))
        for chunk in chunks
        if isinstance(chunk, ToolCallChunk)
        for call in chunk.tool_calls
    ]


def test_reasoning_and_answer_go_to_their_channels() -> None:
    chunks = _chunks(
        " to=self<|message|>Two plus two.<|eom|><|start|>assistant to=user<|message|>It is 4."
    )
    assert _text(chunks, thinking=True) == "Two plus two."
    assert _text(chunks, thinking=False) == "It is 4."
    last = chunks[-1]
    assert isinstance(last, TokenChunk) and last.finish_reason == "stop"


def test_a_turn_without_reasoning_and_a_preamble() -> None:
    chunks = _chunks(
        " to=commentary<|message|>Working on it.<|eom|>"
        "<|start|>assistant to=user<|message|>Done."
    )
    assert _text(chunks, thinking=True) == ""
    assert _text(chunks, thinking=False) == "Working on it.Done."


def test_a_marker_next_to_held_back_text() -> None:
    """The detokenizer holds text back to word boundaries, so a header's end can arrive
    with the marker after it, and a message's start with the next words."""
    responses = [
        GenerationResponse(text=" to=self<|message|>Hm", token=0, usage=None),
        GenerationResponse(text=".<|eom|><|start|>assistant", token=1, usage=None),
        GenerationResponse(text=" to=user<|message|>Yes", token=2, usage=None),
        GenerationResponse(text=".", token=3, usage=None, finish_reason="stop"),
    ]
    chunks = [
        chunk
        for chunk in apply_all_parsers(
            (response for response in responses),
            "",
            None,
            _Tokenizer(),  # pyright: ignore[reportArgumentType]
            MuseSparkModel,
            _MODEL_ID,
            None,
        )
        if chunk is not None
    ]
    assert _text(chunks, thinking=True) == "Hm."
    assert _text(chunks, thinking=False) == "Yes."


def test_parallel_tool_calls_parse_back_to_typed_arguments() -> None:
    chunks = _chunks(
        " to=self<|message|>Need the weather.<|eom|>"
        "<|start|>assistant to=commentary<|message|>Let me check.<|eom|>"
        f"<|start|>assistant to=get_weather<|message|>{_call('get_weather', _WEATHER_ARGUMENTS)}<|eom|>"
        f"<|start|>assistant to=write_file<|message|>{_call('write_file', _WRITE_ARGUMENTS)}",
        _TOOLS,
    )
    assert _text(chunks, thinking=True) == "Need the weather."
    assert _text(chunks, thinking=False) == "Let me check."
    assert _calls(chunks) == [
        ("get_weather", _WEATHER_ARGUMENTS),
        ("write_file", _WRITE_ARGUMENTS),
    ]
    # One tool-call chunk at the end: the API ends the stream at it.
    assert isinstance(chunks[-1], ToolCallChunk)
    assert sum(isinstance(chunk, ToolCallChunk) for chunk in chunks) == 1


def test_without_a_schema_values_stay_text() -> None:
    calls = parse_tool_calls(_call("get_weather", _WEATHER_ARGUMENTS))
    assert calls is not None
    assert json.loads(calls[0].arguments) == {
        "city": "Paris",
        "days": "3",
        "metric": "true",
        "note": "  spaces stay  ",
    }


def test_escaped_names_are_unescaped() -> None:
    calls = parse_tool_calls(_call('ns.say"hi"', {"a&b": "<x>"}))
    assert calls is not None
    assert calls[0].name == 'ns.say"hi"'
    assert json.loads(calls[0].arguments) == {"a&b": "<x>"}


@pytest.mark.parametrize(
    "text",
    [
        '<atem:function_calls>\n<atem:invoke name="f">\n<atem:parameter name="a">1'
        "</atem:invoke>\n</atem:function_calls>",
        '<atem:function_calls>\n<atem:invoke name="f">\n</atem:function_calls>',
        "<atem:function_calls>\n</atem:function_calls>",
        'Sure: <atem:function_calls>\n<atem:invoke name="f">\n</atem:invoke>\n'
        "</atem:function_calls>",
    ],
)
def test_a_malformed_call_comes_back_as_text(text: str) -> None:
    assert parse_tool_calls(text) is None
    chunks = _chunks(f" to=f<|message|>{text}", _TOOLS)
    assert not any(isinstance(chunk, ToolCallChunk) for chunk in chunks)
    assert _text(chunks, thinking=False) == text


def test_text_without_a_header_is_not_lost() -> None:
    text = "An answer with no header at all. " * 10
    chunks = _chunks(text)
    assert _text(chunks, thinking=False) == text


@pytest.mark.parametrize(
    ("header", "recipient"),
    [
        (" to=self", "self"),
        ("assistant to=user", "user"),
        ("assistant to=functions.lookup ", "functions.lookup"),
        ("assistant", "user"),
    ],
)
def test_recipients(header: str, recipient: str) -> None:
    assert recipient_of(header) == recipient


# From the official chat_template.jinja: the effort default, its check and the system
# message's effort line, around a minimal rendering of the messages.
_EFFORT_TEMPLATE = (
    "{%- set reasoning_effort = reasoning_effort | default('xhigh', true) -%}"
    "{%- set reasoning_effort_map = {'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}"
    "{%- if not (reasoning_effort is string and reasoning_effort in reasoning_effort_map) -%}"
    "{{- raise_exception('Unsupported reasoning_effort: ' ~ reasoning_effort) -}}"
    "{%- endif -%}"
    "{{- '<|start|>system<|message|>Reasoning strength: ' ~ "
    "reasoning_effort_map[reasoning_effort] ~ '.<|eot|>' -}}"
    "{%- for m in messages -%}"
    "{{- '<|start|>' ~ m.role ~ '<|message|>' ~ m.content ~ '<|eot|>' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}"
)


def _raise_exception(message: str) -> None:
    raise ValueError(message)


@final
class _TemplateTokenizer:
    def __init__(self) -> None:
        self.chat_template = _EFFORT_TEMPLATE
        self.has_thinking = False
        self.think_start = None
        self.think_end = None
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
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                raise_exception=_raise_exception,
                **kwargs,
            )
        )


def _render(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
    model: ModelId = _MODEL_ID,
) -> str:
    params = TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    return apply_chat_template(_TemplateTokenizer(), params)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("effort", "thinking", "strength"),
    [
        (None, None, 512),  # the template's default, xhigh
        ("xhigh", None, 512),
        ("high", None, 256),
        ("medium", None, 32),
        ("low", None, 8),
        # The model can't turn reasoning off: exo's lowest levels give the least.
        ("minimal", None, 8),
        ("none", False, 8),
    ],
)
def test_effort_uses_the_templates_vocabulary(
    effort: ReasoningEffort | None, thinking: bool | None, strength: int
) -> None:
    assert _render(effort, thinking) == (
        f"<|start|>system<|message|>Reasoning strength: {strength}.<|eot|>"
        "<|start|>user<|message|>hi<|eot|><|start|>assistant"
    )


def test_other_models_receive_the_effort_unchanged() -> None:
    with pytest.raises(ValueError, match="Unsupported reasoning_effort: minimal"):
        _render("minimal", None, ModelId("someone/Other-Model"))
