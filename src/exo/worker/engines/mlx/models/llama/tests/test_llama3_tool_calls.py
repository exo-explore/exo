"""Llama 3.x tool calls, through exo's request path, the official chat templates and exo's
output parser chain.

fixtures/llama3_2_chat_template.jinja is the chat_template of
mlx-community/Llama-3.2-3B-Instruct-4bit's tokenizer_config.json (revision
7f0dc925e0d0afb0322d96f9255cfddf2ba5636e), the same in every Llama 3.2 1B and 3B card.
fixtures/llama3_1_chat_template.jinja is the one of
mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (revision
241a666dad6cb93c8ff213d39a7f34a36bf26db4), the same in every Llama 3.1 8B and 70B and
Llama 3.3 70B card.
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from mlx_lm.models.llama import Model as LlamaModelClass

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    StreamingChoiceResponse,
    ToolCall,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.llama.parser import (
    in_ipython_environment,
    parse_llama3,
)
from exo.worker.engines.mlx.models.llama.spec import LLAMA_3_EOS_TOKENS
from exo.worker.engines.mlx.models.spec_registry import added_eos_tokens_for_model_id
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    add_eos_tokens,
    apply_chat_template,
    get_eos_token_ids_for_model,
)
from exo.worker.runner.llm_inference.model_output_parsers import apply_all_parsers

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = {
    "3.2": (FIXTURES / "llama3_2_chat_template.jinja").read_text(),
    "3.1": (FIXTURES / "llama3_1_chat_template.jinja").read_text(),
}
MODEL_IDS = {
    "3.2": ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
    "3.1": ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
}
# apply_all_parsers finds the output parser by the loaded model's class.
LlamaModel = cast(type[Model], LlamaModelClass)

SPECIAL_TOKENS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
)

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}
ADD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    },
}
TOOLS = [WEATHER_TOOL, TIME_TOOL, ADD_TOOL]

# What Llama 3.2 3B (4-bit) wrote on exo before this fix, at temperature 0, for "What's
# the weather in Paris right now?" with WEATHER_TOOL: its JSON call, <|eom_id|>, and,
# since <|eom_id|> wasn't an EOS token, more turns of its own.
MAIN_OUTPUT = (
    '{"name": "get_weather", "parameters": {"city": "Paris", "unit": "celsius"}}'
    "<|eom_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris", "unit": '
    '"celsius"}}<|eom_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    "I've used the celsius unit for the response."
)


def _raise_exception(message: str) -> NoReturn:
    raise TemplateError(message)


def _tojson(
    value: object,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


class TemplateOnlyTokenizer:
    """Renders a real template in the Jinja environment transformers'
    apply_chat_template uses (sandboxed, trim/lstrip blocks, loop controls, its tojson
    and raise_exception, the special tokens)."""

    has_thinking = False

    def __init__(self, chat_template: str) -> None:
        self.chat_template = chat_template
        self._tokenizer = self
        self._chat_template = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        return_dict: bool = False,
        **kwargs: object,
    ) -> str:
        assert not tokenize and not return_dict
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
        )
        environment.filters["tojson"] = _tojson
        template = environment.from_string(chat_template or self.chat_template)
        return template.render(
            raise_exception=_raise_exception,
            messages=conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            bos_token="<|begin_of_text|>",
            eos_token="<|eot_id|>",
            **kwargs,
        )


def _request(version: str, body: dict[str, Any]) -> str:
    """The prompt exo renders for a Chat Completions request."""
    request = ChatCompletionRequest.model_validate(
        {"model": MODEL_IDS[version], **body}
    )
    params = asyncio.run(chat_request_to_text_generation(request))
    return apply_chat_template(
        TemplateOnlyTokenizer(TEMPLATES[version]),  # pyright: ignore[reportArgumentType]
        params,
        "llama",
    )


def _tool_prompt(version: str, tools: list[dict[str, Any]] | None = None) -> str:
    return _request(
        version,
        {
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [WEATHER_TOOL] if tools is None else tools,
        },
    )


def _rendered_call(version: str, name: str, arguments: dict[str, Any]) -> str:
    """How the template writes an assistant turn that calls a tool."""
    prompt = _request(
        version,
        {
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                # (A history ending with an assistant message is a prefill.)
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            ],
            "tools": [WEATHER_TOOL, TIME_TOOL, ADD_TOOL],
        },
    )
    header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    turn = prompt[prompt.rindex(header, 0, prompt.rindex(header)) + len(header) :]
    return turn[: turn.index("<|eot_id|>")]


_SPECIAL = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


def _pieces(text: str) -> list[str]:
    """Special tokens whole, the rest in small pieces, as the tokenizer would."""
    pieces: list[str] = []
    for part in _SPECIAL.split(text):
        if _SPECIAL.fullmatch(part):
            pieces.append(part)
        else:
            pieces.extend(part[i : i + 3] for i in range(0, len(part), 3))
    return pieces


def _responses(
    text: str, finish_reason: str = "stop"
) -> Generator[GenerationResponse | None]:
    pieces = _pieces(text)
    for index, piece in enumerate(pieces):
        yield GenerationResponse(text=piece, token=index, usage=None)
    yield GenerationResponse(
        text="",
        token=len(pieces),
        usage=None,
        finish_reason=finish_reason,  # pyright: ignore[reportArgumentType]
    )


class _Tokenizer:
    # Llama 3's tokenizer has no thinking tags.
    has_thinking = False
    think_start = None
    think_end = None


def _chunks(
    text: str,
    prompt: str,
    tools: list[dict[str, Any]] | None = TOOLS,
    finish_reason: str = "stop",
) -> list[GenerationChunk]:
    chunks = apply_all_parsers(
        _responses(text, finish_reason),
        prompt,
        None,
        _Tokenizer(),  # pyright: ignore[reportArgumentType]
        LlamaModel,
        MODEL_IDS["3.2"],
        tools,
    )
    return [chunk for chunk in chunks if chunk is not None]


def _content(chunks: list[GenerationChunk]) -> str:
    return "".join(chunk.text for chunk in chunks if isinstance(chunk, TokenChunk))


def _calls(chunks: list[GenerationChunk]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call.name, cast(dict[str, Any], json.loads(call.arguments)))
        for chunk in chunks
        if isinstance(chunk, ToolCallChunk)
        for call in chunk.tool_calls
    ]


def _assert_no_special_tokens(chunks: list[GenerationChunk]) -> None:
    for chunk in chunks:
        dumped = chunk.model_dump_json()
        for token in SPECIAL_TOKENS:
            assert token not in dumped, (token, chunk)


@pytest.mark.parametrize("version", TEMPLATES)
def test_tools_put_the_prompt_in_the_ipython_environment(version: str) -> None:
    assert in_ipython_environment(_tool_prompt(version))
    no_tools = _request(
        version, {"messages": [{"role": "user", "content": "What's 2 + 2?"}]}
    )
    assert not in_ipython_environment(no_tools)


@pytest.mark.parametrize("version", TEMPLATES)
@pytest.mark.parametrize("opener", ["", "<|python_tag|>", " "])
def test_the_templates_tool_call_comes_back_as_a_tool_call(
    version: str, opener: str
) -> None:
    arguments = {"city": "Paris", "unit": "celsius"}
    rendered = _rendered_call(version, "get_weather", arguments)
    assert rendered == (
        '{"name": "get_weather", "parameters": {"city": "Paris", "unit": "celsius"}}'
    )

    chunks = _chunks(opener + rendered, _tool_prompt(version), [WEATHER_TOOL])

    assert _calls(chunks) == [("get_weather", arguments)]
    assert _content(chunks) == ""
    assert isinstance(chunks[-1], ToolCallChunk)
    _assert_no_special_tokens(chunks)


def test_what_llama_3_2_wrote_before_the_fix_is_one_tool_call() -> None:
    # A checkpoint whose EOS ids don't list <|eom_id|> runs on as it did; the message
    # still ends there and nothing after it leaks.
    chunks = _chunks(MAIN_OUTPUT, _tool_prompt("3.2"), [WEATHER_TOOL])

    assert _calls(chunks) == [("get_weather", {"city": "Paris", "unit": "celsius"})]
    assert _content(chunks) == ""
    _assert_no_special_tokens(chunks)


def test_a_built_in_tool_call_comes_back_as_a_tool_call() -> None:
    # Llama 3.1's template writes a call to one of its built-in tools in Python.
    prompt = TemplateOnlyTokenizer(TEMPLATES["3.1"]).apply_chat_template(
        [
            {"role": "user", "content": "Search the weather in Paris."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "brave_search",
                            "arguments": {"query": "weather in Paris"},
                        }
                    }
                ],
            },
        ],
        builtin_tools=["brave_search", "wolfram_alpha"],
    )
    turn = prompt.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
    assert turn == '<|python_tag|>brave_search.call(query="weather in Paris")<|eom_id|>'
    assert in_ipython_environment(prompt)

    brave_search = {"type": "function", "function": {"name": "brave_search"}}
    chunks = _chunks(turn.removesuffix("<|eom_id|>"), prompt, [brave_search])

    assert _calls(chunks) == [("brave_search", {"query": "weather in Paris"})]
    _assert_no_special_tokens(chunks)


def test_an_answer_streams_as_content() -> None:
    chunks = _chunks("Bonjour ! Comment allez-vous ?", _tool_prompt("3.2"))

    assert _calls(chunks) == []
    assert _content(chunks) == "Bonjour ! Comment allez-vous ?"
    # Passed on as it comes, not held back to the end.
    assert sum(1 for chunk in chunks if isinstance(chunk, TokenChunk)) > 5


def test_code_after_the_python_tag_comes_back_as_text() -> None:
    code = "import math\nprint(math.sqrt(2))"
    chunks = _chunks("<|python_tag|>" + code, _tool_prompt("3.1"))

    assert _calls(chunks) == []
    assert _content(chunks) == code
    _assert_no_special_tokens(chunks)


def test_a_call_cut_off_by_the_token_limit_comes_back_as_text() -> None:
    text = '{"name": "get_weather", "parameters": {"ci'
    chunks = _chunks(text, _tool_prompt("3.2"), [WEATHER_TOOL], "length")

    assert _calls(chunks) == []
    assert _content(chunks) == text
    last = chunks[-1]
    assert isinstance(last, TokenChunk) and last.finish_reason == "length"


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "get_weather", "parameters": {"city": "Paris"}}; '
        '{"name": "get_time", "parameters": {"city": "Paris"}}',
        '{"name": "get_weather", "parameters": {"city": "Paris"}}\n'
        '{"name": "get_time", "parameters": {"city": "Paris"}}',
        '<|python_tag|>[{"name": "get_weather", "parameters": {"city": "Paris"}}, '
        '{"name": "get_time", "arguments": {"city": "Paris"}}]',
    ],
)
def test_calls_one_after_another(text: str) -> None:
    chunks = _chunks(text, _tool_prompt("3.1"), [WEATHER_TOOL, TIME_TOOL])

    assert _calls(chunks) == [
        ("get_weather", {"city": "Paris"}),
        ("get_time", {"city": "Paris"}),
    ]
    _assert_no_special_tokens(chunks)


def test_text_around_a_call_stays_content() -> None:
    before = _chunks(
        'Let me check. <|python_tag|>{"name": "get_time", "parameters": {"city": "Rome"}}',
        _tool_prompt("3.2"),
    )
    assert _content(before) == "Let me check."
    assert _calls(before) == [("get_time", {"city": "Rome"})]

    # What Llama 3.2 3B wrote after the results of two parallel calls.
    own_line = _chunks(
        "Here is the JSON for a function call with its proper arguments that best "
        'answers the given prompt:\n\n{"name": "get_weather", "parameters": {"city": '
        '"Paris", "unit": "celsius"}}',
        _tool_prompt("3.2"),
    )
    assert _content(own_line) == (
        "Here is the JSON for a function call with its proper arguments that best "
        "answers the given prompt:"
    )
    assert _calls(own_line) == [("get_weather", {"city": "Paris", "unit": "celsius"})]

    after = _chunks(
        '{"name": "get_time", "parameters": {"city": "Rome"}}\n\nThat gives the time.',
        _tool_prompt("3.2"),
    )
    assert _calls(after) == [("get_time", {"city": "Rome"})]
    assert _content(after) == "That gives the time."
    assert isinstance(after[-1], ToolCallChunk)

    around = _chunks(
        'Checking.\n{"name": "get_time", "parameters": {"city": "Rome"}}\nDone.',
        _tool_prompt("3.2"),
    )
    assert _calls(around) == [("get_time", {"city": "Rome"})]
    assert _content(around) == "Checking.\nDone."


def test_json_in_an_answer_that_is_no_call_comes_back_as_written() -> None:
    text = 'The config:\n{\n  "name": "exo",\n  "port": 52415\n}\nThen restart it.'
    chunks = _chunks(text, _tool_prompt("3.2"), [WEATHER_TOOL])

    assert _calls(chunks) == []
    assert _content(chunks) == text


@pytest.mark.parametrize(
    "separator",
    ["<|python_tag|>", "<|eom_id|><|start_header_id|>assistant<|end_header_id|>\n\n"],
)
def test_nothing_after_a_call_leaks(separator: str) -> None:
    call = '{"name": "get_time", "parameters": {"city": "Rome"}}'
    chunks = _chunks(call + separator + call, _tool_prompt("3.2"), [TIME_TOOL])

    expected = [("get_time", {"city": "Rome"})]
    # A second call after <|python_tag|> is a call too; after the end of the message,
    # nothing is.
    assert _calls(chunks) == (
        expected * 2 if separator == "<|python_tag|>" else expected
    )
    assert _content(chunks) == ""
    _assert_no_special_tokens(chunks)


def test_a_new_header_ends_the_message() -> None:
    chunks = _chunks(
        '{"name": "get_time", "parameters": {"city": "Rome"}}'
        "<|start_header_id|>assistant<|end_header_id|>\n\nMore.",
        _tool_prompt("3.2"),
    )
    assert _calls(chunks) == [("get_time", {"city": "Rome"})]
    assert _content(chunks) == ""
    _assert_no_special_tokens(chunks)


def test_arguments_follow_the_tool_schemas() -> None:
    chunks = _chunks(
        '{"name": "add", "parameters": {"a": "17", "b": 25}}',
        _tool_prompt("3.2", [ADD_TOOL]),
        [ADD_TOOL],
    )
    assert _calls(chunks) == [("add", {"a": 17, "b": 25})]


def test_json_asked_for_without_tools_stays_content() -> None:
    prompt = _request(
        "3.2",
        {
            "messages": [
                {"role": "user", "content": "A JSON object with name and parameters."}
            ]
        },
    )
    text = '{"name": "get_weather", "parameters": {"city": "Paris"}}'

    chunks = _chunks(text, prompt, None)

    assert _calls(chunks) == []
    assert _content(chunks) == text


def _streamed_late(chunks: list[GenerationChunk]) -> str:
    """The content that came only with the message's last response."""
    last = chunks[-1]
    return last.text if isinstance(last, TokenChunk) else ""


def test_json_in_an_answer_streams_once_it_is_no_call() -> None:
    text = 'The config:\n{\n  "name": "exo",\n  "port": 52415\n}\nThen restart it.'
    chunks = _chunks(text, _tool_prompt("3.2"))

    assert _content(chunks) == text
    # Held back only up to the "}" that ends the object.
    assert "restart" not in _streamed_late(chunks)


def test_code_with_a_brace_line_keeps_streaming() -> None:
    # A "{" that opens a line is held back only while what follows can be JSON.
    text = "Here:\nint main()\n{\n  return 0;\n}\n" + " ".join(
        f"word{index}" for index in range(50)
    )
    chunks = _chunks(text, _tool_prompt("3.2"))

    assert _calls(chunks) == []
    assert _content(chunks) == text
    assert "word" not in _streamed_late(chunks)
    assert sum(1 for chunk in chunks if isinstance(chunk, TokenChunk)) > 50


def test_a_fenced_json_example_in_an_answer_is_not_a_tool_call() -> None:
    # Coding agents send tools with every request; an answer that shows a call as an
    # example stays text.
    text = (
        "A call looks like this:\n```json\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "```\nThat is all."
    )
    chunks = _chunks(text, _tool_prompt("3.2"))

    assert _calls(chunks) == []
    assert _content(chunks) == text

    # After the fence, a call on a line of its own is one again.
    call = '{"name": "get_weather", "parameters": {"city": "Rome"}}'
    after = _chunks(text + "\n" + call, _tool_prompt("3.2"))
    assert _calls(after) == [("get_weather", {"city": "Rome"})]
    assert _content(after) == text


@pytest.mark.parametrize("opener", ["", "<|python_tag|>"])
def test_a_call_to_a_tool_the_request_has_not_stays_text(opener: str) -> None:
    text = '{"name": "web_search", "parameters": {"query": "weather in Paris"}}'
    chunks = _chunks(opener + text, _tool_prompt("3.2"), [WEATHER_TOOL])

    assert _calls(chunks) == []
    assert _content(chunks) == text
    _assert_no_special_tokens(chunks)


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "add", "parameters": {"a": NaN, "b": 1}}',
        '{"name": "add", "parameters": {"a": -Infinity, "b": 1}}',
        # A number too big for a float, which Python reads as an infinity.
        '{"name": "add", "parameters": {"a": 1e999, "b": 1}}',
        '<|python_tag|>{"name": "add", "parameters": {"a": Infinity, "b": 1}}',
        "<|python_tag|>add.call(a=1e999, b=1)",
    ],
)
def test_arguments_json_cannot_hold_stay_text(text: str) -> None:
    chunks = _chunks(text, _tool_prompt("3.2", [ADD_TOOL]), [ADD_TOOL])

    assert _calls(chunks) == []
    assert _content(chunks) == text.removeprefix("<|python_tag|>")


def _parsed(pieces: list[str]) -> tuple[str, list[tuple[str, str]]]:
    def responses() -> Generator[GenerationResponse | None]:
        for index, piece in enumerate(pieces):
            yield GenerationResponse(text=piece, token=index, usage=None)
        yield GenerationResponse(
            text="", token=len(pieces), usage=None, finish_reason="stop"
        )

    content = ""
    calls: list[tuple[str, str]] = []
    for response in parse_llama3(responses(), frozenset({"get_weather", "get_time"})):
        if isinstance(response, ToolCallResponse):
            calls += [(call.name, call.arguments) for call in response.tool_calls]
        elif response is not None:
            content += response.text
    return content, calls


@pytest.mark.parametrize(
    "text",
    [
        MAIN_OUTPUT,
        'Checking.\n  {"name": "get_time", "parameters": {"city": "Rome"}}\nDone.',
        'The config:\n{\n  "name": "exo",\n  "port": 52415\n}\n{"name": "get_time", '
        '"parameters": {"city": "Rome"}}',
        "Here:\nint main()\n{\n  return 0;\n}\nThat is C.",
        '```json\n{"name": "get_time", "parameters": {}}\n```\n'
        '{"name": "get_weather", "parameters": {"city": "Rome"}}',
        'Text {not a call}\n{"a": [1, 2.5e3, true, null, "x\\"y"]}\nmore\n',
        'Sure.\n{"name": "get_weather", "parameters": {"city": "Rome"}}; '
        '{"name": "get_time", "parameters": {"city": "Rome"}}',
        'Let me check. <|python_tag|>{"name": "get_time", "parameters": {}}',
        '{\n<|python_tag|>{"name": "get_time", "parameters": {}}',
    ],
)
def test_where_the_text_is_split_does_not_matter(text: str) -> None:
    whole = _parsed([text])
    # The tokenizer never splits a special token.
    inside_special_tokens = {
        index
        for match in _SPECIAL.finditer(text)
        for index in range(match.start() + 1, match.end())
    }
    for index in range(1, len(text)):
        if index not in inside_special_tokens:
            assert _parsed([text[:index], text[index:]]) == whole, index
    assert _parsed(_pieces(text)) == whole


async def _stream(
    chunks: list[GenerationChunk],
) -> AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk]:
    for chunk in chunks:
        assert isinstance(chunk, (ErrorChunk, ToolCallChunk, TokenChunk))
        yield chunk


async def _non_streaming(chunks: list[GenerationChunk]) -> ChatCompletionChoice:
    body = "".join(
        [part async for part in collect_chat_response(CommandId("c"), _stream(chunks))]
    )
    [choice] = ChatCompletionResponse.model_validate_json(body).choices
    assert isinstance(choice, ChatCompletionChoice)
    return choice


async def _streaming(chunks: list[GenerationChunk]) -> list[StreamingChoiceResponse]:
    events = [
        part async for part in generate_chat_stream(CommandId("c"), _stream(chunks))
    ]
    assert events[-1] == "data: [DONE]\n\n"
    choices: list[StreamingChoiceResponse] = []
    for event in events[:-1]:
        if event.startswith("data: "):
            response = ChatCompletionResponse.model_validate_json(
                event.removeprefix("data: ")
            )
            for choice in response.choices:
                assert isinstance(choice, StreamingChoiceResponse)
                choices.append(choice)
    return choices


def _arguments(call: ToolCall) -> object:
    return cast(object, json.loads(call.function.arguments))


def test_the_api_returns_openai_tool_calls_streaming_and_not() -> None:
    chunks = _chunks(MAIN_OUTPUT, _tool_prompt("3.2"), [WEATHER_TOOL])
    expected = {"city": "Paris", "unit": "celsius"}

    choice = asyncio.run(_non_streaming(chunks))
    assert choice.finish_reason == "tool_calls"
    assert not choice.message.content
    [call] = choice.message.tool_calls or []
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert _arguments(call) == expected

    streamed = asyncio.run(_streaming(chunks))
    [last] = [choice for choice in streamed if choice.finish_reason is not None]
    assert last.finish_reason == "tool_calls"
    [streamed_call] = last.delta.tool_calls or []
    assert streamed_call.function.name == "get_weather"
    assert _arguments(streamed_call) == expected
    assert all(not choice.delta.content for choice in streamed)

    everything = choice.model_dump_json() + "".join(
        choice.model_dump_json() for choice in streamed
    )
    for token in SPECIAL_TOKENS:
        assert token not in everything


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-8bit",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "mlx-community/llama-3.3-70b-instruct-fp16",
        "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-4bit",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ],
)
def test_llama_3_ids_get_the_end_of_message_tokens(model_id: str) -> None:
    # <|end_of_text|>, <|eom_id|>, <|eot_id|>: the checkpoints' config.json eos_token_id,
    # added to the tokenizer's own EOS token, which no id rule replaces.
    assert added_eos_tokens_for_model_id(model_id) == LLAMA_3_EOS_TOKENS
    assert get_eos_token_ids_for_model(ModelId(model_id)) is None


def test_other_llama_architecture_ids_keep_their_eos() -> None:
    for model_id in ("openlm-research/open_llama_3b", "mistralai/Mistral-7B-v0.3"):
        assert get_eos_token_ids_for_model(ModelId(model_id)) is None
        assert added_eos_tokens_for_model_id(model_id) == ()


@dataclass(frozen=True)
class _AddedToken:
    content: str
    special: bool


class _LoadedTokenizer:
    """What add_eos_tokens reads and changes of a loaded tokenizer. Its EOS ids start as
    mlx-lm's TokenizerWrapper sets them without explicit ids: the tokenizer's own."""

    def __init__(
        self, eos_token_id: int, added_tokens: dict[int, tuple[str, bool]]
    ) -> None:
        self.eos_token_ids = {eos_token_id}
        self.added_tokens_decoder = {
            token_id: _AddedToken(content, special)
            for token_id, (content, special) in added_tokens.items()
        }

    def add_eos_token(self, token: str) -> None:
        self.eos_token_ids.add(int(token))


_END_OF_TEXT = ("<|end_of_text|>", True)
_EOT = ("<|eot_id|>", True)
_IM_END = ("<|im_end|>", True)


@pytest.mark.parametrize(
    ("model_id", "eos_token_id", "added_tokens", "expected"),
    [
        (
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            128009,
            {128001: _END_OF_TEXT, 128008: ("<|eom_id|>", True), 128009: _EOT},
            {128001, 128008, 128009},
        ),
        # Llama 3.0 has no <|eom_id|>: 128008 is a reserved token.
        (
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            128009,
            {
                128001: _END_OF_TEXT,
                128008: ("<|reserved_special_token_4|>", True),
                128009: _EOT,
            },
            {128001, 128009},
        ),
        # ChatML fine-tunes named after Llama 3.1 and Llama 3 (the token layout of two on
        # Hugging Face): their turns end with <|im_end|>, and the second gives 128008 to
        # <tools>.
        (
            "example/Llama-3.1-8B-ChatML-fine-tune",
            128040,
            {
                128001: _END_OF_TEXT,
                128008: ("<|eom_id|>", True),
                128009: _EOT,
                128040: _IM_END,
            },
            {128001, 128008, 128009, 128040},
        ),
        (
            "example/Llama-3-8B-ChatML-fine-tune",
            128003,
            {
                128001: _END_OF_TEXT,
                128003: _IM_END,
                128008: ("<tools>", False),
                128009: _EOT,
            },
            {128001, 128003, 128009},
        ),
    ],
)
def test_the_end_of_message_tokens_add_to_the_tokenizers_own(
    model_id: str,
    eos_token_id: int,
    added_tokens: dict[int, tuple[str, bool]],
    expected: set[int],
) -> None:
    # The tokenizers' added tokens at these ids, from their tokenizer_config.json.
    tokenizer = _LoadedTokenizer(eos_token_id, added_tokens)

    add_eos_tokens(
        tokenizer,  # pyright: ignore[reportArgumentType]
        added_eos_tokens_for_model_id(model_id),
    )

    assert tokenizer.eos_token_ids == expected


def _history(version: str, results_in_order: bool) -> str:
    calls = [
        {
            "id": f"call_{city}",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": city}),
            },
        }
        for city in ("Paris", "Rome")
    ]
    results = [
        {"role": "tool", "tool_call_id": f"call_{city}", "content": temperature}
        for city, temperature in (("Paris", "18C"), ("Rome", "25C"))
    ]
    return _request(
        version,
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": None, "tool_calls": calls},
                *(results if results_in_order else results[::-1]),
            ],
            "tools": [WEATHER_TOOL],
        },
    )


@pytest.mark.parametrize("version", TEMPLATES)
@pytest.mark.parametrize("results_in_order", [True, False])
def test_a_history_with_parallel_calls_renders_one_call_per_message(
    version: str, results_in_order: bool
) -> None:
    # The templates raise "This model only supports single tool-calls at once!" on an
    # assistant message with more than one call.
    prompt = _history(version, results_in_order)

    turns = prompt.split("<|start_header_id|>")[3:]
    assert turns == [
        'assistant<|end_header_id|>\n\n{"name": "get_weather", "parameters": '
        '{"city": "Paris"}}<|eot_id|>',
        'ipython<|end_header_id|>\n\n"18C"<|eot_id|>',
        'assistant<|end_header_id|>\n\n{"name": "get_weather", "parameters": '
        '{"city": "Rome"}}<|eot_id|>',
        'ipython<|end_header_id|>\n\n"25C"<|eot_id|>',
        "assistant<|end_header_id|>\n\n",
    ]
