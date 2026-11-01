"""Hy3's output format and reasoning switch: tool calls as the official chat template
renders them, the reasoning tags, and the reasoning_effort mapping."""

import json
from pathlib import Path
from typing import Any, cast

import jinja2
import pytest
from mlx_lm.tokenizer_utils import load as mlx_lm_load_tokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.models.hy_v3.model_ids import is_hy3_model_id
from exo.worker.engines.mlx.models.hy_v3.reasoning import template_effort
from exo.worker.engines.mlx.models.hy_v3.tokenizer import (
    HyV3TokenizerWrapper,
    load_tokenizer,
    parse_tool_calls,
)
from exo.worker.engines.mlx.models.registry import (
    reasoning_effort_hook_for,
    tokenizer_loader_for,
)
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

# The tool-call part of tencent/Hy3's chat_template.jinja (the assistant turn's
# tool_calls loop), with its token variables filled in.
_TOOL_CALL_TEMPLATE = (
    "<tool_calls:opensource>\n"
    "{%- for tool in tool_calls %}"
    "{%- set arguments = tool['function']['arguments'] %}"
    "{{- '<tool_call:opensource>' + tool['function']['name'] + '<tool_sep:opensource>' + '\\n' }}"
    "{%- for key, value in arguments.items() %}"
    "{{- '<arg_key:opensource>' + key + '</arg_key:opensource>' + '\\n' }}"
    "{%- if value is not string %}{%- set value = value | tojson(ensure_ascii=False) %}{%- endif %}"
    "{{- '<arg_value:opensource>' + value + '</arg_value:opensource>' + '\\n' }}"
    "{%- endfor %}"
    "{{- '</tool_call:opensource>' + '\\n' }}"
    "{%- endfor %}"
    "</tool_calls:opensource>"
)

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "integer"},
                    "options": {"type": "object"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]

_CALLS: list[dict[str, Any]] = [
    {
        "function": {
            "name": "write_file",
            "arguments": {
                "path": "package.json",
                "content": '{"name": "demo", "version": "1.0.0"}',
                "mode": 420,
                "options": {"overwrite": True, "tags": ["a", "ü"]},
            },
        }
    },
    {"function": {"name": "get_weather", "arguments": {"city": "2"}}},
]


def _tojson(value: object, ensure_ascii: bool = False) -> str:
    # transformers' chat-template tojson filter, which takes ensure_ascii
    return json.dumps(value, ensure_ascii=ensure_ascii)


def _rendered_calls() -> str:
    environment = jinja2.Environment()
    environment.filters["tojson"] = _tojson
    return environment.from_string(_TOOL_CALL_TEMPLATE).render(tool_calls=_CALLS)


def test_tool_calls_round_trip_with_their_types() -> None:
    parser = make_mlx_parser(
        "<tool_calls:opensource>", "</tool_calls:opensource>", parse_tool_calls
    )
    parsed = parser.parse(_rendered_calls(), _TOOLS)
    assert parsed is not None
    assert [(call.name, json.loads(call.arguments)) for call in parsed] == [
        (c["function"]["name"], c["function"]["arguments"]) for c in _CALLS
    ]


def test_without_tools_values_decode_as_json() -> None:
    calls = parse_tool_calls(_rendered_calls(), None)
    assert calls[1]["arguments"] == {"city": 2}
    assert calls[0]["arguments"]["content"] == {"name": "demo", "version": "1.0.0"}


def test_a_call_without_arguments_and_preview_tags() -> None:
    text = "\n<tool_call>list_files<tool_sep>\n</tool_call>\n"
    assert parse_tool_calls(text, None) == [{"name": "list_files", "arguments": {}}]


def test_text_without_a_call_is_not_a_call() -> None:
    with pytest.raises(ValueError):
        parse_tool_calls("I would call a tool here.", None)


def _tokenizer(pieces: list[str]) -> PreTrainedTokenizerFast:
    vocab = {piece: i for i, piece in enumerate(["<unk>", "a", "b", *pieces])}
    backend = Tokenizer(models.WordLevel(vocab, "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def test_wrapper_knows_the_opensource_tags() -> None:
    tokens = [
        "<think:opensource>",
        "</think:opensource>",
        "<tool_calls:opensource>",
        "</tool_calls:opensource>",
    ]
    wrapper = HyV3TokenizerWrapper(_tokenizer(tokens), None)
    assert wrapper.has_thinking
    assert (wrapper.think_start, wrapper.think_end) == tuple(tokens[:2])
    assert wrapper.think_start_tokens == (3,)
    assert wrapper.think_end_tokens == (4,)
    assert (wrapper.tool_call_start, wrapper.tool_call_end) == tuple(tokens[2:])


def test_wrapper_without_the_tags_keeps_mlx_lms_inference() -> None:
    wrapper = HyV3TokenizerWrapper(
        _tokenizer(["<think>", "</think>", "<tool_calls>", "</tool_calls>"]), None
    )
    assert (wrapper.think_start, wrapper.think_end) == ("<think>", "</think>")
    assert wrapper.tool_call_start == "<tool_calls>"


def test_conversions_naming_an_unknown_tool_parser_load(tmp_path: Path) -> None:
    """The MLX conversions' tokenizer_config.json names tool_parser_type
    "hy_v3_opensource", a module only the unmerged upstream port had. mlx-lm's loader
    imports it and fails; the family's loader doesn't use it."""
    _tokenizer(
        [
            "<think:opensource>",
            "</think:opensource>",
            "<tool_calls:opensource>",
            "</tool_calls:opensource>",
        ]
    ).save_pretrained(tmp_path)  # pyright: ignore[reportUnknownMemberType]
    config_path = tmp_path / "tokenizer_config.json"
    config = cast(dict[str, object], json.loads(config_path.read_text()))
    config["tool_parser_type"] = "hy_v3_opensource"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ModuleNotFoundError):
        mlx_lm_load_tokenizer(tmp_path)
    wrapper = load_tokenizer(tmp_path, None)
    assert wrapper.has_thinking
    assert wrapper.tool_call_start == "<tool_calls:opensource>"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("tencent/Hy3", True),
        ("tencent/Hy3-FP8", True),
        ("XavierLocalAI/Hy3-4bit", True),
        ("mlx-community/Hy3-oQ2e", True),
        ("kernelpool/Hy3-8bit", True),
        ("tencent/Hunyuan3D-2", False),
        ("calcuis/hy3d-gguf", False),
        ("mlx-community/Qwen3.8-27B-4bit", False),
    ],
)
def test_hy3_model_ids(model_id: str, expected: bool) -> None:
    assert is_hy3_model_id(model_id) is expected
    assert (tokenizer_loader_for(ModelId(model_id)) is not None) is expected


def _task(
    effort: ReasoningEffort | None,
    model: str = "XavierLocalAI/Hy3-4bit",
    enable_thinking: bool | None = None,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=effort,
        enable_thinking=enable_thinking,
    )


# Hy3's template accepts exactly these and raises on anything else.
_TEMPLATE_EFFORTS = {"no_think", "low", "high"}


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (None, None),
        ("none", "no_think"),
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "high"),
    ],
)
def test_reasoning_effort_mapping(
    effort: ReasoningEffort | None, expected: str | None
) -> None:
    task = _task(effort)
    hook = reasoning_effort_hook_for(task)
    assert hook is not None
    mapped = template_effort(task)
    assert mapped == expected
    assert mapped is None or mapped in _TEMPLATE_EFFORTS
    assert hook.thinking_off_suffix(task) is None


def test_other_models_keep_their_reasoning_effort() -> None:
    assert (
        reasoning_effort_hook_for(_task("medium", "mlx-community/GLM-4.7-4bit")) is None
    )


# tencent/Hy3's chat_template.jinja: its reasoning_effort check, the effort marker it adds
# to the system prompt (without tools), the user turn and the generation prompt.
_EFFORT_TEMPLATE = (
    "{%- if not reasoning_effort is defined %}{%- set reasoning_effort = 'no_think' %}"
    "{%- elif reasoning_effort not in ['high', 'low', 'no_think'] %}"
    "{{- raise_exception('reasoning_effort error : ' + reasoning_effort) }}{%- endif %}"
    "{{- '<bos>' }}"
    "{%- if reasoning_effort != '' and not tools %}"
    "{{- '<｜reasoning_mode:opensource｜>reasoning_effort:' + reasoning_effort }}{%- endif %}"
    "{%- for m in messages %}{{- '<｜hy_User:opensource｜>' + m.content }}{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{%- if reasoning_effort in ['low', 'high'] %}"
    "{{- '<｜hy_Assistant:opensource｜><think:opensource>' }}"
    "{%- else %}{{- '<｜hy_Assistant:opensource｜><think:opensource></think:opensource>' }}"
    "{%- endif %}{%- endif %}"
)


class _FakeTokenizer:
    """Renders _EFFORT_TEMPLATE; exo renders requests that leave thinking unset with the
    wrapped Hugging Face tokenizer directly, so this fake is both."""

    def __init__(self) -> None:
        self.chat_template = _EFFORT_TEMPLATE
        self.has_thinking = True
        self.think_start = "<think:opensource>"
        self.think_end = "</think:opensource>"
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
        kwargs.pop("return_dict", None)

        def raise_exception(message: str) -> None:
            raise jinja2.TemplateError(message)

        return (
            jinja2.Environment()
            .from_string(self.chat_template)
            .render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                raise_exception=raise_exception,
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    ("effort", "thinking", "marker", "ending"),
    [
        (None, None, "no_think", "<think:opensource></think:opensource>"),
        (None, False, "no_think", "<think:opensource></think:opensource>"),
        (None, True, "high", "<think:opensource>"),
        ("low", None, "low", "<think:opensource>"),
        ("xhigh", None, "high", "<think:opensource>"),
    ],
)
def test_requests_render_with_an_effort_the_template_accepts(
    effort: ReasoningEffort | None, thinking: bool | None, marker: str, ending: str
) -> None:
    prompt = apply_chat_template(
        _FakeTokenizer(),  # pyright: ignore[reportArgumentType]
        _task(effort, enable_thinking=thinking),
    )
    assert prompt.startswith(
        f"<bos><｜reasoning_mode:opensource｜>reasoning_effort:{marker}"
    )
    assert prompt.endswith("<｜hy_Assistant:opensource｜>" + ending)


def test_without_the_mapping_exos_levels_do_not_render() -> None:
    with pytest.raises(jinja2.TemplateError):
        _FakeTokenizer().apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium",
        )
