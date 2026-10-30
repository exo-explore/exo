"""Qwen3.8 through exo's request path and the model's own chat template.

fixtures/qwen3_8_chat_template.jinja is chat_template.jinja from Qwen/Qwen3.8-27B
(revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), byte-identical to the one in every
mlx-community Qwen3.8-27B conversion. It accepts only "xhigh" (its default), "medium" and
"low" for `reasoning_effort` and raises on anything else while thinking is on.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, NoReturn

import anyio
import pytest
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionRequest
from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import ReasoningEffort
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    get_eos_token_ids_for_model,
)

QWEN3_8_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen3_8_chat_template.jinja"
).read_text()
MODEL_ID = ModelId("mlx-community/Qwen3.8-27B-4bit")
QUANTIZATIONS = ["4bit", "8bit", "bf16", "mxfp4", "nvfp4"]

XHIGH = "Reasoning effort is set to xhigh."
LOW = "Reasoning effort is set to low."
THINKING_ON = "<|im_start|>assistant\n<think>\n"
THINKING_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


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
    """Renders the real template in the Jinja environment transformers'
    apply_chat_template uses (sandboxed, trim/lstrip blocks, loop controls, its
    tojson and raise_exception)."""

    chat_template = QWEN3_8_TEMPLATE

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        **kwargs: bool | str,
    ) -> str:
        assert not tokenize
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
            **kwargs,
        )


def render(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "reasoning_effort": reasoning_effort,
            "enable_thinking": enable_thinking,
            "tools": tools,
        }
    )
    params = asyncio.run(chat_request_to_text_generation(request))
    return apply_chat_template(TemplateOnlyTokenizer(), params)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("reasoning_effort", "enable_thinking", "instruction", "thinking"),
    [
        # Nothing requested, or thinking switched on without a level: the template's default (xhigh).
        (None, None, XHIGH, True),
        (None, True, XHIGH, True),
        ("high", None, XHIGH, True),
        ("xhigh", None, XHIGH, True),
        ("high", True, XHIGH, True),
        ("medium", None, None, True),
        ("medium", True, None, True),
        ("low", None, LOW, True),
        ("minimal", None, LOW, True),
        # Thinking off: no effort reaches the template.
        ("none", None, None, False),
        (None, False, None, False),
        ("high", False, None, False),
        ("low", False, None, False),
        # Contradictory request: the explicit enable_thinking wins and the template default applies.
        ("none", True, XHIGH, True),
    ],
)
def test_every_reasoning_effort_renders(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
    instruction: str | None,
    thinking: bool,
) -> None:
    prompt = render(reasoning_effort, enable_thinking)

    assert prompt.endswith(THINKING_ON if thinking else THINKING_OFF)
    for candidate in (XHIGH, LOW):
        assert (candidate in prompt) == (candidate == instruction)
    if instruction is None:
        assert "Reasoning effort" not in prompt
        assert "<|im_start|>system" not in prompt


def test_reasoning_effort_with_tools() -> None:
    prompt = render("minimal", None, tools=[WEATHER_TOOL])

    system = prompt.split("<|im_end|>")[0]
    assert system.startswith("<|im_start|>system\n" + LOW)
    assert '"name": "get_weather"' in system
    assert prompt.endswith(THINKING_ON)


def test_qwen3_8_stops_on_im_end_and_endoftext() -> None:
    for quantization in QUANTIZATIONS:
        model_id = ModelId(f"mlx-community/Qwen3.8-27B-{quantization}")
        assert get_eos_token_ids_for_model(model_id) == [248046, 248044]


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_qwen3_8_card_sampling_defaults(quantization: str) -> None:
    path = (
        Path(RESOURCES_DIR)
        / "inference_model_cards"
        / f"mlx-community--Qwen3.8-27B-{quantization}.toml"
    )
    card = asyncio.run(ModelCard.load_from_path(anyio.Path(path)))
    defaults = card.sampling_defaults
    non_thinking = defaults.non_thinking

    assert card.model_id == f"mlx-community/Qwen3.8-27B-{quantization}"
    assert (card.n_layers, card.hidden_size, card.num_key_value_heads) == (64, 5120, 4)
    # Thinking is the model's default mode; Qwen recommends no presence penalty for it.
    assert (defaults.temperature, defaults.top_p, defaults.top_k) == (1.0, 0.95, 20)
    assert defaults.presence_penalty == 0.0
    assert non_thinking is not None
    assert (non_thinking.temperature, non_thinking.top_p, non_thinking.top_k) == (
        0.7,
        0.8,
        20,
    )
    assert non_thinking.presence_penalty == 1.5
