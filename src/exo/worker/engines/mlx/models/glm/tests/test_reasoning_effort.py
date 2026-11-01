"""GLM-5.2 / GLM-5.3 reasoning-effort mapping in render_chat_template.

The fake tokenizer renders the effort and generation-prompt lines of the vendor
templates verbatim, so the tests check what the model actually receives.
"""

from typing import Any, final

import jinja2
import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.utils_mlx import apply_chat_template

# From zai-org/GLM-5.3 chat_template.jinja (lines 2-3 and the generation prompt).
GLM53_TEMPLATE = (
    "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined"
    " and reasoning_effort in ['low', 'high'] else 'max' -%}"
    "{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: "
    "{{ effective_reasoning_effort | capitalize }}{%- endif -%}"
    "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' -}}{%- endif -%}"
)

# From zai-org/GLM-5.2 chat_template.jinja (lines 2-3 and the generation prompt): only
# "high" is understood and enable_thinking=false closes the reasoning block.
GLM52_TEMPLATE = (
    "{%- set effective_reasoning_effort = 'high' if reasoning_effort is defined"
    " and reasoning_effort == 'high' else 'max' -%}"
    "{%- if (enable_thinking is not defined or enable_thinking) and"
    " effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: "
    "{{ effective_reasoning_effort | capitalize }}{%- endif -%}"
    "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{{- '<think></think>' if (enable_thinking is defined and not enable_thinking)"
    " else '<think>' -}}{%- endif -%}"
)


@final
class FakeTokenizer:
    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.has_thinking = True
        self.think_start = "<think>"
        self.think_end = "</think>"
        # exo renders requests that leave thinking unset with the wrapped Hugging Face
        # tokenizer directly; this fake is both.
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
        environment = jinja2.Environment()
        return environment.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )


def _render(
    model: str,
    template: str,
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
) -> str:
    params = TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    return apply_chat_template(FakeTokenizer(template), params)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("effort", "thinking", "expected_effort"),
    [
        (None, None, "Max"),
        ("xhigh", True, "Max"),
        ("high", True, "High"),
        ("medium", True, "High"),  # exo's default for enable_thinking=True
        ("low", True, "Low"),
        ("minimal", True, "Low"),
    ],
)
def test_glm53_effort_mapping(
    effort: ReasoningEffort | None, thinking: bool | None, expected_effort: str
) -> None:
    prompt = _render("mlx-community/GLM-5.3-4bit", GLM53_TEMPLATE, effort, thinking)
    assert prompt.startswith(f"<|system|>Reasoning Effort: {expected_effort}")
    assert prompt.endswith("<|assistant|><think>")


def test_glm53_flash_uses_the_same_mapping() -> None:
    prompt = _render("mlx-community/GLM-5.3-Flash-4bit", GLM53_TEMPLATE, "medium", True)
    assert prompt.startswith("<|system|>Reasoning Effort: High")


def test_glm53_thinking_off_closes_the_reasoning_block() -> None:
    prompt = _render("mlx-community/GLM-5.3-4bit", GLM53_TEMPLATE, "none", False)
    assert prompt.startswith("<|system|>Reasoning Effort: Low")
    assert prompt.endswith("<|assistant|><think></think>")


@pytest.mark.parametrize(
    ("effort", "expected_effort"),
    [
        (None, "Max"),
        ("xhigh", "Max"),
        ("high", "High"),
        ("medium", "High"),
        ("low", "High"),
    ],
)
def test_glm52_effort_mapping(
    effort: ReasoningEffort | None, expected_effort: str
) -> None:
    prompt = _render("mlx-community/GLM-5.2-4bit", GLM52_TEMPLATE, effort, True)
    assert prompt.startswith(f"<|system|>Reasoning Effort: {expected_effort}")
    assert prompt.endswith("<|assistant|><think>")


def test_glm52_thinking_off_uses_the_template_switch() -> None:
    prompt = _render("mlx-community/GLM-5.2-4bit", GLM52_TEMPLATE, "none", False)
    assert "Reasoning Effort" not in prompt
    assert prompt.endswith("<|assistant|><think></think>")


def test_other_models_receive_the_effort_unchanged() -> None:
    template = "{{ reasoning_effort }}"
    prompt = _render("mlx-community/GLM-5.1-MXFP4-Q8", template, "medium", True)
    assert prompt == "medium"
