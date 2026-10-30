"""Qwen3.8 reasoning_effort mapping in render_chat_template, on the real chat template
(Qwen/Qwen3.8-Flash-Next's, byte-identical to Qwen/Qwen3.8-27B's and to the mlx-community
conversions')."""

from pathlib import Path
from typing import Any, NoReturn, final

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

TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen3_8_chat_template.jinja"
).read_text()
XHIGH = "Reasoning effort is set to xhigh."
LOW = "Reasoning effort is set to low."


def _raise(message: str) -> NoReturn:
    raise ValueError(message)


@final
class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = TEMPLATE
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
        kwargs.pop("return_dict", None)
        environment = jinja2.Environment()
        environment.globals["raise_exception"] = _raise  # pyright: ignore[reportArgumentType]
        return environment.from_string(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
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
    return apply_chat_template(FakeTokenizer(), params)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("effort", "instruction"),
    [
        (None, XHIGH),
        ("xhigh", XHIGH),
        ("high", XHIGH),
        ("medium", None),
        ("low", LOW),
        ("minimal", LOW),
    ],
)
@pytest.mark.parametrize(
    "model", ["mlx-community/Qwen3.8-Flash-Next-4bit", "mlx-community/Qwen3.8-27B-4bit"]
)
def test_every_effort_renders(
    model: str, effort: ReasoningEffort | None, instruction: str | None
) -> None:
    prompt = _render(model, effort, True if effort is not None else None)
    if instruction is None:
        assert "Reasoning effort is set" not in prompt
    else:
        assert instruction in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_thinking_off_ignores_the_effort() -> None:
    prompt = _render("mlx-community/Qwen3.8-Flash-Next-4bit", "none", False)
    assert "Reasoning effort is set" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_the_template_rejects_unmapped_efforts() -> None:
    """Without the mapping, "high" reaches the template and raises (the bug it fixes)."""
    with pytest.raises(ValueError, match="Unexpected reasoning effort high"):
        _render("some-org/other-model-using-this-template", "high", True)
