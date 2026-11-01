"""DeepSeek V4 prompts: the vendored release encoder against DeepSeek's golden prompts, and
exo's prompt encoder (reasoning effort for release and preview builds, exact think tags)
on top of it."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    ReasoningEffort,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.deepseek_v4 import encoding, prompt

FIXTURES = Path(__file__).parent / "fixtures"
_HIGH = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
_MAX = "Reasoning Effort: Beyond maximum"


def _golden(case: int) -> tuple[list[dict[str, Any]], str]:
    loaded = cast(
        object, json.loads((FIXTURES / f"test_input_{case}.json").read_text())
    )
    if isinstance(loaded, dict):
        data = cast(dict[str, Any], loaded)
        messages = cast(list[dict[str, Any]], data["messages"])
        messages[0]["tools"] = data["tools"]
    else:
        messages = cast(list[dict[str, Any]], loaded)
    return messages, (FIXTURES / f"test_output_{case}.txt").read_text()


@pytest.mark.parametrize(
    ("case", "thinking_mode"),
    [(1, "thinking"), (2, "thinking"), (3, "thinking"), (4, "chat")],
)
def test_vendored_encoder_matches_the_official_goldens(
    case: int, thinking_mode: str
) -> None:
    """DeepSeek's own tests (encoding/test_encoding_dsv4.py in the release repos)."""
    messages, gold = _golden(case)
    rendered = encoding.encode_messages(messages, thinking_mode=thinking_mode)
    assert rendered == gold


@pytest.mark.parametrize(
    ("effort", "prefix"),
    [(None, ""), ("low", ""), ("high", _HIGH), ("max", _MAX)],
)
def test_release_reasoning_effort_prefix(effort: str | None, prefix: str) -> None:
    rendered = encoding.encode_messages(
        [{"role": "user", "content": "What is 2 + 2?"}],
        thinking_mode="thinking",
        reasoning_effort=effort,
    )
    body = rendered.removeprefix(encoding.bos_token)
    assert body.startswith(prefix)
    if not prefix:
        assert not body.startswith("Reasoning Effort")


def test_reasoning_effort_prefix_only_in_thinking_mode() -> None:
    rendered = encoding.encode_messages(
        [{"role": "user", "content": "hi"}],
        thinking_mode="chat",
        reasoning_effort="max",
    )
    assert "Reasoning Effort" not in rendered


def _task(
    model: str,
    effort: ReasoningEffort | None = None,
    enable_thinking: bool | None = None,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        reasoning_effort=effort,
        enable_thinking=enable_thinking,
    )


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        # Release builds: low (default), high and max.
        ("deepseek-ai/DeepSeek-V4-Flash-0731", None, None),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "minimal", None),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "medium", None),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "high", "high"),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "xhigh", "max"),
        ("deepseek-ai/DeepSeek-V4-Pro-0813", "high", "high"),
        ("deepseek-ai/DeepSeek-V4-Pro-0813", "xhigh", "max"),
        # Preview builds know one effort prompt (the release encoder's "high" text),
        # which exo's top effort turns on.
        ("mlx-community/DeepSeek-V4-Flash", "high", None),
        ("mlx-community/DeepSeek-V4-Flash", "xhigh", "high"),
        ("mlx-community/DeepSeek-V4-Pro", "xhigh", "high"),
    ],
)
def test_reasoning_effort_mapping(
    model: str, effort: ReasoningEffort | None, expected: str | None
) -> None:
    assert (
        prompt._v4_reasoning_effort(_task(model, effort))  # pyright: ignore[reportPrivateUsage]
        == expected
    )


def test_release_prompt_renders_the_max_effort_for_xhigh() -> None:
    rendered = prompt.encode_prompt(
        [{"role": "user", "content": "hi"}],
        _task("deepseek-ai/DeepSeek-V4-Flash-0731", "xhigh"),
    )
    assert rendered.removeprefix(encoding.bos_token).startswith(_MAX)


def test_chat_mode_ends_with_a_bare_think_end_that_exo_keeps() -> None:
    """The official encoder closes thinking with a bare `</think>` in chat mode; exo must
    not put a `<think>` before it."""
    task = _task("deepseek-ai/DeepSeek-V4-Flash-0731", enable_thinking=False)
    rendered = prompt.encode_prompt([{"role": "user", "content": "hi"}], task)
    assert rendered.endswith("<｜Assistant｜></think>")
    assert registry.keeps_think_tags(task, "deepseek_v4")
    assert registry.model_type_keeps_think_tags("deepseek_v4")


@pytest.mark.parametrize("model_type", [None, "llama", "deepseek_v32"])
def test_other_model_types_get_their_think_tags_repaired(
    model_type: str | None,
) -> None:
    assert not registry.model_type_keeps_think_tags(model_type)
