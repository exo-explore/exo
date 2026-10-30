"""DeepSeek V4.1 prompts: the vendored encoder against DeepSeek's golden prompts, and exo's
prompt encoder (reasoning effort, tools, thinking) on top of it."""

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
from exo.worker.engines.mlx.models.deepseek_v41 import encoding, prompt

FIXTURES = Path(__file__).parent / "fixtures"
TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize("case", range(1, 6))
def test_matches_deepseeks_golden_prompts(case: int) -> None:
    cases = encoding.load_cases(str(FIXTURES / f"test_input_{case}.json"))
    rendered = ""
    for c in cases:
        messages = cast(list[dict[str, Any]], c["messages"])
        mode = cast(str | None, c.get("thinking_mode")) or "chat"
        context = cast(list[dict[str, Any]] | None, c.get("context"))
        effort = cast(str | int | None, c.get("reasoning_effort"))
        rendered += cast(
            str,
            encoding.encode_messages(
                messages, thinking_mode=mode, context=context, reasoning_effort=effort
            ),
        )
    assert rendered == (FIXTURES / f"test_output_{case}.txt").read_text()


def _task(
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("deepseek-ai/DeepSeek-V4.1-Flash"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )


_USER = [{"role": "user", "content": "What is 2+2?"}]


def test_chat_mode_closes_thinking_right_away() -> None:
    rendered = prompt.encode_prompt(_USER, _task(enable_thinking=False))
    assert rendered == (
        "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>"
    )


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("medium", 75),
        ("minimal", 50),
        ("low", 50),
        ("high", 75),
        ("xhigh", 100),
    ],
)
def test_thinking_mode_renders_the_numeric_reasoning_effort(
    effort: ReasoningEffort | None, budget: int
) -> None:
    rendered = prompt.encode_prompt(_USER, _task(reasoning_effort=effort))
    assert rendered == (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: "
        f"{budget} (range 1-100, the higher the value, the more thorough the reasoning)"
        "\n\n<｜User｜>What is 2+2?<｜Assistant｜><think>"
    )


def test_tools_go_on_a_new_empty_system_message() -> None:
    rendered = prompt.encode_prompt(_USER, _task(enable_thinking=False, tools=[TOOL]))
    assert rendered.startswith("<｜begin▁of▁sentence｜><｜System｜>\n\n## Tools\n")
    assert '"name": "get_weather"' in rendered
    assert "<｜DSML｜ calls>" in rendered
    assert rendered.endswith("<｜User｜>What is 2+2?<｜Assistant｜></think>")


def test_tools_join_the_leading_system_message() -> None:
    messages = [{"role": "system", "content": "Be brief."}, *_USER]
    rendered = prompt.encode_prompt(
        messages, _task(enable_thinking=False, tools=[TOOL])
    )
    assert rendered.startswith(
        "<｜begin▁of▁sentence｜><｜System｜>Be brief.\n\n## Tools\n"
    )


def test_history_tool_calls_and_results_render_as_dsml() -> None:
    messages: list[dict[str, Any]] = [
        *_USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Rome"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"temp_c": 17}),
        },
    ]
    rendered = prompt.encode_prompt(
        messages, _task(enable_thinking=False, tools=[TOOL])
    )
    assert (
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">Rome</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>"
    ) in rendered
    assert rendered.endswith(
        '<｜User｜><tool_result>{"temp_c": 17}</tool_result><｜Assistant｜></think>'
    )


def test_think_tags_in_earlier_assistant_turns_are_dropped() -> None:
    messages = [
        *_USER,
        {"role": "assistant", "content": "<think>scratch</think>4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    rendered = prompt.encode_prompt(messages, _task(enable_thinking=False))
    assert "scratch" not in rendered
    assert "<｜Assistant｜></think>4<｜end▁of▁sentence｜><｜User｜>And 3+3?" in rendered
