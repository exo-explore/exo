import json
from collections.abc import Generator
from typing import Any

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.dsml_encoding import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    DSML_TOKEN,
    EOS_TOKEN,
    THINKING_END,
    THINKING_START,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
    USER_TOKEN,
    encode_messages,
    parse_dsml_output,
)
from exo.worker.runner.llm_inference.runner import parse_deepseek_v32

# ── Shared fixtures ──────────────────────────────────────────────

_WEATHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"},
                },
                "required": ["timezone"],
            },
        },
    },
]


def _simulate_tokens(
    texts: list[str],
    finish_on_last: bool = True,
) -> Generator[GenerationResponse]:
    """Simulate a model producing tokens from a list of text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if (is_last and finish_on_last) else None,
            usage=None,
        )


# ── Test: Standard text response (no tool calls) ────────────────


class TestE2EStandardResponse:
    """Model generates a plain text response — no tool calling involved."""

    def test_plain_text_passthrough(self):
        """Simulate model producing: 'The weather in NYC is 72°F and sunny.'"""
        # Step 1: Encode the prompt (with tools available)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Verify prompt structure
        assert BOS_TOKEN in prompt
        assert "## Tools" in prompt
        assert "get_weather" in prompt
        assert f"{USER_TOKEN}What's the weather in NYC?{ASSISTANT_TOKEN}" in prompt

        # Step 2: Simulate model response — plain text tokens (no DSML)
        model_tokens = [
            "The weather",
            " in NYC",
            " is 72",
            "°F",
            " and sunny",
            ".",
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        # Step 3: Verify all tokens pass through as GenerationResponse
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        assert len(gen_results) == 6
        full_text = "".join(r.text for r in gen_results)
        assert full_text == "The weather in NYC is 72°F and sunny."
        assert gen_results[-1].finish_reason == "stop"


# ── Test: Tool call response ─────────────────────────────────────


class TestE2EToolCallResponse:
    """Model generates a DSML tool call — realistic token boundaries."""

    def test_realistic_tool_call_tokens(self):
        """Simulate model generating a get_weather tool call with realistic token splits.

        Real models split DSML markers across tokens unpredictably.
        This simulates how DeepSeek V3.2 actually tokenizes DSML output.
        """
        # Step 1: Encode prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in San Francisco?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)
        assert "get_weather" in prompt

        # Step 2: Simulate realistic token-by-token model output
        # The model first produces some text, then a DSML tool call block
        model_tokens = [
            "I'll check the weather for you.",
            "\n\n",
            f"<{DSML_TOKEN}",  # marker split across tokens
            "function_calls>\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">',
            "San Francisco",
            f"</{DSML_TOKEN}parameter>\n",
            f'<{DSML_TOKEN}parameter name="units" string="false">',
            '"celsius"',
            f"</{DSML_TOKEN}parameter>\n",
            f"</{DSML_TOKEN}invoke>\n",
            f"</{DSML_TOKEN}function_calls>",
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        # Step 3: Verify
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        # Should have text tokens before tool call + one ToolCallResponse
        assert len(tool_results) == 1
        assert len(tool_results[0].tool_calls) == 1

        tc = tool_results[0].tool_calls[0]
        assert tc.name == "get_weather"
        args = json.loads(tc.arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "San Francisco"
        assert args["units"] == "celsius"

        # The text before the tool call should still be yielded
        text_before = "".join(r.text for r in gen_results if not r.is_thinking)
        assert "check the weather" in text_before

    def test_multiple_tool_calls_in_one_block(self):
        """Model generates two tool calls in a single function_calls block."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in NYC and time in EST?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)
        assert "get_weather" in prompt
        assert "get_time" in prompt

        # Simulate model output with two invocations
        model_tokens = [
            "Let me check both.\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">EST</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert len(tool_results[0].tool_calls) == 2
        assert tool_results[0].tool_calls[0].name == "get_weather"
        assert tool_results[0].tool_calls[1].name == "get_time"

        args0 = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        args1 = json.loads(tool_results[0].tool_calls[1].arguments)  # pyright: ignore[reportAny]
        assert args0 == {"city": "NYC"}
        assert args1 == {"timezone": "EST"}


# ── Test: Multi-turn tool use flow ───────────────────────────────


class TestE2EMultiTurnToolUse:
    """Full multi-turn: user asks → model calls tool → tool result → model answers."""

    def test_encode_multi_turn_with_tool_results(self):
        """Verify the prompt for turn 2 (after tool results) is correctly encoded."""
        # Turn 1: user asks, model calls tool
        # Turn 2: tool result provided, model answers
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 72, "condition": "sunny"}'},
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Verify multi-turn structure
        assert BOS_TOKEN in prompt
        assert "You are a weather assistant." in prompt
        assert "## Tools" in prompt

        # The assistant's tool call should be encoded as DSML
        assert TOOL_CALLS_START in prompt
        assert f'<{DSML_TOKEN}invoke name="get_weather">' in prompt
        assert EOS_TOKEN in prompt

        # The tool result should be wrapped in function_results
        assert "<function_results>" in prompt
        assert "<result>" in prompt
        assert "72" in prompt
        assert "</function_results>" in prompt

        # Now simulate model answering after seeing the tool result
        model_tokens = [
            "The current",
            " weather in NYC",
            " is 72°F",
            " and sunny.",
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        full_text = "".join(r.text for r in gen_results)
        assert full_text == "The current weather in NYC is 72°F and sunny."

    def test_multi_tool_results_encoding(self):
        """Verify encoding when model called two tools and both return results."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "LA"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "PST"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": "85F, clear skies"},
            {"role": "tool", "content": "3:42 PM PST"},
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Should have one function_results block with two results
        assert prompt.count("<function_results>") == 1
        assert prompt.count("</function_results>") == 1
        assert "<result>85F, clear skies</result>" in prompt
        assert "<result>3:42 PM PST</result>" in prompt


# ── Test: Thinking + tool call ───────────────────────────────────


class TestE2EThinkingAndToolCall:
    """Model uses thinking mode, reasons, then makes a tool call."""

    def test_thinking_then_tool_call(self):
        """Model thinks first, then produces a DSML tool call block."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
        ]
        prompt = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        # Thinking mode: prompt should end with <think>
        assert prompt.endswith(THINKING_START)

        # Simulate: model outputs <think>, thinks, closes thinking, then tool call.
        # In the full pipeline, parse_thinking_models handles the case where
        # <think> is in the prompt. Here we test parse_deepseek_v32 directly,
        # which detects <think>/<think> markers in the stream.
        model_tokens = [
            THINKING_START,
            "The user wants weather",
            " information. I should use",
            " the get_weather tool.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">',
            "San Francisco",
            f"</{DSML_TOKEN}parameter>\n",
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        # Should have thinking tokens + tool call
        thinking_results = [r for r in gen_results if r.is_thinking]

        assert len(thinking_results) >= 1
        thinking_text = "".join(r.text for r in thinking_results)
        assert "get_weather tool" in thinking_text

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"
        args = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "San Francisco"

    def test_thinking_prompt_encoding(self):
        """Verify thinking mode affects prompt encoding correctly."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be thorough."},
            {"role": "user", "content": "What's the weather?"},
        ]

        # With thinking enabled
        prompt_think = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_think.endswith(THINKING_START)

        # With thinking disabled
        prompt_no_think = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="chat"
        )
        assert prompt_no_think.endswith(THINKING_END)

        # Both should have the same tool definitions
        assert "get_weather" in prompt_think
        assert "get_weather" in prompt_no_think


# ── Test: Round-trip encode → parse ──────────────────────────────


class TestE2ERoundTrip:
    """Verify that DSML we encode can be parsed back correctly."""

    def test_encoded_tool_call_is_parseable(self):
        """Encode an assistant tool call message, then parse the DSML output."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo", "units": "celsius"}',
                        },
                    }
                ],
            },
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Extract the DSML function_calls block from the prompt
        start = prompt.index(TOOL_CALLS_START)
        end = prompt.index(TOOL_CALLS_END) + len(TOOL_CALLS_END)
        dsml_block = prompt[start:end]

        # Parse it back
        parsed = parse_dsml_output(dsml_block)
        assert parsed is not None
        assert len(parsed) == 1
        assert parsed[0].name == "get_weather"
        args = json.loads(parsed[0].arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "Tokyo"
        assert args["units"] == "celsius"

    def test_encoded_multi_tool_call_round_trips(self):
        """Encode multiple tool calls, verify they parse back correctly."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Both please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "CET"}',
                        },
                    },
                ],
            },
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        start = prompt.index(TOOL_CALLS_START)
        end = prompt.index(TOOL_CALLS_END) + len(TOOL_CALLS_END)
        dsml_block = prompt[start:end]

        parsed = parse_dsml_output(dsml_block)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0].name == "get_weather"
        assert parsed[1].name == "get_time"
        assert json.loads(parsed[0].arguments) == {"city": "Paris"}
        assert json.loads(parsed[1].arguments) == {"timezone": "CET"}


# ── Test: Edge cases with realistic token boundaries ─────────────


class TestE2EEdgeCases:
    """Edge cases that occur in real model inference."""

    def test_dsml_marker_split_at_fullwidth_pipe(self):
        """The fullwidth pipe character ｜ might be its own token."""
        # This is a realistic tokenization: the DSML marker is split at the ｜ chars
        model_tokens = [
            "Let me help.\n\n",
            "<\uff5c",  # start of ｜DSML｜
            "DSML\uff5c",  # rest of DSML token
            "function_calls>\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_tool_call_with_nested_json_object(self):
        """Model passes a complex JSON object as a non-string parameter."""
        dsml_block = (
            f"{TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="create_event">\n'
            f'<{DSML_TOKEN}parameter name="title" string="true">Team Standup</{DSML_TOKEN}parameter>\n'
            f'<{DSML_TOKEN}parameter name="config" string="false">'
            f'{{"recurring": true, "days": ["mon", "wed", "fri"], "time": "09:00"}}'
            f"</{DSML_TOKEN}parameter>\n"
            f"</{DSML_TOKEN}invoke>\n"
            f"{TOOL_CALLS_END}"
        )

        # Feed as single token (model might produce it all at once after prefill)
        results = list(parse_deepseek_v32(_simulate_tokens([dsml_block])))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        tc = tool_results[0].tool_calls[0]
        assert tc.name == "create_event"
        args = json.loads(tc.arguments)  # pyright: ignore[reportAny]
        assert args["title"] == "Team Standup"
        assert args["config"]["recurring"] is True
        assert args["config"]["days"] == ["mon", "wed", "fri"]

    def test_text_with_angle_brackets_not_mistaken_for_dsml(self):
        """Angle brackets in normal text should not trigger DSML buffering."""
        model_tokens = [
            "The formula is ",
            "<x, y>",
            " where x > 0",
            " and y < 100.",
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        full_text = "".join(r.text for r in gen_results)
        assert "formula" in full_text
        assert "<x, y>" in full_text

    def test_empty_model_response(self):
        """Model produces only EOS (empty response)."""
        model_tokens = [""]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        assert len(gen_results) == 1
        assert gen_results[0].text == ""
        assert gen_results[0].finish_reason == "stop"


# ── Test: Full EPDP spec round-trip ──────────────────────────────


class TestE2EFullRoundTrip:
    """Full round-trip matching the vLLM EPDP spec.

    Simulates the complete multi-turn flow:
      Turn 1: user asks → think → tool call → tool result → think → answer
      Turn 2: user asks again → old reasoning stripped → think → answer
    """

    def test_single_tool_full_flow_with_thinking(self):
        """Complete flow: user → think → tool call → tool result → think → answer.

        This is the core EPDP flow from the vLLM spec.
        """
        # ── Turn 1.1: User asks, encode prompt ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "How's the weather in Hangzhou?"},
        ]
        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)
        assert "## Tools" in prompt_1
        assert "get_weather" in prompt_1

        # ── Turn 1.1: Model thinks, then calls tool ──
        model_tokens_1 = [
            THINKING_START,
            "The user wants to know the weather in Hangzhou.",
            " I need to use the get_weather tool.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Hangzhou</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_1)))

        # Verify: thinking tokens + tool call
        gen_1 = [r for r in results_1 if isinstance(r, GenerationResponse)]
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]
        thinking_1 = [r for r in gen_1 if r.is_thinking]

        assert len(thinking_1) >= 1
        assert "get_weather tool" in "".join(r.text for r in thinking_1)
        assert len(tool_1) == 1
        assert tool_1[0].tool_calls[0].name == "get_weather"
        tc_args = json.loads(tool_1[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert tc_args == {"city": "Hangzhou"}

        # ── Turn 1.2: Add assistant response + tool result to messages ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The user wants to know the weather in Hangzhou. I need to use the get_weather tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": '{"temperature": "7~13°C", "condition": "Cloudy"}',
            }
        )

        # Encode prompt for turn 1.2
        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Verify: prompt has the full conversation structure
        assert TOOL_CALLS_START in prompt_2  # assistant's encoded tool call
        assert EOS_TOKEN in prompt_2  # assistant turn ends with EOS
        assert "<function_results>" in prompt_2
        assert "<result>" in prompt_2
        assert "Cloudy" in prompt_2
        assert "</function_results>" in prompt_2
        # After tool results with thinking enabled → <think> appended
        assert prompt_2.endswith(THINKING_START)
        # The assistant's reasoning_content should appear (it's after last_user_idx)
        assert "get_weather tool" in prompt_2

        # ── Turn 1.2: Model thinks about results, then answers ──
        model_tokens_2 = [
            THINKING_START,
            "The weather in Hangzhou is Cloudy, 7~13°C.",
            " I'll tell the user.",
            THINKING_END,
            "The weather in Hangzhou is currently cloudy with temperatures between 7°C and 13°C.",
        ]
        results_2 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_2)))

        gen_2 = [r for r in results_2 if isinstance(r, GenerationResponse)]
        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        thinking_2 = [r for r in gen_2 if r.is_thinking]
        non_thinking_2 = [r for r in gen_2 if not r.is_thinking]

        assert len(tool_2) == 0  # No more tool calls
        assert len(thinking_2) >= 1
        assert "Cloudy" in "".join(r.text for r in thinking_2)
        assert len(non_thinking_2) >= 1
        final_text = "".join(r.text for r in non_thinking_2)
        assert "7°C" in final_text
        assert "13°C" in final_text

    def test_multi_tool_full_flow(self):
        """Flow with two tools: user → think → 2 tool calls → 2 results → think → answer."""
        # ── Initial prompt ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You help with weather and time."},
            {"role": "user", "content": "Weather in NYC and time in EST?"},
        ]
        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)

        # ── Model thinks, calls both tools ──
        model_tokens_1 = [
            THINKING_START,
            "Two requests: weather and time. I'll call both.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">EST</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_1)))
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]

        assert len(tool_1) == 1
        assert len(tool_1[0].tool_calls) == 2
        assert tool_1[0].tool_calls[0].name == "get_weather"
        assert tool_1[0].tool_calls[1].name == "get_time"

        # ── Add assistant + both tool results ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Two requests: weather and time. I'll call both.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "EST"}',
                        },
                    },
                ],
            }
        )
        messages.append({"role": "tool", "content": "72°F, sunny"})
        messages.append({"role": "tool", "content": "2:30 PM EST"})

        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Verify multi-tool result encoding
        # Count is 2: 1 in _TOOLS_SYSTEM_TEMPLATE example + 1 in conversation
        assert prompt_2.count("<function_results>") == 2
        assert prompt_2.count("</function_results>") == 2
        assert "<result>72°F, sunny</result>" in prompt_2
        assert "<result>2:30 PM EST</result>" in prompt_2
        assert prompt_2.endswith(THINKING_START)

        # ── Model thinks about results, answers ──
        model_tokens_2 = [
            THINKING_START,
            "Got both results. Weather is 72°F sunny, time is 2:30 PM.",
            THINKING_END,
            "In NYC it's currently 72°F and sunny. The time in EST is 2:30 PM.",
        ]
        results_2 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_2)))

        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        gen_2 = [r for r in results_2 if isinstance(r, GenerationResponse)]
        non_thinking_2 = [r for r in gen_2 if not r.is_thinking]

        assert len(tool_2) == 0
        final_text = "".join(r.text for r in non_thinking_2)
        assert "72°F" in final_text
        assert "2:30 PM" in final_text

    def test_two_user_turns_reasoning_stripped(self):
        """Turn 2: old reasoning_content is stripped from history.

        Per the vLLM spec, clear_reasoning_content is called between user turns
        to save bandwidth. Our _drop_old_thinking handles this.
        """
        # Full turn 1 conversation (already completed)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Hangzhou?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need to call get_weather for Hangzhou.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Cloudy 7~13°C"},
            {
                "role": "assistant",
                "content": "The weather in Hangzhou is cloudy, 7-13°C.",
                "reasoning_content": "The tool returned cloudy weather. I'll summarize.",
            },
            # Turn 2: user asks again
            {"role": "user", "content": "What about Beijing?"},
        ]

        prompt = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Old reasoning_content from turn 1 assistants should be STRIPPED
        # (they're before the last user message at index 5)
        assert "I need to call get_weather" not in prompt
        assert "tool returned cloudy" not in prompt

        # But the assistant's content and tool calls should still be there
        assert "cloudy, 7-13°C" in prompt
        assert TOOL_CALLS_START in prompt

        # Prompt ends with <think> for the new turn
        assert prompt.endswith(THINKING_START)

        # ── Turn 2: Model thinks, calls tool for Beijing ──
        model_tokens = [
            THINKING_START,
            "Now the user wants Beijing weather.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Beijing</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"
        args = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {"city": "Beijing"}

    def test_chained_tool_calls_loop(self):
        """Model calls tool, gets result, calls another tool, gets result, answers.

        This simulates the inner while loop from the vLLM spec where the model
        may need multiple sub-turns of tool calling before it has enough info.
        """
        # ── Sub-turn 1: user asks, model calls get_time ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather in Hangzhou tomorrow?"},
        ]

        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)

        # Model first calls get_time to figure out the date
        model_tokens_1 = [
            THINKING_START,
            "I need the current date first to calculate tomorrow.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">Asia/Shanghai</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_1)))
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]
        assert len(tool_1) == 1
        assert tool_1[0].tool_calls[0].name == "get_time"

        # ── Sub-turn 2: add tool result, model calls get_weather ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need the current date first to calculate tomorrow.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "Asia/Shanghai"}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "2025-12-01 14:30 CST"})

        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert "<result>2025-12-01 14:30 CST</result>" in prompt_2
        assert prompt_2.endswith(THINKING_START)

        # Model now knows the date, calls get_weather
        model_tokens_2 = [
            THINKING_START,
            "Today is 2025-12-01, so tomorrow is 2025-12-02.",
            " Now I can check weather for Hangzhou.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Hangzhou</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_2 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_2)))
        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        assert len(tool_2) == 1
        assert tool_2[0].tool_calls[0].name == "get_weather"

        # ── Sub-turn 3: add weather result, model answers ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Today is 2025-12-01, so tomorrow is 2025-12-02. Now I can check weather for Hangzhou.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "Sunny, 5~12°C"})

        prompt_3 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        # Should have both function_results blocks (one per tool round)
        # Count is 3: 1 in _TOOLS_SYSTEM_TEMPLATE example + 2 in conversation
        assert prompt_3.count("<function_results>") == 3
        assert prompt_3.count("</function_results>") == 3
        assert "<result>2025-12-01 14:30 CST</result>" in prompt_3
        assert "<result>Sunny, 5~12°C</result>" in prompt_3
        assert prompt_3.endswith(THINKING_START)

        # Model finally answers
        model_tokens_3 = [
            THINKING_START,
            "I have the weather for tomorrow in Hangzhou.",
            THINKING_END,
            "Tomorrow in Hangzhou will be sunny with temperatures between 5°C and 12°C.",
        ]
        results_3 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_3)))

        tool_3 = [r for r in results_3 if isinstance(r, ToolCallResponse)]
        gen_3 = [r for r in results_3 if isinstance(r, GenerationResponse)]
        non_thinking_3 = [r for r in gen_3 if not r.is_thinking]

        assert len(tool_3) == 0  # No more tool calls — loop ends
        final_text = "".join(r.text for r in non_thinking_3)
        assert "sunny" in final_text.lower()
        assert "5°C" in final_text
        assert "12°C" in final_text
