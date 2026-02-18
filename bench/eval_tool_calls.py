# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the numeric result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    },
}

SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search for products in a catalog by query, category, and price",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "food", "books"],
                    "description": "Product category to filter by",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in USD",
                },
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS: list[dict[str, Any]] = [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL]


@dataclass
class Scenario:
    name: str
    description: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    expect_tool_call: bool
    expected_function: str | None = None
    required_arg_keys: list[str] | None = None
    # For multi-turn: fake tool result to inject, then verify the follow-up.
    tool_result: str | None = None


SCENARIOS: list[Scenario] = [
    # -- Should call a tool --------------------------------------------------
    Scenario(
        name="weather_simple",
        description="Basic weather query -> get_current_weather",
        messages=[
            {"role": "user", "content": "What's the weather like in Tokyo right now?"}
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="get_current_weather",
        required_arg_keys=["location"],
    ),
    Scenario(
        name="calculator_simple",
        description="Math question -> calculate",
        messages=[
            {
                "role": "user",
                "content": "Use the calculator to compute 3847 * 926 + 17293",
            }
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="calculate",
        required_arg_keys=["expression"],
    ),
    Scenario(
        name="search_with_filters",
        description="Product search with category and price filter",
        messages=[{"role": "user", "content": "Find me electronics under $50"}],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="search_products",
        required_arg_keys=["query"],
    ),
    # -- Multi-turn: tool call then follow-up --------------------------------
    Scenario(
        name="weather_multi_turn",
        description="Weather query -> tool result -> natural language summary",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="get_current_weather",
        required_arg_keys=["location"],
        tool_result=json.dumps(
            {
                "temperature": "18C",
                "condition": "partly cloudy",
                "humidity": "65%",
                "wind": "12 km/h NW",
            }
        ),
    ),
    Scenario(
        name="calculator_multi_turn",
        description="Math query -> tool result -> model reports the answer",
        messages=[
            {
                "role": "user",
                "content": "Use the calculator to compute 1847 * 263 + 5921",
            }
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="calculate",
        required_arg_keys=["expression"],
        tool_result=json.dumps({"result": 491682}),
    ),
    Scenario(
        name="search_multi_turn",
        description="Search query -> tool result -> model summarizes products",
        messages=[
            {"role": "user", "content": "Search for books about machine learning"}
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="search_products",
        required_arg_keys=["query"],
        tool_result=json.dumps(
            {
                "results": [
                    {
                        "name": "Hands-On Machine Learning",
                        "price": 45.99,
                        "rating": 4.8,
                    },
                    {
                        "name": "Deep Learning with Python",
                        "price": 39.99,
                        "rating": 4.6,
                    },
                ]
            }
        ),
    ),
    # -- Sequential tool calls: thinking + tool call, NO final answer ----------
    Scenario(
        name="chained_tool_calls_same",
        description="Thinking + weather(Tokyo) -> result -> model must call weather(London)",
        messages=[
            {"role": "user", "content": "Compare the weather in Tokyo and London."},
            {
                "role": "assistant",
                "content": "I'll check both cities. Let me start with Tokyo.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": json.dumps({"location": "Tokyo"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps({"temperature": "25C", "condition": "sunny"}),
            },
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="get_current_weather",
        required_arg_keys=["location"],
    ),
    Scenario(
        name="chained_tool_calls_different",
        description="Thinking + weather(Berlin) -> result -> model must call calculator",
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Berlin, and also use the calculator to compute 4819 * 37 + 291.",
            },
            {
                "role": "assistant",
                "content": "I'll handle both. Let me check Berlin's weather first.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": json.dumps({"location": "Berlin"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": json.dumps({"temperature": "12C", "condition": "rainy"}),
            },
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="calculate",
        required_arg_keys=["expression"],
    ),
    Scenario(
        name="chained_tool_calls_three",
        description="Two prior thinking+tool calls -> results -> model must make a third",
        messages=[
            {"role": "user", "content": "Compare weather in Tokyo, Paris, and London."},
            {
                "role": "assistant",
                "content": "I'll check all three cities. Starting with Tokyo.",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": json.dumps({"location": "Tokyo"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": json.dumps({"temperature": "25C", "condition": "sunny"}),
            },
            {
                "role": "assistant",
                "content": "Got Tokyo. Now checking Paris.",
                "tool_calls": [
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_4",
                "content": json.dumps({"temperature": "18C", "condition": "cloudy"}),
            },
        ],
        tools=ALL_TOOLS,
        expect_tool_call=True,
        expected_function="get_current_weather",
        required_arg_keys=["location"],
    ),
    # -- Should NOT call a tool ----------------------------------------------
    Scenario(
        name="no_tool_joke",
        description="Joke request should NOT trigger any tool",
        messages=[{"role": "user", "content": "Tell me a funny joke about cats."}],
        tools=ALL_TOOLS,
        expect_tool_call=False,
    ),
    Scenario(
        name="no_tool_factual",
        description="Factual question answerable from training data",
        messages=[{"role": "user", "content": "What is the capital of Japan?"}],
        tools=ALL_TOOLS,
        expect_tool_call=False,
    ),
]

ApiName = Literal["openai", "claude", "responses"]


@dataclass
class ParsedResponse:
    finish_reason: str  # "tool_calls" | "stop" | ...
    has_tool_call: bool
    tool_call: dict[str, str] | None  # {"id": ..., "name": ..., "arguments": ...}
    content: str | None


@dataclass
class ScenarioResult:
    name: str
    api: str
    phase: str  # "tool_call" or "follow_up"
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


def validate_args(args_str: str, required_keys: list[str]) -> tuple[bool, str | None]:
    """Parse JSON arguments and check required keys exist."""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, f"Invalid JSON: {exc}"
    if not isinstance(args, dict):
        return False, f"Expected dict, got {type(args).__name__}"
    missing = [k for k in required_keys if k not in args]
    if missing:
        return False, f"Missing keys: {missing}"
    return True, None


def call_api(
    client: httpx.Client,
    host: str,
    port: int,
    path: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    """POST to http://{host}:{port}{path}, return (response_json, latency_ms)."""
    url = f"http://{host}:{port}{path}"
    t0 = time.monotonic()
    resp = client.post(url, json=body, timeout=timeout)
    latency = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), latency


def _openai_build_request(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build request for /v1/chat/completions."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    return "/v1/chat/completions", body


def _openai_parse_response(data: dict[str, Any]) -> ParsedResponse:
    """Parse OpenAI Chat Completions response into common format."""
    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason", "")
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls")
    content = message.get("content")

    has_tool_call = isinstance(tool_calls, list) and len(tool_calls) > 0
    tool_call_info: dict[str, str] | None = None
    if has_tool_call:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        tool_call_info = {
            "id": tc.get("id", "call_0"),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        }

    return ParsedResponse(
        finish_reason=finish_reason,
        has_tool_call=has_tool_call,
        tool_call=tool_call_info,
        content=content,
    )


def _openai_build_followup(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    parsed: ParsedResponse,
    tool_result: str,
) -> tuple[str, dict[str, Any]]:
    """Build multi-turn follow-up for OpenAI Chat Completions."""
    assert parsed.tool_call is not None
    tc = parsed.tool_call
    followup_messages: list[dict[str, Any]] = list(messages) + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_result,
        },
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": followup_messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    return "/v1/chat/completions", body


def _claude_translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-format tools to Claude format."""
    claude_tools: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        claude_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            }
        )
    return claude_tools


def _claude_translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-format messages to Claude Messages format."""
    claude_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "user":
            claude_messages.append(
                {
                    "role": "user",
                    "content": msg["content"],
                }
            )
        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text_content = msg.get("content")
            if text_content and isinstance(text_content, str) and text_content.strip():
                content_blocks.append({"type": "text", "text": text_content})
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", "call_0"),
                            "name": fn.get("name", ""),
                            "input": args_dict,
                        }
                    )
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            claude_messages.append(
                {
                    "role": "assistant",
                    "content": content_blocks,
                }
            )
        elif role == "tool":
            claude_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", "call_0"),
                            "content": msg.get("content", ""),
                        }
                    ],
                }
            )
        elif role == "system":
            pass

    return claude_messages


def _claude_build_request(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build request for /v1/messages."""
    claude_messages = _claude_translate_messages(messages)
    claude_tools = _claude_translate_tools(tools)

    system_content: str | None = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break

    body: dict[str, Any] = {
        "model": model,
        "messages": claude_messages,
        "tools": claude_tools,
        "max_tokens": 4096,
    }
    if system_content is not None:
        body["system"] = system_content

    return "/v1/messages", body


def _claude_parse_response(data: dict[str, Any]) -> ParsedResponse:
    """Parse Claude Messages response into common format."""
    stop_reason = data.get("stop_reason", "")
    content_blocks = data.get("content", [])

    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "end_turn":
        finish_reason = "stop"
    else:
        finish_reason = stop_reason

    tool_call_info: dict[str, str] | None = None
    text_parts: list[str] = []
    has_tool_call = False

    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "tool_use":
            has_tool_call = True
            if tool_call_info is None:
                input_data = block.get("input", {})
                tool_call_info = {
                    "id": block.get("id", "call_0"),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(input_data)
                    if isinstance(input_data, dict)
                    else str(input_data),
                }
        elif block_type == "text":
            text = block.get("text", "")
            if text.strip():
                text_parts.append(text)

    content = "\n".join(text_parts) if text_parts else None

    return ParsedResponse(
        finish_reason=finish_reason,
        has_tool_call=has_tool_call,
        tool_call=tool_call_info,
        content=content,
    )


def _claude_build_followup(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    parsed: ParsedResponse,
    tool_result: str,
) -> tuple[str, dict[str, Any]]:
    """Build multi-turn follow-up for Claude Messages."""
    assert parsed.tool_call is not None
    tc = parsed.tool_call

    try:
        args_dict = json.loads(tc["arguments"])
    except (json.JSONDecodeError, TypeError):
        args_dict = {}

    claude_messages = _claude_translate_messages(messages)

    claude_messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": args_dict,
                }
            ],
        }
    )

    claude_messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": tool_result,
                }
            ],
        }
    )

    claude_tools = _claude_translate_tools(tools)

    system_content: str | None = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break

    body: dict[str, Any] = {
        "model": model,
        "messages": claude_messages,
        "tools": claude_tools,
        "max_tokens": 4096,
    }
    if system_content is not None:
        body["system"] = system_content

    return "/v1/messages", body


def _responses_translate_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI chat messages to Responses API input items."""
    items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role in ("user", "system"):
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": msg["content"],
                }
            )
        elif role == "assistant":
            text_content = msg.get("content")
            if text_content and isinstance(text_content, str) and text_content.strip():
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": text_content,
                    }
                )
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", "call_0"),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", "call_0"),
                    "output": msg.get("content", ""),
                }
            )

    return items


def _responses_build_request(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build request for /v1/responses."""
    input_items = _responses_translate_input(messages)

    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "tools": tools,
        "temperature": 0.0,
        "max_output_tokens": 4096,
    }
    return "/v1/responses", body


def _responses_parse_response(data: dict[str, Any]) -> ParsedResponse:
    """Parse OpenAI Responses API response into common format."""
    output = data.get("output", [])

    tool_call_info: dict[str, str] | None = None
    text_parts: list[str] = []
    has_tool_call = False

    for item in output:
        item_type = item.get("type")
        if item_type == "function_call":
            has_tool_call = True
            if tool_call_info is None:
                tool_call_info = {
                    "id": item.get("call_id", "call_0"),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                }
        elif item_type == "message":
            msg_content = item.get("content", [])
            if isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text and text.strip():
                            text_parts.append(text)
            elif isinstance(msg_content, str) and msg_content.strip():
                text_parts.append(msg_content)

    content = "\n".join(text_parts) if text_parts else None

    if has_tool_call:
        finish_reason = "tool_calls"
    else:
        status = data.get("status", "completed")
        finish_reason = "stop" if status == "completed" else status

    return ParsedResponse(
        finish_reason=finish_reason,
        has_tool_call=has_tool_call,
        tool_call=tool_call_info,
        content=content,
    )


def _responses_build_followup(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    parsed: ParsedResponse,
    tool_result: str,
) -> tuple[str, dict[str, Any]]:
    """Build multi-turn follow-up for Responses API."""
    assert parsed.tool_call is not None
    tc = parsed.tool_call

    input_items = _responses_translate_input(messages)

    input_items.append(
        {
            "type": "function_call",
            "call_id": tc["id"],
            "name": tc["name"],
            "arguments": tc["arguments"],
        }
    )

    input_items.append(
        {
            "type": "function_call_output",
            "call_id": tc["id"],
            "output": tool_result,
        }
    )

    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "tools": tools,
        "temperature": 0.0,
        "max_output_tokens": 4096,
    }
    return "/v1/responses", body


ADAPTERS: dict[ApiName, dict[str, Any]] = {
    "openai": {
        "build_request": _openai_build_request,
        "parse_response": _openai_parse_response,
        "build_followup": _openai_build_followup,
    },
    "claude": {
        "build_request": _claude_build_request,
        "parse_response": _claude_parse_response,
        "build_followup": _claude_build_followup,
    },
    "responses": {
        "build_request": _responses_build_request,
        "parse_response": _responses_parse_response,
        "build_followup": _responses_build_followup,
    },
}


def run_scenario(
    client: httpx.Client,
    host: str,
    port: int,
    model: str,
    scenario: Scenario,
    api_name: ApiName,
    timeout: float,
    verbose: bool,
) -> list[ScenarioResult]:
    """Run a single scenario against one API adapter. Returns 1-2 results."""
    adapter = ADAPTERS[api_name]
    build_request = adapter["build_request"]
    parse_response = adapter["parse_response"]
    build_followup = adapter["build_followup"]
    results: list[ScenarioResult] = []

    # --- Phase 1: initial request ---
    path, body = build_request(model, scenario.messages, scenario.tools)

    try:
        data, latency = call_api(client, host, port, path, body, timeout)
    except Exception as exc:
        results.append(
            ScenarioResult(
                name=scenario.name,
                api=api_name,
                phase="tool_call",
                passed=False,
                error=f"API error: {exc}",
            )
        )
        return results

    if verbose:
        print(
            f"    [{api_name}] response: {json.dumps(data, indent=2)}", file=sys.stderr
        )

    parsed = parse_response(data)
    checks: dict[str, bool] = {}

    if scenario.expect_tool_call:
        checks["finish_reason_tool_calls"] = parsed.finish_reason == "tool_calls"
        checks["has_tool_call"] = parsed.has_tool_call

        args_err: str | None = None
        if parsed.has_tool_call and parsed.tool_call is not None:
            checks["correct_function"] = (
                scenario.expected_function is None
                or parsed.tool_call["name"] == scenario.expected_function
            )
            if scenario.required_arg_keys:
                ok, args_err = validate_args(
                    parsed.tool_call["arguments"], scenario.required_arg_keys
                )
                checks["valid_arguments"] = ok
            else:
                checks["valid_arguments"] = True
        else:
            checks["correct_function"] = False
            checks["valid_arguments"] = False
            args_err = "No tool call returned"

        passed = all(checks.values())
        error = args_err if not passed else None
    else:
        checks["finish_reason_stop"] = parsed.finish_reason == "stop"
        checks["no_tool_call"] = not parsed.has_tool_call
        checks["has_content"] = (
            parsed.content is not None and len(parsed.content.strip()) > 0
        )
        passed = all(checks.values())
        error = (
            None
            if passed
            else (
                f"finish_reason={parsed.finish_reason}, "
                f"tool_call={'yes' if parsed.has_tool_call else 'no'}, "
                f"content={'yes' if parsed.content else 'no'}"
            )
        )

    results.append(
        ScenarioResult(
            name=scenario.name,
            api=api_name,
            phase="tool_call",
            passed=passed,
            checks=checks,
            error=error,
            latency_ms=latency,
        )
    )

    # --- Phase 2: multi-turn follow-up ---
    if (
        scenario.tool_result is not None
        and parsed.has_tool_call
        and parsed.tool_call is not None
    ):
        followup_path, followup_body = build_followup(
            scenario.messages,
            scenario.tools,
            model,
            parsed,
            scenario.tool_result,
        )

        try:
            data2, latency2 = call_api(
                client, host, port, followup_path, followup_body, timeout
            )
        except Exception as exc:
            results.append(
                ScenarioResult(
                    name=scenario.name,
                    api=api_name,
                    phase="follow_up",
                    passed=False,
                    error=f"API error: {exc}",
                )
            )
            return results

        if verbose:
            print(
                f"    [{api_name}] follow_up response: {json.dumps(data2, indent=2)}",
                file=sys.stderr,
            )

        parsed2 = parse_response(data2)
        checks2: dict[str, bool] = {}
        checks2["finish_reason_stop"] = parsed2.finish_reason == "stop"
        checks2["no_tool_call"] = not parsed2.has_tool_call
        checks2["has_content"] = (
            parsed2.content is not None and len(parsed2.content.strip()) > 0
        )

        passed2 = all(checks2.values())
        error2: str | None = None
        if not passed2:
            error2 = (
                f"finish_reason={parsed2.finish_reason}, "
                f"tool_call={'yes' if parsed2.has_tool_call else 'no'}, "
                f"content={'yes' if parsed2.content else 'no'}"
            )
        results.append(
            ScenarioResult(
                name=scenario.name,
                api=api_name,
                phase="follow_up",
                passed=passed2,
                checks=checks2,
                error=error2,
                latency_ms=latency2,
            )
        )

    return results


def result_to_dict(result: ScenarioResult) -> dict[str, Any]:
    """Convert a ScenarioResult to a JSON-serializable dict."""
    return {
        "name": result.name,
        "api": result.api,
        "phase": result.phase,
        "passed": result.passed,
        "checks": result.checks,
        "error": result.error,
        "latency_ms": round(result.latency_ms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-API tool-calling eval for exo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --model mlx-community/Qwen3-30B-A3B-4bit
  %(prog)s --model my-model --api openai --repeat 3
  %(prog)s --model my-model --api all --scenarios weather_simple calculator_multi_turn
  %(prog)s --model my-model --stdout
""",
    )
    parser.add_argument("--model", required=True, help="Model ID to test")
    parser.add_argument(
        "--host",
        default=os.environ.get("EXO_HOST", "localhost"),
        help="API host (default: $EXO_HOST or localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EXO_PORT", "52415")),
        help="API port (default: $EXO_PORT or 52415)",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "claude", "responses", "all"],
        default="all",
        help="Which API adapter(s) to test (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each scenario N times (default: 1)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Run only these scenarios (by name)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full API responses to stderr",
    )
    parser.add_argument(
        "--json-out",
        default="bench/eval_results.json",
        help="Write JSON results to file (default: bench/eval_results.json)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write JSON results to stdout instead of file",
    )
    args = parser.parse_args()

    # Select scenarios
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s.name in args.scenarios]
        if not scenarios:
            print(
                f"No matching scenarios. Available: {[s.name for s in SCENARIOS]}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Select APIs
    api_names: list[ApiName] = (
        ["openai", "claude", "responses"] if args.api == "all" else [args.api]
    )

    total_runs = len(scenarios) * args.repeat * len(api_names)
    log = sys.stderr if args.stdout else sys.stdout

    print(f"Model:     {args.model}", file=log)
    print(f"Endpoint:  http://{args.host}:{args.port}", file=log)
    print(f"APIs:      {', '.join(api_names)}", file=log)
    print(
        f"Scenarios: {len(scenarios)} x {args.repeat} repeats x {len(api_names)} APIs = {total_runs} runs",
        file=log,
    )
    print("=" * 72, file=log)

    all_results: list[ScenarioResult] = []

    with httpx.Client() as client:
        for run_idx in range(args.repeat):
            if args.repeat > 1:
                print(f"\n--- Run {run_idx + 1}/{args.repeat} ---", file=log)

            for scenario in scenarios:
                for api_name in api_names:
                    print(
                        f"\n  [{api_name:>9}] {scenario.name}: {scenario.description}",
                        file=log,
                    )

                    scenario_results = run_scenario(
                        client,
                        args.host,
                        args.port,
                        args.model,
                        scenario,
                        api_name,
                        args.timeout,
                        args.verbose,
                    )
                    all_results.extend(scenario_results)

                    for r in scenario_results:
                        status = "PASS" if r.passed else "FAIL"
                        print(
                            f"    [{r.phase:>10}] {status}  ({r.latency_ms:.0f}ms)",
                            file=log,
                        )
                        for check_name, check_ok in r.checks.items():
                            mark = "+" if check_ok else "-"
                            print(f"      {mark} {check_name}", file=log)
                        if r.error:
                            print(f"      ! {r.error}", file=log)

    # --- Summary ---
    print(f"\n{'=' * 72}", file=log)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)

    tool_call_results = [r for r in all_results if r.phase == "tool_call"]
    follow_up_results = [r for r in all_results if r.phase == "follow_up"]
    tc_passed = sum(1 for r in tool_call_results if r.passed)
    fu_passed = sum(1 for r in follow_up_results if r.passed)
    avg_latency = sum(r.latency_ms for r in all_results) / total if total else 0

    print(
        f"Total:       {passed}/{total} passed ({100 * passed / total:.0f}%)", file=log
    )
    print(f"Tool call:   {tc_passed}/{len(tool_call_results)} passed", file=log)
    if follow_up_results:
        print(f"Follow-up:   {fu_passed}/{len(follow_up_results)} passed", file=log)
    print(f"Avg latency: {avg_latency:.0f}ms", file=log)

    # Per-API breakdown
    for api_name in api_names:
        api_results = [r for r in all_results if r.api == api_name]
        api_passed = sum(1 for r in api_results if r.passed)
        print(f"  {api_name:>9}: {api_passed}/{len(api_results)} passed", file=log)

    if passed < total:
        print("\nFailed:", file=log)
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name} [{r.api}/{r.phase}]: {r.error}", file=log)

    json_results = [result_to_dict(r) for r in all_results]

    if args.stdout:
        print(json.dumps(json_results, indent=2))
    else:
        json_path = args.json_out
        parent = os.path.dirname(json_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
            f.write("\n")
        print(f"\nJSON results written to {json_path}", file=log)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
