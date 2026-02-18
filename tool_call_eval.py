#!/usr/bin/env python3
"""Tool-calling eval for exo's OpenAI-compatible API.

Tests whether models correctly:
- Trigger tool calls when appropriate
- Return valid JSON arguments matching function schemas
- Handle multi-turn tool use (call -> result -> final answer)
- Avoid calling tools when unnecessary

Start exo with a model first, then run:
    uv run python tool_call_eval.py --model <model-id>
    uv run python tool_call_eval.py --model <model-id> --host 10.0.0.5 --port 52415
    uv run python tool_call_eval.py --model <model-id> --repeat 3
    uv run python tool_call_eval.py --model <model-id> --scenarios weather_simple calculator_multi_turn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import httpx

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
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

CALCULATOR_TOOL = {
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

SEARCH_TOOL = {
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

ALL_TOOLS = [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL]

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    messages: list[dict[str, object]]
    tools: list[dict[str, object]]
    expect_tool_call: bool
    expected_function: str | None = None
    required_arg_keys: list[str] | None = None
    # For multi-turn: fake tool result to inject, then verify the follow-up.
    tool_result: str | None = None


SCENARIOS = [
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
    # This is the critical scenario for the Harmony recipient placement fix.
    #
    # When an assistant message has both thinking content and a tool_call,
    # AND there is no subsequent final-answer assistant message, the Jinja
    # template renders BOTH the analysis and the tool call:
    #
    #   <|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>
    #   <|start|>assistant to=functions.X<|channel|>commentary json<|message|>...<|call|>
    #
    # The two consecutive assistant messages have INCONSISTENT start patterns
    # (one has <|channel|> immediately, the other has to= first).
    # This confuses the model when it needs to generate its own tool call.
    #
    # The reformat fix makes both start with <|start|>assistant<|channel|>,
    # only differing in the channel name (analysis vs commentary).
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

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    phase: str  # "tool_call" or "follow_up"
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def validate_args(args_str: str, required_keys: list[str]) -> tuple[bool, str | None]:
    """Parse JSON arguments and check required keys exist."""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError) as e:
        return False, f"Invalid JSON: {e}"
    if not isinstance(args, dict):
        return False, f"Expected dict, got {type(args).__name__}"
    missing = [k for k in required_keys if k not in args]
    if missing:
        return False, f"Missing keys: {missing}"
    return True, None


def call_api(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]],
    timeout: float,
) -> tuple[dict[str, object], float]:
    """POST to /chat/completions, return (response_json, latency_ms)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    t0 = time.monotonic()
    resp = client.post(url, json=body, timeout=timeout)
    latency = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), latency


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    client: httpx.Client,
    base_url: str,
    model: str,
    scenario: Scenario,
    timeout: float,
    verbose: bool,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []

    # --- Phase 1: initial request ---
    try:
        data, latency = call_api(
            client, base_url, model, scenario.messages, scenario.tools, timeout
        )
    except Exception as e:
        results.append(
            ScenarioResult(
                name=scenario.name,
                phase="tool_call",
                passed=False,
                error=f"API error: {e}",
            )
        )
        return results

    if verbose:
        print(f"    response: {json.dumps(data, indent=2)}")

    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls")
    content = message.get("content")

    checks: dict[str, bool] = {}

    if scenario.expect_tool_call:
        checks["finish_reason_tool_calls"] = finish_reason == "tool_calls"
        checks["has_tool_call"] = isinstance(tool_calls, list) and len(tool_calls) > 0

        args_err: str | None = None
        if checks["has_tool_call"]:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            checks["correct_function"] = (
                scenario.expected_function is None
                or fn.get("name") == scenario.expected_function
            )
            if scenario.required_arg_keys:
                ok, args_err = validate_args(
                    fn.get("arguments", ""), scenario.required_arg_keys
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
        checks["finish_reason_stop"] = finish_reason == "stop"
        checks["no_tool_call"] = tool_calls is None or len(tool_calls) == 0
        checks["has_content"] = isinstance(content, str) and len(content.strip()) > 0
        passed = all(checks.values())
        error = (
            None
            if passed
            else (
                f"finish_reason={finish_reason}, "
                f"tool_calls={'yes' if tool_calls else 'no'}, "
                f"content={'yes' if content else 'no'}"
            )
        )

    results.append(
        ScenarioResult(
            name=scenario.name,
            phase="tool_call",
            passed=passed,
            checks=checks,
            error=error,
            latency_ms=latency,
        )
    )

    # --- Phase 2: multi-turn follow-up ---
    if scenario.tool_result is not None and checks.get("has_tool_call"):
        tc = tool_calls[0]
        fn = tc.get("function", {})
        follow_up_messages: list[dict[str, object]] = list(scenario.messages) + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.get("id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tc.get("id", "call_0"),
                "content": scenario.tool_result,
            },
        ]

        try:
            data2, latency2 = call_api(
                client,
                base_url,
                model,
                follow_up_messages,
                scenario.tools,
                timeout,
            )
        except Exception as e:
            results.append(
                ScenarioResult(
                    name=scenario.name,
                    phase="follow_up",
                    passed=False,
                    error=f"API error: {e}",
                )
            )
            return results

        if verbose:
            print(f"    follow_up response: {json.dumps(data2, indent=2)}")

        choice2 = data2["choices"][0]
        message2 = choice2.get("message", {})
        checks2: dict[str, bool] = {}
        checks2["finish_reason_stop"] = choice2.get("finish_reason") == "stop"
        tc2 = message2.get("tool_calls")
        checks2["no_tool_call"] = tc2 is None or len(tc2) == 0
        c2 = message2.get("content")
        checks2["has_content"] = isinstance(c2, str) and len(c2.strip()) > 0

        passed2 = all(checks2.values())
        error2 = None
        if not passed2:
            error2 = (
                f"finish_reason={choice2.get('finish_reason')}, "
                f"tool_calls={'yes' if tc2 else 'no'}, "
                f"content={'yes' if c2 else 'no'}"
            )
        results.append(
            ScenarioResult(
                name=scenario.name,
                phase="follow_up",
                passed=passed2,
                checks=checks2,
                error=error2,
                latency_ms=latency2,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-calling eval for exo")
    parser.add_argument("--model", required=True, help="Model ID to test")
    parser.add_argument("--host", default=os.environ.get("EXO_HOST", "localhost"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EXO_PORT", "52415")),
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Per-request timeout (seconds)"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat each scenario N times"
    )
    parser.add_argument(
        "--scenarios", nargs="*", help="Run only these scenarios (by name)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full API responses"
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s.name in args.scenarios]
        if not scenarios:
            print(f"No matching scenarios. Available: {[s.name for s in SCENARIOS]}")
            sys.exit(1)

    base_url = f"http://{args.host}:{args.port}/v1"
    total_runs = len(scenarios) * args.repeat
    print(f"Model:     {args.model}")
    print(f"Endpoint:  {base_url}")
    print(f"Scenarios: {len(scenarios)} x {args.repeat} = {total_runs} runs")
    print("=" * 64)

    all_results: list[ScenarioResult] = []

    with httpx.Client() as client:
        for run_idx in range(args.repeat):
            if args.repeat > 1:
                print(f"\n--- Run {run_idx + 1}/{args.repeat} ---")

            for scenario in scenarios:
                print(f"\n  {scenario.name}: {scenario.description}")

                results = run_scenario(
                    client,
                    base_url,
                    args.model,
                    scenario,
                    args.timeout,
                    args.verbose,
                )
                all_results.extend(results)

                for r in results:
                    status = "PASS" if r.passed else "FAIL"
                    print(f"    [{r.phase:>10}] {status}  ({r.latency_ms:.0f}ms)")
                    for check_name, check_ok in r.checks.items():
                        mark = "+" if check_ok else "-"
                        print(f"      {mark} {check_name}")
                    if r.error:
                        print(f"      ! {r.error}")

    # --- Summary ---
    print(f"\n{'=' * 64}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)

    tool_call_results = [r for r in all_results if r.phase == "tool_call"]
    follow_up_results = [r for r in all_results if r.phase == "follow_up"]
    tc_passed = sum(1 for r in tool_call_results if r.passed)
    fu_passed = sum(1 for r in follow_up_results if r.passed)
    avg_latency = sum(r.latency_ms for r in all_results) / total if total else 0

    print(f"Total:       {passed}/{total} passed ({100 * passed / total:.0f}%)")
    print(f"Tool call:   {tc_passed}/{len(tool_call_results)} passed")
    if follow_up_results:
        print(f"Follow-up:   {fu_passed}/{len(follow_up_results)} passed")
    print(f"Avg latency: {avg_latency:.0f}ms")

    if passed < total:
        print("\nFailed:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name} [{r.phase}]: {r.error}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
