#!/usr/bin/env python3
"""
EXO Tool Calling Example

This script demonstrates how to use tool calling with EXO's distributed inference API.

Requirements:
- EXO server running (uv run exo)
- A model instance with tool calling support (e.g., Llama 3.1 or Qwen 2.5)

Usage:
    python examples/tool_calling_example.py
"""

import json

import requests


def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    Mock weather function - in production, this would call a real weather API.

    Args:
        location: City and state, e.g. "San Francisco, CA"
        unit: Temperature unit, "celsius" or "fahrenheit"

    Returns:
        dict with weather information
    """
    # Mock data - replace with actual API call
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "humidity": 65,
    }


def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


# Available tools registry
TOOLS = {"get_weather": get_weather, "multiply": multiply}

# Tool definitions for the API (OpenAI format)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
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
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def execute_tool_call(tool_call: dict) -> str:
    """
    Execute a tool call and return the result as a string.

    Args:
        tool_call: Tool call dict from the API response

    Returns:
        String result of the tool execution
    """
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    print(f"  Executing: {function_name}({arguments})")

    if function_name in TOOLS:
        result = TOOLS[function_name](**arguments)
        return json.dumps(result)
    else:
        return json.dumps({"error": f"Unknown function: {function_name}"})


def chat_with_tools(
    user_message: str,
    model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    api_url: str = "http://localhost:52415/v1/chat/completions",
    max_iterations: int = 5,
) -> str:
    """
    Have a conversation with tool calling support.

    Args:
        user_message: The user's message
        model: Model to use for generation
        api_url: EXO API endpoint
        max_iterations: Maximum number of tool call iterations

    Returns:
        Final response from the model
    """
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Make API request
        response = requests.post(
            api_url,
            json={"model": model, "messages": messages, "tools": TOOL_DEFINITIONS},
        )

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        result = response.json()
        choice = result["choices"][0]
        finish_reason = choice["finish_reason"]

        print(f"Finish reason: {finish_reason}")

        # Check if model wants to call tools
        if finish_reason == "tool_calls":
            tool_calls = choice["message"]["tool_calls"]
            print(f"Model requested {len(tool_calls)} tool call(s)")

            # Add assistant message with tool calls to history
            messages.append(choice["message"])

            # Execute each tool and add results to messages
            for tool_call in tool_calls:
                result_content = execute_tool_call(tool_call)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": result_content,
                    }
                )

            # Continue conversation with tool results
            continue

        # No more tool calls - return final response
        return choice["message"]["content"]

    return "Maximum iterations reached without final response"


def main():
    """Run example conversations with tool calling."""
    print("=" * 60)
    print("EXO Tool Calling Example")
    print("=" * 60)

    examples = [
        "What's the weather like in San Francisco?",
        "What is 12234585 multiplied by 48838483920?",
        "What's the weather in Tokyo in Fahrenheit?",
    ]

    for i, user_message in enumerate(examples, 1):
        print(f"\n\n{'='*60}")
        print(f"Example {i}: {user_message}")
        print(f"{'='*60}")

        try:
            response = chat_with_tools(user_message)
            print(f"\n[Final Response]")
            print(response)
        except Exception as e:
            print(f"\n[Error]: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
