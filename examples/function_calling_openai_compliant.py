#!/usr/bin/env python3
"""
OpenAI-Compliant Function Calling Example for EXO

This example demonstrates the server-side tool calling implementation
that returns OpenAI-compliant responses with tool_calls arrays.

The server now:
1. Parses tool calls from model output automatically
2. Returns properly formatted tool_calls array
3. Sets finish_reason to "tool_calls" when tools are invoked
4. Handles both streaming and non-streaming responses

No client-side parsing needed anymore!
"""

import json
import requests

def get_current_weather(location: str, unit: str = "celsius"):
    """Mock weather data function"""
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "forecast": "Sunny with light clouds"
    }


def chat_completion(messages, tools=None, stream=False):
    """Send chat completion request to EXO server"""
    payload = {
        "model": "llama-3.2-1b",  # or your preferred model
        "messages": messages,
        "temperature": 0.7,
        "stream": stream
    }

    if tools:
        payload["tools"] = tools

    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json=payload,
        stream=stream
    )

    if stream:
        return response
    else:
        return response.json()


def main():
    """
    Demonstrates OpenAI-compliant tool calling workflow.

    The server now returns responses in proper OpenAI format:
    {
      "choices": [{
        "message": {
          "role": "assistant",
          "content": "...",  # content before tool calls (or null)
          "tool_calls": [{  # OpenAI-formatted tool calls
            "id": "call_xyz123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\"}"  # JSON string
            }
          }]
        },
        "finish_reason": "tool_calls"  # Set when tools are called
      }]
    }
    """

    # Define tools in OpenAI format
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }]

    # Initial conversation
    messages = [{
        "role": "user",
        "content": "Hi there, what's the weather in Boston?"
    }]

    print("User: Hi there, what's the weather in Boston?\n")

    # Get initial response with tools
    print("Sending request to EXO server...")
    response = chat_completion(messages, tools=tools)

    print(f"\nServer Response:")
    print(json.dumps(response, indent=2))

    # Extract assistant message
    assistant_message = response["choices"][0]["message"]
    messages.append(assistant_message)

    # Check if assistant called any tools
    if "tool_calls" in assistant_message:
        print(f"\n✅ Tool calls detected! The server parsed them automatically.")
        print(f"Number of tool calls: {len(assistant_message['tool_calls'])}")
        print(f"Finish reason: {response['choices'][0]['finish_reason']}")

        # Execute each tool call
        for tool_call in assistant_message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])

            print(f"\nExecuting tool: {function_name}")
            print(f"Arguments: {function_args}")

            # Call the actual function
            if function_name == "get_current_weather":
                result = get_current_weather(**function_args)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            print(f"Result: {result}")

            # Add tool response to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],  # Link back to the tool call
                "name": function_name,
                "content": json.dumps(result)
            })

        # Get final response with tool results
        print("\nSending tool results back to model...")
        final_response = chat_completion(messages, tools=tools)

        final_message = final_response["choices"][0]["message"]
        print(f"\nAssistant: {final_message.get('content', '')}")

        messages.append(final_message)

    else:
        print(f"\nAssistant: {assistant_message.get('content', '')}")
        print("\n(Model chose not to call any tools)")

    # Print full conversation
    print("\n" + "="*60)
    print("Full Conversation History:")
    print("="*60)
    for msg in messages:
        role = msg["role"].upper()
        if "tool_calls" in msg:
            print(f"\n{role}: [Called {len(msg['tool_calls'])} tool(s)]")
            for tc in msg["tool_calls"]:
                print(f"  - {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "TOOL":
            print(f"\n{role} ({msg['name']}): {msg['content']}")
        else:
            print(f"\n{role}: {msg.get('content', '')}")


def demo_parallel_tools():
    """
    Demonstrates parallel tool calling (multiple tools in one response).

    The server can detect and return multiple tool calls from a single
    model response, enabling efficient parallel execution.
    """
    print("\n" + "="*60)
    print("DEMO: Parallel Tool Calling")
    print("="*60)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]

    messages = [{
        "role": "user",
        "content": "What's the weather in Boston, New York, and San Francisco?"
    }]

    response = chat_completion(messages, tools=tools)
    assistant_message = response["choices"][0]["message"]

    if "tool_calls" in assistant_message:
        print(f"\n✅ Parallel tool calls detected!")
        print(f"Number of simultaneous calls: {len(assistant_message['tool_calls'])}")

        for i, tc in enumerate(assistant_message["tool_calls"], 1):
            args = json.loads(tc["function"]["arguments"])
            print(f"{i}. {tc['function']['name']}(location={args.get('location')})")
    else:
        print("\n⚠️  Model did not make parallel tool calls")


if __name__ == "__main__":
    print("="*60)
    print("  EXO OpenAI-Compliant Tool Calling Demo")
    print("="*60)
    print("\nThis demonstrates server-side tool calling with OpenAI format.")
    print("No client-side parsing required!\n")

    try:
        main()
        demo_parallel_tools()

        print("\n" + "="*60)
        print("Key Implementation Features:")
        print("="*60)
        print("✅ Server-side parsing of tool calls from model output")
        print("✅ OpenAI-compliant response format with tool_calls array")
        print("✅ Proper finish_reason='tool_calls' when tools are invoked")
        print("✅ Support for parallel tool calling")
        print("✅ Works with both streaming and non-streaming")
        print("✅ Arguments always returned as JSON strings (not objects)")
        print("✅ Unique tool_call IDs generated server-side")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to EXO server")
        print("Make sure EXO is running on http://localhost:52415")
        print("\nStart the server with:")
        print("  exo --inference-engine mlx")
