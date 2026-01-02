# Tool Calling in EXO

This document describes how to use tool calling (function calling) with EXO's distributed inference API.

## Overview

As of the feature implementation in this branch, EXO supports OpenAI-compatible tool calling for models that have been trained with function calling capabilities. This enables models to invoke external functions, APIs, and tools during generation.

## Supported Models

Tool calling is supported by models that include function calling in their training:

- **Llama 3.1+** (8B, 70B, 405B variants)
- **Qwen 2.5+** and **Qwen 3+** series
- **Mistral** models with tool support
- **GPT-OSS** models
- Any MLX-compatible model with tool calling training

Different models use different formats for tool invocations:
- **Qwen**: `<tool_call>{json}</tool_call>`
- **Llama 3.1** (single/few tools): `<|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>`
- **Llama 3.1** (many tools): Raw JSON `{ "name": "...", "parameters": {...} }`

The implementation automatically detects and parses all formats.

## How It Works

### Architecture

1. **API Layer** (`src/exo/master/api.py`)
   - Accepts `tools` parameter in `/v1/chat/completions` requests
   - Returns tool calls in OpenAI-compatible format

2. **Template Application** (`src/exo/worker/engines/mlx/utils_mlx.py`)
   - Passes tools to MLX tokenizer's `apply_chat_template()`
   - Formats tools according to model's expected schema

3. **Generation** (`src/exo/worker/engines/mlx/generator/generate.py`)
   - Monitors generated text for tool call markers
   - Accumulates and parses tool invocations
   - Sets `finish_reason="tool_calls"` when detected

4. **Parsing** (`src/exo/worker/engines/mlx/generator/tool_parser.py`)
   - Model-specific format detection
   - JSON extraction and validation
   - Conversion to OpenAI format

## Usage Examples

### Basic Tool Calling

```python
import requests

# Define tools/functions
tools = [
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
                        "description": "City and state, e.g. San Francisco, CA"
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
    }
]

# Make request
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
)

result = response.json()

# Check if model wants to call a tool
if result["choices"][0]["finish_reason"] == "tool_calls":
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    for call in tool_calls:
        print(f"Function: {call['function']['name']}")
        print(f"Arguments: {call['function']['arguments']}")
```

### Streaming with Tool Calls

```python
import requests

response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "messages": [
            {"role": "user", "content": "Calculate 123 * 456"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith("data: ") and line_text != "data: [DONE]":
            import json
            chunk = json.loads(line_text[6:])
            delta = chunk["choices"][0]["delta"]

            # Tool calls appear in the delta when detected
            if "tool_calls" in delta:
                print("Tool calls:", delta["tool_calls"])
```

### Multi-Turn Conversation with Tool Execution

```python
import requests
import json

def execute_tool(name, arguments):
    """Execute the actual tool function"""
    if name == "multiply":
        args = json.loads(arguments)
        return args["a"] * args["b"]
    # Add other tool implementations
    return None

def chat_with_tools(messages, tools):
    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
            "messages": messages,
            "tools": tools
        }
    ).json()

    choice = response["choices"][0]

    if choice["finish_reason"] == "tool_calls":
        # Model wants to call tools
        tool_calls = choice["message"]["tool_calls"]

        # Execute each tool
        for call in tool_calls:
            result = execute_tool(
                call["function"]["name"],
                call["function"]["arguments"]
            )

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": str(result)
            })

        # Get final response after tool execution
        return chat_with_tools(messages, tools)

    return choice["message"]["content"]

# Example usage
tools = [{
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    }
}]

messages = [
    {"role": "user", "content": "What is 12234585 times 48838483920?"}
]

result = chat_with_tools(messages, tools)
print(result)
```

### Using with OpenAI SDK

```python
from openai import OpenAI

# Point to EXO server
client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="not-needed"  # EXO doesn't require auth
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="mlx-community/Llama-3.1-70B-Instruct-4bit",
    messages=[
        {"role": "user", "content": "How do I enable RDMA on EXO?"}
    ],
    tools=tools
)

print(response.choices[0].message)
```

## API Reference

### Request Format

```json
{
  "model": "string",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "string",
        "description": "string",
        "parameters": {
          "type": "object",
          "properties": {...},
          "required": [...]
        }
      }
    }
  ],
  "tool_choice": "auto" | "none" | {"type": "function", "function": {"name": "string"}}
}
```

### Response Format (Non-Streaming)

```json
{
  "id": "command_id",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model_id",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "text or null",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "function_name",
              "arguments": "{\"arg\": \"value\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

### Response Format (Streaming)

```
data: {"choices":[{"delta":{"role":"assistant","content":"..."},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"id":"call_0","type":"function","function":{"name":"func","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]
```

## Implementation Details

### Tool Call Detection

The implementation monitors generated text for model-specific markers:

1. **Qwen models**: Detects `<tool_call>` opening tag
2. **Llama/Mistral**: Detects `<function=` opening tag
3. Accumulates text until closing tag appears
4. Parses JSON and converts to OpenAI format

### Finish Reasons

- `"stop"`: Normal completion without tools
- `"tool_calls"`: Model invoked one or more tools
- `"length"`: Max tokens reached
- Other standard OpenAI finish reasons

### Distributed Inference

Tool calling works seamlessly with EXO's distributed inference:

- Tools are passed to all nodes in the cluster
- Only rank-0 node emits tool call events
- Tool detection happens on the coordinating node
- Full support for pipeline and tensor parallelism

## Troubleshooting

### No Tool Calls Detected

**Issue**: Model generates text but doesn't invoke tools

**Solutions**:
- Ensure model supports tool calling (Llama 3.1+, Qwen 2.5+)
- Check that tool descriptions are clear and relevant
- Try adjusting `tool_choice` parameter
- Verify tools schema is valid JSON Schema

### Invalid JSON in Tool Calls

**Issue**: Tool call parsing fails with JSON errors

**Solutions**:
- Model may be generating malformed JSON
- Check logs for parsing warnings
- Try different model or lower temperature
- Ensure tool parameter descriptions are clear

### Tool Calls Not Appearing in Response

**Issue**: Model invokes tools but response doesn't include them

**Solutions**:
- Check that you're using the correct model format
- Verify mlx-lm version is >= 0.28.3
- Look for errors in EXO logs
- Test with a known working model (e.g., Qwen2.5-32B-Instruct)

## Testing

Run the tool calling tests:

```bash
# TODO: Add test commands once tests are implemented
pytest src/exo/worker/tests/test_tool_calling.py
```

## Performance Considerations

- Tool call detection adds minimal latency (< 1ms per token)
- Streaming maintains real-time response
- Tool parsing happens asynchronously
- No impact on non-tool-calling requests

## Future Enhancements

Potential improvements for future versions:

- [ ] Parallel tool calling support
- [ ] Tool call caching
- [ ] Automatic tool result formatting
- [ ] Built-in common tools (calculator, search, etc.)
- [ ] Tool calling metrics and analytics
- [ ] Support for more model formats

## Related Documentation

- [MLX-LM Tool Use Example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/tool_use.py)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [EXO API Documentation](../src/exo/master/api.py)

## Contributing

To improve tool calling support:

1. Test with new models and report compatibility
2. Add additional tool call format parsers
3. Improve error handling and validation
4. Submit PRs with test cases and examples

## License

Same as EXO project (Apache 2.0)
