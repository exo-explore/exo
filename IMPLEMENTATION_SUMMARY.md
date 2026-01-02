# Tool Calling Implementation Summary

## Issue
Fixes #1074 - Support Tool Use via the API

## Overview
Implemented comprehensive OpenAI-compatible tool/function calling support for EXO's distributed inference system. The implementation enables Llama 3.1, Qwen, and other tool-capable models to invoke external functions through EXO's API.

## Implementation Details

### Files Created
1. **`src/exo/worker/engines/mlx/generator/tool_parser.py`** (150 lines)
   - Model-specific tool call format detection
   - Supports multiple formats:
     - Qwen: `<tool_call>{...}</tool_call>`
     - Llama 3.1 (single/few tools): `<|python_tag|>{...}<|eom_id|>`
     - Llama 3.1 (many tools): Raw JSON `{ "name": "...", "parameters": {...} }`
   - JSON parsing and OpenAI format conversion
   - Tool call text extraction for clean responses

2. **`docs/TOOL_CALLING.md`** (435 lines)
   - Complete usage documentation
   - API examples (basic, streaming, multi-turn)
   - OpenAI SDK integration examples
   - Troubleshooting guide
   - Architecture overview

3. **`examples/tool_calling_example.py`** (220 lines)
   - Working demo with weather and calculator tools
   - Multi-turn conversation handling
   - Tool execution and result integration

### Files Modified
1. **`src/exo/worker/engines/mlx/generator/generate.py`**
   - Extract tools from task parameters
   - Real-time tool call detection (both tagged and raw JSON formats)
   - Tool call accumulation during streaming
   - Automatic `finish_reason` override to "tool_calls"
   - Pass tool calls through GenerationResponse

2. **`src/exo/worker/engines/mlx/utils_mlx.py`**
   - Updated `apply_chat_template()` to accept tools parameter
   - Pass tools to MLX tokenizer's apply_chat_template()
   - Logging for tool template application

3. **`src/exo/worker/engines/mlx/__init__.py`**
   - Added tools parameter to TokenizerWrapper type definition

4. **`src/exo/shared/types/worker/runner_response.py`**
   - Added `tool_calls` field to GenerationResponse

5. **`src/exo/shared/types/chunks.py`**
   - Added `tool_calls` field to TokenChunk

6. **`src/exo/worker/runner/runner.py`**
   - Pass tool_calls through event pipeline to API layer

7. **`src/exo/master/api.py`**
   - Include tool_calls in streaming delta with index field
   - Aggregate tool_calls in non-streaming responses
   - Full OpenAI specification compliance

## Key Features

### Multiple Format Support
The implementation automatically detects and parses three different tool call formats:
- **Qwen models**: XML-style tags with JSON content
- **Llama 3.1 (few tools)**: Python tag markers with JSON
- **Llama 3.1 (many tools)**: Raw JSON without markers

### Streaming Support
- Real-time tool call detection during generation
- Proper streaming delta format with index fields
- Compatible with OpenAI SDK streaming

### Distributed Inference
- Works seamlessly with EXO's pipeline and tensor parallelism
- Tools passed to all nodes in cluster
- Tool detection on rank-0 node only

## Testing

### Tested Models
1. **Llama 3.1 8B (4-bit)** - mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
   - Single tool: `<|python_tag|>` format ✅
   - Multiple tools (11): Raw JSON format ✅
   - Streaming: Full OpenAI compatibility ✅

2. **Qwen3 30B (4-bit)** - mlx-community/Qwen3-30B-A3B-4bit
   - To be tested with different format

### Test Clients
- **cURL**: Direct API testing ✅
- **opencode CLI**: Full tool calling workflow ✅
- **Python requests**: Multi-turn conversations ✅
- **OpenAI SDK**: Compatible (documented)

### Issues Discovered and Fixed

1. **Llama 3.1 python_tag Format**
   - Initial assumption: `<function=...>` format
   - Actual format: `<|python_tag|>...<|eom_id|>`
   - Fixed in commit 0d85a4d

2. **Raw JSON Format Detection**
   - Models use different formats based on tool count
   - Many tools (11+) trigger raw JSON without tags
   - Added parser support in commit 360d75f
   - Added generation detection in commit b8e835c

3. **Streaming Index Field**
   - OpenAI SDK requires `index` field on each tool call
   - Added in streaming delta in commit 6b7b37b

## Commits

All commits on branch `feat/tool-calling-v2`:

1. `673a19a` - feat: implement tool calling support for EXO (main implementation)
2. `0d85a4d` - fix: update tool parser to support Llama 3.1 python_tag format
3. `360d75f` - fix: add support for raw JSON tool call format
4. `b8e835c` - fix: detect raw JSON tool calls during generation
5. `6b7b37b` - fix: add index field to streaming tool calls
6. `8d3ce08` - docs: update tool calling formats with tested implementations

## API Compatibility

### Request Format
```json
{
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "function_name",
        "description": "...",
        "parameters": { "type": "object", ... }
      }
    }
  ],
  "stream": true/false
}
```

### Response Format (Streaming)
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [
        {
          "index": 0,
          "id": "call_0",
          "type": "function",
          "function": {
            "name": "...",
            "arguments": "{...}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Next Steps

1. Test with Qwen3-30B to verify different format handling
2. Test with additional models (Mistral, GPT-OSS)
3. Submit PR to exo-explore/exo
4. Address any feedback from maintainers

## Related Links

- Issue: https://github.com/exo-explore/exo/issues/1074
- MLX-LM Tool Use PR: https://github.com/ml-explore/mlx-lm/pull/217
- OpenAI Tool Calling Spec: https://platform.openai.com/docs/guides/function-calling

## Contributors

- Implementation: Claude Code
- Testing: Lucas Jackson (@lucasajackson)
- Based on: MLX-LM native tool calling support (PR #217)
