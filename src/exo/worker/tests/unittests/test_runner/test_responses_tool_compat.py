import json

import pytest

from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.types import ToolCallItem
from exo.api.types.openai_responses import (
    ApplyPatchCallInputItem,
    FunctionCallInputItem,
    FunctionCallOutputInputItem,
    ReasoningInputItem,
    ResponsesRequest,
)
from exo.worker.runner.llm_inference.tool_parsers import _coerce_tool_calls_to_schema
from exo.shared.types.chunks import TokenChunk, ToolCallChunk


@pytest.mark.asyncio
async def test_custom_responses_tool_gets_freeform_input_schema():
    request = ResponsesRequest(
        model="test-model",
        input="edit a file",
        tools=[
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
                "format": {"type": "grammar", "description": "Patch text"},
            }
        ],
    )

    params = await responses_request_to_text_generation(request)

    assert params.tools is not None
    function = params.tools[0]["function"]
    assert function["name"] == "apply_patch"
    assert function["parameters"]["required"] == ["input"]
    assert function["parameters"]["properties"]["input"]["type"] == "string"


@pytest.mark.asyncio
async def test_apply_patch_replay_uses_codex_input_argument():
    request = ResponsesRequest(
        model="test-model",
        input=[
            ApplyPatchCallInputItem(
                call_id="call_1",
                patch="*** Begin Patch\n*** End Patch",
            )
        ],
    )

    params = await responses_request_to_text_generation(request)

    assert params.chat_template_messages is not None
    tool_call = params.chat_template_messages[0]["tool_calls"][0]
    assert json.loads(tool_call["function"]["arguments"]) == {
        "input": "*** Begin Patch\n*** End Patch"
    }


def test_freeform_tool_args_are_coerced_to_input():
    tool_calls = [
        ToolCallItem(
            id="call_1",
            name="apply_patch",
            arguments=json.dumps({"patch": "*** Begin Patch\n*** End Patch"}),
        )
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    coerced = _coerce_tool_calls_to_schema(tool_calls, tools)

    assert json.loads(coerced[0].arguments) == {
        "input": "*** Begin Patch\n*** End Patch"
    }


def test_apply_patch_input_drops_duplicate_end_marker():
    tool_calls = [
        ToolCallItem(
            id="call_1",
            name="apply_patch",
            arguments=json.dumps(
                {
                    "input": "\n".join(
                        [
                            "*** Begin Patch",
                            "*** Add File: hello.txt",
                            "+hello",
                            "*** End Patch",
                            "*** End Patch",
                        ]
                    )
                }
            ),
        )
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    coerced = _coerce_tool_calls_to_schema(tool_calls, tools)

    assert json.loads(coerced[0].arguments) == {
        "input": "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: hello.txt",
                "+hello",
                "*** End Patch",
            ]
        )
    }


def test_apply_patch_add_file_prefixes_content_lines():
    tool_calls = [
        ToolCallItem(
            id="call_1",
            name="apply_patch",
            arguments=json.dumps(
                {
                    "input": "\n".join(
                        [
                            "*** Begin Patch",
                            "*** Add File: hello.txt",
                            "hello",
                            "*** End Patch",
                        ]
                    )
                }
            ),
        )
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    coerced = _coerce_tool_calls_to_schema(tool_calls, tools)

    assert json.loads(coerced[0].arguments) == {
        "input": "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: hello.txt",
                "+hello",
                "*** End Patch",
            ]
        )
    }


@pytest.mark.asyncio
async def test_reasoning_replay_does_not_split_tool_call_and_output():
    request = ResponsesRequest(
        model="test-model",
        input=[
            FunctionCallInputItem(
                call_id="call_1",
                name="exec_command",
                arguments=json.dumps({"cmd": "printf ok"}),
            ),
            ReasoningInputItem(summary=[{"text": "thinking"}]),
            FunctionCallOutputInputItem(call_id="call_1", output="ok"),
        ],
    )

    params = await responses_request_to_text_generation(request)

    assert params.chat_template_messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "printf ok"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_tool_call_response_omits_pre_tool_message():
    async def chunks():
        yield TokenChunk(
            model="test-model",
            text="Creating hello.txt",
            token_id=1,
            usage=None,
        )
        yield ToolCallChunk(
            model="test-model",
            tool_calls=[
                ToolCallItem(
                    id="call_1",
                    name="apply_patch",
                    arguments=json.dumps({"input": "*** Begin Patch\n*** End Patch"}),
                )
            ],
            usage=None,
        )

    responses = [
        json.loads(chunk)
        async for chunk in collect_responses_response("cmd_1", "test-model", chunks())
    ]

    output = responses[0]["output"]
    assert [item["type"] for item in output] == ["function_call"]
    assert responses[0]["output_text"] == "Creating hello.txt"


@pytest.mark.asyncio
async def test_streaming_tool_call_response_buffers_pre_tool_text():
    async def chunks():
        yield TokenChunk(
            model="test-model",
            text="Creating hello.txt",
            token_id=1,
            usage=None,
        )
        yield ToolCallChunk(
            model="test-model",
            tool_calls=[
                ToolCallItem(
                    id="call_1",
                    name="apply_patch",
                    arguments=json.dumps({"input": "*** Begin Patch\n*** End Patch"}),
                )
            ],
            usage=None,
        )

    events = [
        event
        async for event in generate_responses_stream("cmd_1", "test-model", chunks())
    ]

    assert "response.output_text.delta" not in "".join(events)
    completed = json.loads(events[-1].split("data: ", 1)[1])
    output = completed["response"]["output"]
    assert [item["type"] for item in output] == ["function_call"]
    assert completed["response"]["output_text"] == "Creating hello.txt"
