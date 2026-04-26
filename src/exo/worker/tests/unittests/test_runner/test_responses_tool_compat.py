import json
from typing import Any, cast

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
from exo.shared.types.chunks import TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId, ModelId
from exo.worker.runner.llm_inference.tool_parsers import coerce_tool_calls_to_schema

_TEST_MODEL = ModelId("test-model")
_TEST_COMMAND_ID = CommandId("cmd_1")


@pytest.mark.asyncio
async def test_custom_responses_tool_gets_freeform_input_schema() -> None:
    request = ResponsesRequest(
        model=_TEST_MODEL,
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
    function = cast(dict[str, Any], params.tools[0]["function"])
    assert function["name"] == "apply_patch"
    assert function["parameters"]["required"] == ["input"]
    assert function["parameters"]["properties"]["input"]["type"] == "string"


@pytest.mark.asyncio
async def test_apply_patch_replay_uses_codex_input_argument() -> None:
    request = ResponsesRequest(
        model=_TEST_MODEL,
        input=[
            ApplyPatchCallInputItem(
                call_id="call_1",
                patch="*** Begin Patch\n*** End Patch",
            )
        ],
    )

    params = await responses_request_to_text_generation(request)

    assert params.chat_template_messages is not None
    # chat_template_messages is typed loosely as a list of dicts of mixed
    # values; coerce the navigation through the assistant tool_calls payload.
    first_message = cast(dict[str, Any], params.chat_template_messages[0])
    tool_call = cast(dict[str, Any], first_message["tool_calls"][0])
    assert json.loads(cast(str, tool_call["function"]["arguments"])) == {
        "input": "*** Begin Patch\n*** End Patch"
    }


def test_freeform_tool_args_are_coerced_to_input() -> None:
    tool_calls = [
        ToolCallItem(
            id="call_1",
            name="apply_patch",
            arguments=json.dumps({"patch": "*** Begin Patch\n*** End Patch"}),
        )
    ]
    tools: list[dict[str, Any]] = [
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

    coerced = coerce_tool_calls_to_schema(tool_calls, tools)

    assert json.loads(coerced[0].arguments) == {
        "input": "*** Begin Patch\n*** End Patch"
    }


def test_apply_patch_input_drops_duplicate_end_marker() -> None:
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
    tools: list[dict[str, Any]] = [
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

    coerced = coerce_tool_calls_to_schema(tool_calls, tools)

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


def test_apply_patch_add_file_prefixes_content_lines() -> None:
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
    tools: list[dict[str, Any]] = [
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

    coerced = coerce_tool_calls_to_schema(tool_calls, tools)

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
async def test_reasoning_replay_does_not_split_tool_call_and_output() -> None:
    request = ResponsesRequest(
        model=_TEST_MODEL,
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
async def test_tool_call_response_omits_pre_tool_message() -> None:
    async def chunks():
        yield TokenChunk(
            model=_TEST_MODEL,
            text="Creating hello.txt",
            token_id=1,
            usage=None,
        )
        yield ToolCallChunk(
            model=_TEST_MODEL,
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
        async for chunk in collect_responses_response(
            _TEST_COMMAND_ID, "test-model", chunks()
        )
    ]

    output = cast(list[dict[str, Any]], responses[0]["output"])
    assert [item["type"] for item in output] == ["function_call"]
    assert responses[0]["output_text"] == "Creating hello.txt"


@pytest.mark.asyncio
async def test_streaming_text_deltas_are_not_buffered() -> None:
    async def chunks():
        yield TokenChunk(
            model=_TEST_MODEL,
            text="Creating hello.txt",
            token_id=1,
            usage=None,
        )
        yield ToolCallChunk(
            model=_TEST_MODEL,
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
        async for event in generate_responses_stream(
            _TEST_COMMAND_ID, "test-model", chunks()
        )
    ]

    assert "response.output_text.delta" in "".join(events)
    completed = cast(dict[str, Any], json.loads(events[-1].split("data: ", 1)[1]))
    output = cast(list[dict[str, Any]], completed["response"]["output"])
    assert [item["type"] for item in output] == ["function_call"]
    assert completed["response"]["output_text"] == "Creating hello.txt"
