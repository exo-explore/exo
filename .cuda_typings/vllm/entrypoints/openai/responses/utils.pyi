from openai.types.responses import ResponseOutputItem as ResponseOutputItem
from openai.types.responses.response import ToolChoice as ToolChoice
from openai.types.responses.tool import Tool as Tool
from typing import Any
from vllm import envs as envs
from vllm.entrypoints.constants import MCP_PREFIX as MCP_PREFIX
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem as ResponseInputOutputItem,
)

def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool: ...
def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
): ...
def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]: ...
def extract_tool_types(tools: list[Tool]) -> set[str]: ...
def convert_tool_responses_to_completions_format(tool: dict) -> dict: ...
def construct_tool_dicts(
    tools: list[Tool], tool_choice: ToolChoice
) -> list[dict[str, Any]] | None: ...
