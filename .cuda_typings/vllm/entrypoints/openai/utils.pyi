from fastapi import Request as Request
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponseChoice as ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice as ChatCompletionResponseStreamChoice,
)

def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT: ...
async def validate_json_request(raw_request: Request): ...
