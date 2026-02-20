"""OpenAI Chat Completions API adapter for converting requests/responses."""

import time
from collections.abc import AsyncGenerator
from typing import Any

from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageText,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
    ToolCall,
    Usage,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def chat_request_to_text_generation(
    request: ChatCompletionRequest,
) -> TextGenerationTaskParams:
    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []

    for msg in request.messages:
        # Normalize content to string
        content: str
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        else:
            # List of ChatCompletionMessageText
            content = "\n".join(item.text for item in msg.content)

        # Extract system message as instructions
        if msg.role == "system":
            if instructions is None:
                instructions = content
            else:
                # Append additional system messages
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": "system", "content": content})
        else:
            # Skip messages with no meaningful content
            if (
                msg.content is None
                and msg.reasoning_content is None
                and msg.tool_calls is None
            ):
                continue

            if msg.role in ("user", "assistant", "developer"):
                input_messages.append(InputMessage(role=msg.role, content=content))

            # Build full message dict for chat template (preserves tool_calls etc.)
            # Normalize content for model_dump
            msg_copy = msg.model_copy(update={"content": content})
            dumped: dict[str, Any] = msg_copy.model_dump(exclude_none=True)
            chat_template_messages.append(dumped)

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content="")],
        instructions=instructions,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        enable_thinking=request.enable_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        logprobs=request.logprobs or False,
        top_logprobs=request.top_logprobs,
    )


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    # Build logprobs if available
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    if chunk.is_thinking:
        delta = ChatCompletionMessage(role="assistant", reasoning_content=chunk.text)
    else:
        delta = ChatCompletionMessage(role="assistant", content=chunk.text)

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=delta,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from chunks."""
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                # Use SSE comment so third-party clients ignore it
                yield f": prefill_progress {chunk.model_dump_json()}\n\n"

            case ErrorChunk():
                error_response = ErrorResponse(
                    error=ErrorInfo(
                        message=chunk.error_message or "Internal server error",
                        type="InternalServerError",
                        code=500,
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case ToolCallChunk():
                last_usage = chunk.usage or last_usage

                tool_call_deltas = [
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                ]
                tool_response = ChatCompletionResponse(
                    id=command_id,
                    created=int(time.time()),
                    model=chunk.model,
                    choices=[
                        StreamingChoiceResponse(
                            index=0,
                            delta=ChatCompletionMessage(
                                role="assistant",
                                tool_calls=tool_call_deltas,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=last_usage,
                )
                yield f"data: {tool_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case TokenChunk():
                last_usage = chunk.usage or last_usage

                chunk_response = chunk_to_response(chunk, command_id)
                if chunk.finish_reason is not None:
                    chunk_response = chunk_response.model_copy(
                        update={"usage": last_usage}
                    )
                yield f"data: {chunk_response.model_dump_json()}\n\n"

                if chunk.finish_reason is not None:
                    yield "data: [DONE]\n\n"


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                error_message = chunk.error_message or "Internal server error"
                break

            case TokenChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.logprob is not None:
                    logprobs_content.append(
                        LogprobsContentItem(
                            token=chunk.text,
                            logprob=chunk.logprob,
                            top_logprobs=chunk.top_logprobs or [],
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            case ToolCallChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                tool_calls.extend(
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                )
                finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts) if thinking_parts else None
    assert model is not None

    yield ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                    reasoning_content=combined_thinking,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=last_usage,
    ).model_dump_json()
    return
