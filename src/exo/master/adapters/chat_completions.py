"""OpenAI Chat Completions API adapter for converting requests/responses."""

import time
from collections.abc import AsyncGenerator

from loguru import logger

from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageText,
    ChatCompletionResponse,
    ChatCompletionTaskParams,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import PrefillProgressData, StreamEvent, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.openai_responses import ResponseInputMessage, ResponsesRequest


def chat_request_to_internal(request: ChatCompletionTaskParams) -> ResponsesRequest:
    """Convert Chat Completions API request to ResponsesRequest (canonical internal format).

    Extracts system message as instructions, converts messages to input.
    """
    instructions: str | None = None
    input_messages: list[ResponseInputMessage] = []

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
        else:
            # Convert to ResponseInputMessage (only user, assistant, developer roles)
            if msg.role in ("user", "assistant", "developer"):
                input_messages.append(
                    ResponseInputMessage(role=msg.role, content=content)
                )

    return ResponsesRequest(
        model=request.model,
        input=input_messages if input_messages else "",
        instructions=instructions,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        continue_from_prefix=request.continue_from_prefix,
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

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    event_stream: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from StreamEvents.

    Handles both TokenChunks (token generation) and PrefillProgressData (prefill progress).
    """
    try:
        async for event in event_stream:
            if isinstance(event, PrefillProgressData):
                # Send prefill progress as a named SSE event
                progress_json = f'{{"processed":{event.processed_tokens},"total":{event.total_tokens}}}'
                yield f"event: prefill_progress\ndata: {progress_json}\n\n"
                continue

            # TokenChunk handling
            chunk = event
            if chunk.finish_reason == "error":
                error_response = ErrorResponse(
                    error=ErrorInfo(
                        message=chunk.error_message or "Internal server error",
                        type="InternalServerError",
                        code=500,
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                logger.info(f"generate_chat_stream ending (error): {command_id}")
                return

            chunk_response = chunk_to_response(chunk, command_id)
            yield f"data: {chunk_response.model_dump_json()}\n\n"

            if chunk.finish_reason is not None:
                logger.info(
                    f"generate_chat_stream yielding [DONE] for finish_reason={chunk.finish_reason}: {command_id}"
                )
                yield "data: [DONE]\n\n"
                logger.info(f"generate_chat_stream returning: {command_id}")
                return
    finally:
        logger.info(f"generate_chat_stream finally block: {command_id}")


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> ChatCompletionResponse:
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if chunk.finish_reason == "error":
            error_message = chunk.error_message or "Internal server error"
            break

        if model is None:
            model = chunk.model

        text_parts.append(chunk.text)

        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    assert model is not None

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                ),
                finish_reason=finish_reason,
            )
        ],
    )
