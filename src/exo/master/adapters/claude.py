"""Claude Messages API adapter for converting requests/responses."""

from collections.abc import AsyncGenerator

from exo.shared.types.api import (
    ChatCompletionMessage,
    FinishReason,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.claude_api import (
    ClaudeContentBlockDeltaEvent,
    ClaudeContentBlockStartEvent,
    ClaudeContentBlockStopEvent,
    ClaudeMessageDelta,
    ClaudeMessageDeltaEvent,
    ClaudeMessageDeltaUsage,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessageStart,
    ClaudeMessageStartEvent,
    ClaudeMessageStopEvent,
    ClaudeStopReason,
    ClaudeTextBlock,
    ClaudeTextDelta,
    ClaudeUsage,
)
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import ChatCompletionTaskParams


def finish_reason_to_claude_stop_reason(
    finish_reason: FinishReason | None,
) -> ClaudeStopReason | None:
    """Map OpenAI finish_reason to Claude stop_reason."""
    if finish_reason is None:
        return None
    mapping: dict[FinishReason, ClaudeStopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    return mapping.get(finish_reason, "end_turn")


def claude_request_to_chat_params(
    request: ClaudeMessagesRequest,
) -> ChatCompletionTaskParams:
    """Convert Claude Messages API request to internal ChatCompletionTaskParams."""
    messages: list[ChatCompletionMessage] = []

    # Add system message if present
    if request.system:
        if isinstance(request.system, str):
            messages.append(
                ChatCompletionMessage(role="system", content=request.system)
            )
        else:
            # List of text blocks
            system_text = "".join(block.text for block in request.system)
            messages.append(ChatCompletionMessage(role="system", content=system_text))

    # Convert messages
    for msg in request.messages:
        content: str
        if isinstance(msg.content, str):
            content = msg.content
        else:
            # Concatenate text blocks (images not supported for MVP)
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, ClaudeTextBlock):
                    text_parts.append(block.text)
            content = "".join(text_parts)

        messages.append(ChatCompletionMessage(role=msg.role, content=content))

    return ChatCompletionTaskParams(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop_sequences,
        stream=request.stream,
    )


async def collect_claude_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> ClaudeMessagesResponse:
    """Collect all token chunks and return a single ClaudeMessagesResponse."""
    text_parts: list[str] = []
    stop_reason: ClaudeStopReason | None = None
    last_stats = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if chunk.finish_reason == "error":
            error_message = chunk.error_message or "Internal server error"
            break

        text_parts.append(chunk.text)
        last_stats = chunk.stats or last_stats

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)

    # Use actual usage data from stats if available
    input_tokens = last_stats.prompt_tokens if last_stats else 0
    output_tokens = last_stats.generation_tokens if last_stats else 0

    return ClaudeMessagesResponse(
        id=f"msg_{command_id}",
        model=model,
        content=[ClaudeTextBlock(text=combined_text)],
        stop_reason=stop_reason,
        usage=ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


async def generate_claude_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate Claude Messages API streaming events from TokenChunks."""
    # Initial message_start event
    initial_message = ClaudeMessageStart(
        id=f"msg_{command_id}",
        model=model,
        content=[],
        stop_reason=None,
        usage=ClaudeUsage(input_tokens=0, output_tokens=0),
    )
    start_event = ClaudeMessageStartEvent(message=initial_message)
    yield f"event: message_start\ndata: {start_event.model_dump_json()}\n\n"

    # content_block_start
    block_start = ClaudeContentBlockStartEvent(
        index=0, content_block=ClaudeTextBlock(text="")
    )
    yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

    output_tokens = 0
    stop_reason: ClaudeStopReason | None = None
    last_stats = None

    async for chunk in chunk_stream:
        output_tokens += 1  # Count each chunk as one token
        last_stats = chunk.stats or last_stats

        # content_block_delta
        delta_event = ClaudeContentBlockDeltaEvent(
            index=0,
            delta=ClaudeTextDelta(text=chunk.text),
        )
        yield f"event: content_block_delta\ndata: {delta_event.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    # Use actual token count from stats if available
    if last_stats is not None:
        output_tokens = last_stats.generation_tokens

    # content_block_stop
    block_stop = ClaudeContentBlockStopEvent(index=0)
    yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

    # message_delta
    message_delta = ClaudeMessageDeltaEvent(
        delta=ClaudeMessageDelta(stop_reason=stop_reason),
        usage=ClaudeMessageDeltaUsage(output_tokens=output_tokens),
    )
    yield f"event: message_delta\ndata: {message_delta.model_dump_json()}\n\n"

    # message_stop
    message_stop = ClaudeMessageStopEvent()
    yield f"event: message_stop\ndata: {message_stop.model_dump_json()}\n\n"
