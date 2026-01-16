"""Claude Messages API adapter for converting requests/responses."""

from collections.abc import AsyncGenerator

from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
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


def chat_response_to_claude_response(
    response: ChatCompletionResponse,
) -> ClaudeMessagesResponse:
    """Convert internal ChatCompletionResponse to Claude Messages API response."""
    content_text = ""
    stop_reason: ClaudeStopReason | None = None

    if response.choices:
        choice = response.choices[0]
        if isinstance(choice, ChatCompletionChoice) and choice.message.content:
            content_text = (
                choice.message.content
                if isinstance(choice.message.content, str)
                else str(choice.message.content)
            )
        stop_reason = finish_reason_to_claude_stop_reason(choice.finish_reason)

    # Use actual usage data from response if available
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return ClaudeMessagesResponse(
        id=f"msg_{response.id}",
        model=response.model,
        content=[ClaudeTextBlock(text=content_text)],
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
