"""OpenAI Responses API adapter for converting requests/responses.

ResponsesRequest is the canonical internal format. Responses API is the most featureful,
making it the natural choice for the internal format. All other API formats (Chat
Completions, Claude) are converted TO ResponsesRequest.
"""

from collections.abc import AsyncGenerator

from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.openai_responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputText,
    ResponsesResponse,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)


async def collect_responses_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> ResponsesResponse:
    """Collect all token chunks and return a single ResponsesResponse."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
    accumulated_text = ""
    last_stats = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if chunk.finish_reason == "error":
            error_message = chunk.error_message or "Internal server error"
            break

        accumulated_text += chunk.text
        last_stats = chunk.stats or last_stats

    if error_message is not None:
        raise ValueError(error_message)

    # Create usage from stats if available
    usage = None
    if last_stats is not None:
        usage = ResponseUsage(
            input_tokens=last_stats.prompt_tokens,
            output_tokens=last_stats.generation_tokens,
            total_tokens=last_stats.prompt_tokens + last_stats.generation_tokens,
        )

    output_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text=accumulated_text)],
        status="completed",
    )

    return ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=[output_item],
        output_text=accumulated_text,
        usage=usage,
    )


async def generate_responses_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate OpenAI Responses API streaming events from TokenChunks."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"

    # response.created
    initial_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="in_progress",
        output=[],
        output_text="",
    )
    created_event = ResponseCreatedEvent(response=initial_response)
    yield f"event: response.created\ndata: {created_event.model_dump_json()}\n\n"

    # response.in_progress
    in_progress_event = ResponseInProgressEvent(response=initial_response)
    yield f"event: response.in_progress\ndata: {in_progress_event.model_dump_json()}\n\n"

    # response.output_item.added
    initial_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text="")],
        status="in_progress",
    )
    item_added = ResponseOutputItemAddedEvent(output_index=0, item=initial_item)
    yield f"event: response.output_item.added\ndata: {item_added.model_dump_json()}\n\n"

    # response.content_part.added
    initial_part = ResponseOutputText(text="")
    part_added = ResponseContentPartAddedEvent(
        output_index=0, content_index=0, part=initial_part
    )
    yield f"event: response.content_part.added\ndata: {part_added.model_dump_json()}\n\n"

    accumulated_text = ""
    last_stats = None

    async for chunk in chunk_stream:
        accumulated_text += chunk.text
        last_stats = chunk.stats or last_stats

        # response.output_text.delta
        delta_event = ResponseTextDeltaEvent(
            output_index=0,
            content_index=0,
            delta=chunk.text,
        )
        yield f"event: response.output_text.delta\ndata: {delta_event.model_dump_json()}\n\n"

    # response.output_text.done
    text_done = ResponseTextDoneEvent(
        output_index=0, content_index=0, text=accumulated_text
    )
    yield f"event: response.output_text.done\ndata: {text_done.model_dump_json()}\n\n"

    # response.content_part.done
    final_part = ResponseOutputText(text=accumulated_text)
    part_done = ResponseContentPartDoneEvent(
        output_index=0, content_index=0, part=final_part
    )
    yield f"event: response.content_part.done\ndata: {part_done.model_dump_json()}\n\n"

    # response.output_item.done
    final_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text=accumulated_text)],
        status="completed",
    )
    item_done = ResponseOutputItemDoneEvent(output_index=0, item=final_item)
    yield f"event: response.output_item.done\ndata: {item_done.model_dump_json()}\n\n"

    # Create usage from stats if available
    usage = None
    if last_stats is not None:
        usage = ResponseUsage(
            input_tokens=last_stats.prompt_tokens,
            output_tokens=last_stats.generation_tokens,
            total_tokens=last_stats.prompt_tokens + last_stats.generation_tokens,
        )

    # response.completed
    final_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=[final_item],
        output_text=accumulated_text,
        usage=usage,
    )
    completed_event = ResponseCompletedEvent(response=final_response)
    yield f"event: response.completed\ndata: {completed_event.model_dump_json()}\n\n"
