"""OpenAI Responses API adapter for converting requests/responses."""

from collections.abc import AsyncGenerator
from uuid import uuid4

from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.openai_responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseInProgressEvent,
    ResponseInputTextPart,
    ResponseItem,
    ResponseMessageItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _extract_content(content: str | list[ResponseInputTextPart]) -> str:
    """Extract plain text from a content field that may be a string or list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def responses_request_to_text_generation(
    request: ResponsesRequest,
) -> TextGenerationTaskParams:
    input_value: str | list[InputMessage]
    if isinstance(request.input, str):
        input_value = request.input
    else:
        input_value = [
            InputMessage(role=msg.role, content=_extract_content(msg.content))
            for msg in request.input
        ]
    return TextGenerationTaskParams(
        model=request.model,
        input=input_value,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=request.tools,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        chat_template_messages=request.chat_template_messages,
    )


async def collect_responses_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> ResponsesResponse:
    """Collect all token chunks and return a single ResponsesResponse."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
    accumulated_text = ""
    function_call_items: list[ResponseFunctionCallItem] = []
    last_stats = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        if isinstance(chunk, ToolCallChunk):
            for tool in chunk.tool_calls:
                function_call_items.append(
                    ResponseFunctionCallItem(
                        id=f"fc_{uuid4().hex[:24]}",
                        call_id=f"call_{uuid4().hex[:24]}",
                        name=tool.name,
                        arguments=tool.arguments,
                    )
                )
            last_stats = chunk.stats or last_stats
            continue

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

    output: list[ResponseItem] = [
        ResponseMessageItem(
            id=item_id,
            content=[ResponseOutputText(text=accumulated_text)],
            status="completed",
        )
    ]
    output.extend(function_call_items)

    return ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=output,
        output_text=accumulated_text,
        usage=usage,
    )


async def generate_responses_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
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
    function_call_items: list[ResponseFunctionCallItem] = []
    last_stats = None
    next_output_index = 1  # message item is at 0

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            break

        if isinstance(chunk, ToolCallChunk):
            last_stats = chunk.stats or last_stats
            for tool in chunk.tool_calls:
                fc_id = f"fc_{uuid4().hex[:24]}"
                call_id = f"call_{uuid4().hex[:24]}"

                # response.output_item.added for function_call
                fc_item = ResponseFunctionCallItem(
                    id=fc_id,
                    call_id=call_id,
                    name=tool.name,
                    arguments="",
                    status="in_progress",
                )
                fc_added = ResponseOutputItemAddedEvent(
                    output_index=next_output_index, item=fc_item
                )
                yield f"event: response.output_item.added\ndata: {fc_added.model_dump_json()}\n\n"

                # response.function_call_arguments.delta
                args_delta = ResponseFunctionCallArgumentsDeltaEvent(
                    item_id=fc_id,
                    output_index=next_output_index,
                    delta=tool.arguments,
                )
                yield f"event: response.function_call_arguments.delta\ndata: {args_delta.model_dump_json()}\n\n"

                # response.function_call_arguments.done
                args_done = ResponseFunctionCallArgumentsDoneEvent(
                    item_id=fc_id,
                    output_index=next_output_index,
                    name=tool.name,
                    arguments=tool.arguments,
                )
                yield f"event: response.function_call_arguments.done\ndata: {args_done.model_dump_json()}\n\n"

                # response.output_item.done
                fc_done_item = ResponseFunctionCallItem(
                    id=fc_id,
                    call_id=call_id,
                    name=tool.name,
                    arguments=tool.arguments,
                    status="completed",
                )
                fc_item_done = ResponseOutputItemDoneEvent(
                    output_index=next_output_index, item=fc_done_item
                )
                yield f"event: response.output_item.done\ndata: {fc_item_done.model_dump_json()}\n\n"

                function_call_items.append(fc_done_item)
                next_output_index += 1
            continue

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
    final_message_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text=accumulated_text)],
        status="completed",
    )
    item_done = ResponseOutputItemDoneEvent(output_index=0, item=final_message_item)
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
    output: list[ResponseItem] = [final_message_item]
    output.extend(function_call_items)
    final_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=output,
        output_text=accumulated_text,
        usage=usage,
    )
    completed_event = ResponseCompletedEvent(response=final_response)
    yield f"event: response.completed\ndata: {completed_event.model_dump_json()}\n\n"
