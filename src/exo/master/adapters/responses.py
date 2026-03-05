"""OpenAI Responses API adapter for converting requests/responses."""

from collections.abc import AsyncGenerator
from itertools import count
from typing import Any

from exo.shared.types.api import Usage
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.openai_responses import (
    FunctionCallInputItem,
    ResponseCompletedEvent,
    ResponseContentPart,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseInProgressEvent,
    ResponseInputMessage,
    ResponseItem,
    ResponseMessageItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryText,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _format_sse(event: ResponsesStreamEvent) -> str:
    """Format a streaming event as an SSE message."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _extract_content(content: str | list[ResponseContentPart]) -> str:
    """Extract plain text from a content field that may be a string or list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def responses_request_to_text_generation(
    request: ResponsesRequest,
) -> TextGenerationTaskParams:
    input_value: list[InputMessage]
    built_chat_template: list[dict[str, Any]] | None = None
    if isinstance(request.input, str):
        input_value = [InputMessage(role="user", content=request.input)]
    else:
        input_messages: list[InputMessage] = []
        chat_template_messages: list[dict[str, Any]] = []

        if request.instructions is not None:
            chat_template_messages.append(
                {"role": "system", "content": request.instructions}
            )

        for item in request.input:
            if isinstance(item, ResponseInputMessage):
                content = _extract_content(item.content)
                if item.role in ("user", "assistant", "developer"):
                    input_messages.append(InputMessage(role=item.role, content=content))
                if item.role == "system":
                    chat_template_messages.append(
                        {"role": "system", "content": content}
                    )
                else:
                    chat_template_messages.append(
                        {"role": item.role, "content": content}
                    )
            elif isinstance(item, FunctionCallInputItem):
                chat_template_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": item.arguments,
                                },
                            }
                        ],
                    }
                )
            else:
                chat_template_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.output,
                    }
                )

        input_value = (
            input_messages
            if input_messages
            else [InputMessage(role="user", content="")]
        )
        built_chat_template = chat_template_messages if chat_template_messages else None

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
        chat_template_messages=built_chat_template or request.chat_template_messages,
    )


async def collect_responses_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ResponsesResponse."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
    reasoning_id = f"rs_{command_id}"
    accumulated_text = ""
    thinking_parts: list[str] = []
    function_call_items: list[ResponseFunctionCallItem] = []
    last_usage: Usage | None = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, PrefillProgressChunk):
            continue

        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
            for tool in chunk.tool_calls:
                function_call_items.append(
                    ResponseFunctionCallItem(
                        id=tool.id,
                        call_id=tool.id,
                        name=tool.name,
                        arguments=tool.arguments,
                    )
                )
            continue

        if chunk.is_thinking:
            thinking_parts.append(chunk.text)
            continue

        accumulated_text += chunk.text

    if error_message is not None:
        raise ValueError(error_message)

    # Create usage from usage data if available
    usage = None
    if last_usage is not None:
        usage = ResponseUsage(
            input_tokens=last_usage.prompt_tokens,
            output_tokens=last_usage.completion_tokens,
            total_tokens=last_usage.total_tokens,
        )

    output: list[ResponseItem] = []
    if thinking_parts:
        output.append(
            ResponseReasoningItem(
                id=reasoning_id,
                summary=[ResponseReasoningSummaryText(text="".join(thinking_parts))],
            )
        )
    output.append(
        ResponseMessageItem(
            id=item_id,
            content=[ResponseOutputText(text=accumulated_text)],
            status="completed",
        )
    )
    output.extend(function_call_items)

    yield ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=output,
        output_text=accumulated_text,
        usage=usage,
    ).model_dump_json()
    return


async def generate_responses_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate OpenAI Responses API streaming events from TokenChunks."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
    reasoning_id = f"rs_{command_id}"
    seq = count(1)

    # response.created
    initial_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="in_progress",
        output=[],
        output_text="",
    )
    created_event = ResponseCreatedEvent(
        sequence_number=next(seq), response=initial_response
    )
    yield _format_sse(created_event)

    # response.in_progress
    in_progress_event = ResponseInProgressEvent(
        sequence_number=next(seq), response=initial_response
    )
    yield _format_sse(in_progress_event)

    accumulated_text = ""
    accumulated_thinking = ""
    function_call_items: list[ResponseFunctionCallItem] = []
    last_usage: Usage | None = None
    next_output_index = 0

    # Track dynamic block creation
    reasoning_started = False
    reasoning_output_index = -1
    message_started = False
    message_output_index = -1

    async for chunk in chunk_stream:
        if isinstance(chunk, PrefillProgressChunk):
            continue

        if isinstance(chunk, ErrorChunk):
            break

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
            for tool in chunk.tool_calls:
                fc_id = f"fc_{tool.id}"
                call_id = f"call_{tool.id}"

                # response.output_item.added for function_call
                fc_item = ResponseFunctionCallItem(
                    id=fc_id,
                    call_id=call_id,
                    name=tool.name,
                    arguments="",
                    status="in_progress",
                )
                fc_added = ResponseOutputItemAddedEvent(
                    sequence_number=next(seq),
                    output_index=next_output_index,
                    item=fc_item,
                )
                yield _format_sse(fc_added)

                # response.function_call_arguments.delta
                args_delta = ResponseFunctionCallArgumentsDeltaEvent(
                    sequence_number=next(seq),
                    item_id=fc_id,
                    output_index=next_output_index,
                    delta=tool.arguments,
                )
                yield _format_sse(args_delta)

                # response.function_call_arguments.done
                args_done = ResponseFunctionCallArgumentsDoneEvent(
                    sequence_number=next(seq),
                    item_id=fc_id,
                    output_index=next_output_index,
                    name=tool.name,
                    arguments=tool.arguments,
                )
                yield _format_sse(args_done)

                # response.output_item.done
                fc_done_item = ResponseFunctionCallItem(
                    id=fc_id,
                    call_id=call_id,
                    name=tool.name,
                    arguments=tool.arguments,
                    status="completed",
                )
                fc_item_done = ResponseOutputItemDoneEvent(
                    sequence_number=next(seq),
                    output_index=next_output_index,
                    item=fc_done_item,
                )
                yield _format_sse(fc_item_done)

                function_call_items.append(fc_done_item)
                next_output_index += 1
            continue

        if chunk.is_thinking:
            # Start reasoning block on first thinking token
            if not reasoning_started:
                reasoning_started = True
                reasoning_output_index = next_output_index
                next_output_index += 1

                # response.output_item.added for reasoning
                reasoning_item = ResponseReasoningItem(
                    id=reasoning_id,
                    summary=[],
                    status="in_progress",
                )
                rs_added = ResponseOutputItemAddedEvent(
                    sequence_number=next(seq),
                    output_index=reasoning_output_index,
                    item=reasoning_item,
                )
                yield _format_sse(rs_added)

                # response.reasoning_summary_part.added
                part_added = ResponseReasoningSummaryPartAddedEvent(
                    sequence_number=next(seq),
                    item_id=reasoning_id,
                    output_index=reasoning_output_index,
                    summary_index=0,
                    part=ResponseReasoningSummaryText(text=""),
                )
                yield _format_sse(part_added)

            accumulated_thinking += chunk.text

            # response.reasoning_summary_text.delta
            rs_delta = ResponseReasoningSummaryTextDeltaEvent(
                sequence_number=next(seq),
                item_id=reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                delta=chunk.text,
            )
            yield _format_sse(rs_delta)
            continue

        # Close reasoning block when transitioning to text
        if reasoning_started and not message_started:
            # response.reasoning_summary_text.done
            rs_text_done = ResponseReasoningSummaryTextDoneEvent(
                sequence_number=next(seq),
                item_id=reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                text=accumulated_thinking,
            )
            yield _format_sse(rs_text_done)

            # response.reasoning_summary_part.done
            rs_part_done = ResponseReasoningSummaryPartDoneEvent(
                sequence_number=next(seq),
                item_id=reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                part=ResponseReasoningSummaryText(text=accumulated_thinking),
            )
            yield _format_sse(rs_part_done)

            # response.output_item.done for reasoning
            rs_item_done = ResponseOutputItemDoneEvent(
                sequence_number=next(seq),
                output_index=reasoning_output_index,
                item=ResponseReasoningItem(
                    id=reasoning_id,
                    summary=[ResponseReasoningSummaryText(text=accumulated_thinking)],
                ),
            )
            yield _format_sse(rs_item_done)

        # Start message block on first text token
        if not message_started:
            message_started = True
            message_output_index = next_output_index
            next_output_index += 1

            initial_item = ResponseMessageItem(
                id=item_id,
                content=[ResponseOutputText(text="")],
                status="in_progress",
            )
            item_added = ResponseOutputItemAddedEvent(
                sequence_number=next(seq),
                output_index=message_output_index,
                item=initial_item,
            )
            yield _format_sse(item_added)

            initial_part = ResponseOutputText(text="")
            part_added = ResponseContentPartAddedEvent(
                sequence_number=next(seq),
                item_id=item_id,
                output_index=message_output_index,
                content_index=0,
                part=initial_part,
            )
            yield _format_sse(part_added)

        accumulated_text += chunk.text

        # response.output_text.delta
        delta_event = ResponseTextDeltaEvent(
            sequence_number=next(seq),
            item_id=item_id,
            output_index=message_output_index,
            content_index=0,
            delta=chunk.text,
        )
        yield _format_sse(delta_event)

    # Close reasoning block if it was never followed by text
    if reasoning_started and not message_started:
        rs_text_done = ResponseReasoningSummaryTextDoneEvent(
            sequence_number=next(seq),
            item_id=reasoning_id,
            output_index=reasoning_output_index,
            summary_index=0,
            text=accumulated_thinking,
        )
        yield _format_sse(rs_text_done)

        rs_part_done = ResponseReasoningSummaryPartDoneEvent(
            sequence_number=next(seq),
            item_id=reasoning_id,
            output_index=reasoning_output_index,
            summary_index=0,
            part=ResponseReasoningSummaryText(text=accumulated_thinking),
        )
        yield _format_sse(rs_part_done)

        rs_item_done = ResponseOutputItemDoneEvent(
            sequence_number=next(seq),
            output_index=reasoning_output_index,
            item=ResponseReasoningItem(
                id=reasoning_id,
                summary=[ResponseReasoningSummaryText(text=accumulated_thinking)],
            ),
        )
        yield _format_sse(rs_item_done)

    # If no message block was started, create one now (empty text)
    if not message_started:
        message_output_index = next_output_index
        next_output_index += 1

        initial_item = ResponseMessageItem(
            id=item_id,
            content=[ResponseOutputText(text="")],
            status="in_progress",
        )
        item_added = ResponseOutputItemAddedEvent(
            sequence_number=next(seq),
            output_index=message_output_index,
            item=initial_item,
        )
        yield _format_sse(item_added)

        initial_part = ResponseOutputText(text="")
        part_added_evt = ResponseContentPartAddedEvent(
            sequence_number=next(seq),
            item_id=item_id,
            output_index=message_output_index,
            content_index=0,
            part=initial_part,
        )
        yield _format_sse(part_added_evt)

    # response.output_text.done
    text_done = ResponseTextDoneEvent(
        sequence_number=next(seq),
        item_id=item_id,
        output_index=message_output_index,
        content_index=0,
        text=accumulated_text,
    )
    yield _format_sse(text_done)

    # response.content_part.done
    final_part = ResponseOutputText(text=accumulated_text)
    part_done = ResponseContentPartDoneEvent(
        sequence_number=next(seq),
        item_id=item_id,
        output_index=message_output_index,
        content_index=0,
        part=final_part,
    )
    yield _format_sse(part_done)

    # response.output_item.done
    final_message_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text=accumulated_text)],
        status="completed",
    )
    item_done = ResponseOutputItemDoneEvent(
        sequence_number=next(seq),
        output_index=message_output_index,
        item=final_message_item,
    )
    yield _format_sse(item_done)

    # Create usage from usage data if available
    usage = None
    if last_usage is not None:
        usage = ResponseUsage(
            input_tokens=last_usage.prompt_tokens,
            output_tokens=last_usage.completion_tokens,
            total_tokens=last_usage.total_tokens,
        )

    # response.completed
    output: list[ResponseItem] = []
    if reasoning_started:
        output.append(
            ResponseReasoningItem(
                id=reasoning_id,
                summary=[ResponseReasoningSummaryText(text=accumulated_thinking)],
            )
        )
    output.append(final_message_item)
    output.extend(function_call_items)
    final_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=output,
        output_text=accumulated_text,
        usage=usage,
    )
    completed_event = ResponseCompletedEvent(
        sequence_number=next(seq), response=final_response
    )
    yield _format_sse(completed_event)
