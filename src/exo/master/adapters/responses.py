"""OpenAI Responses API adapter for converting requests/responses."""

from collections.abc import AsyncGenerator
from itertools import count
from typing import Any

from exo.shared.types.api import Usage
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
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
    ResponsesRequest,
    ResponsesResponse,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


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
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ResponsesResponse."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
    accumulated_text = ""
    function_call_items: list[ResponseFunctionCallItem] = []
    last_usage: Usage | None = None
    error_message: str | None = None

    async for chunk in chunk_stream:
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

    output: list[ResponseItem] = [
        ResponseMessageItem(
            id=item_id,
            content=[ResponseOutputText(text=accumulated_text)],
            status="completed",
        )
    ]
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
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate OpenAI Responses API streaming events from TokenChunks."""
    response_id = f"resp_{command_id}"
    item_id = f"item_{command_id}"
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
    yield f"event: response.created\ndata: {created_event.model_dump_json()}\n\n"

    # response.in_progress
    in_progress_event = ResponseInProgressEvent(
        sequence_number=next(seq), response=initial_response
    )
    yield f"event: response.in_progress\ndata: {in_progress_event.model_dump_json()}\n\n"

    # response.output_item.added
    initial_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text="")],
        status="in_progress",
    )
    item_added = ResponseOutputItemAddedEvent(
        sequence_number=next(seq), output_index=0, item=initial_item
    )
    yield f"event: response.output_item.added\ndata: {item_added.model_dump_json()}\n\n"

    # response.content_part.added
    initial_part = ResponseOutputText(text="")
    part_added = ResponseContentPartAddedEvent(
        sequence_number=next(seq),
        item_id=item_id,
        output_index=0,
        content_index=0,
        part=initial_part,
    )
    yield f"event: response.content_part.added\ndata: {part_added.model_dump_json()}\n\n"

    accumulated_text = ""
    function_call_items: list[ResponseFunctionCallItem] = []
    last_usage: Usage | None = None
    next_output_index = 1  # message item is at 0

    async for chunk in chunk_stream:
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
                yield f"event: response.output_item.added\ndata: {fc_added.model_dump_json()}\n\n"

                # response.function_call_arguments.delta
                args_delta = ResponseFunctionCallArgumentsDeltaEvent(
                    sequence_number=next(seq),
                    item_id=fc_id,
                    output_index=next_output_index,
                    delta=tool.arguments,
                )
                yield f"event: response.function_call_arguments.delta\ndata: {args_delta.model_dump_json()}\n\n"

                # response.function_call_arguments.done
                args_done = ResponseFunctionCallArgumentsDoneEvent(
                    sequence_number=next(seq),
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
                    sequence_number=next(seq),
                    output_index=next_output_index,
                    item=fc_done_item,
                )
                yield f"event: response.output_item.done\ndata: {fc_item_done.model_dump_json()}\n\n"

                function_call_items.append(fc_done_item)
                next_output_index += 1
            continue

        accumulated_text += chunk.text

        # response.output_text.delta
        delta_event = ResponseTextDeltaEvent(
            sequence_number=next(seq),
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=chunk.text,
        )
        yield f"event: response.output_text.delta\ndata: {delta_event.model_dump_json()}\n\n"

    # response.output_text.done
    text_done = ResponseTextDoneEvent(
        sequence_number=next(seq),
        item_id=item_id,
        output_index=0,
        content_index=0,
        text=accumulated_text,
    )
    yield f"event: response.output_text.done\ndata: {text_done.model_dump_json()}\n\n"

    # response.content_part.done
    final_part = ResponseOutputText(text=accumulated_text)
    part_done = ResponseContentPartDoneEvent(
        sequence_number=next(seq),
        item_id=item_id,
        output_index=0,
        content_index=0,
        part=final_part,
    )
    yield f"event: response.content_part.done\ndata: {part_done.model_dump_json()}\n\n"

    # response.output_item.done
    final_message_item = ResponseMessageItem(
        id=item_id,
        content=[ResponseOutputText(text=accumulated_text)],
        status="completed",
    )
    item_done = ResponseOutputItemDoneEvent(
        sequence_number=next(seq), output_index=0, item=final_message_item
    )
    yield f"event: response.output_item.done\ndata: {item_done.model_dump_json()}\n\n"

    # Create usage from usage data if available
    usage = None
    if last_usage is not None:
        usage = ResponseUsage(
            input_tokens=last_usage.prompt_tokens,
            output_tokens=last_usage.completion_tokens,
            total_tokens=last_usage.total_tokens,
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
    completed_event = ResponseCompletedEvent(
        sequence_number=next(seq), response=final_response
    )
    yield f"event: response.completed\ndata: {completed_event.model_dump_json()}\n\n"
