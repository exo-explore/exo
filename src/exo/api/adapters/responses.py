"""OpenAI Responses API adapter for converting requests/responses."""

import json
from collections.abc import AsyncGenerator
from itertools import count
from typing import Any

from exo.api.adapters.chat_completions import (
    extract_base64_from_data_url,
    fetch_image_url,
)
from exo.api.types import Usage
from exo.api.types.openai_responses import (
    ApplyPatchCallInputItem,
    ApplyPatchCallOutputInputItem,
    CodeInterpreterCallInputItem,
    CompactionInputItem,
    ComputerCallInputItem,
    ComputerCallOutputInputItem,
    CustomToolCallInputItem,
    CustomToolCallOutputInputItem,
    FileSearchCallInputItem,
    FunctionCallInputItem,
    FunctionCallOutputInputItem,
    ImageGenerationCallInputItem,
    InputTokensDetails,
    ItemReferenceInputItem,
    LocalShellCallInputItem,
    LocalShellCallOutputInputItem,
    McpApprovalRequestInputItem,
    McpApprovalResponseInputItem,
    McpCallInputItem,
    McpListToolsInputItem,
    OutputTokensDetails,
    Reasoning,
    ReasoningInputItem,
    ResponseCompletedEvent,
    ResponseContentPart,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseInProgressEvent,
    ResponseInputImagePart,
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
    ShellCallInputItem,
    ShellCallOutputInputItem,
    ToolSearchCallInputItem,
    ToolSearchOutputInputItem,
    WebSearchCallInputItem,
)
from exo.shared.logging import logger
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import (
    Base64Image,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
    resolve_reasoning_params,
)


def _build_response_usage(usage: Usage) -> ResponseUsage:
    """Build a ResponseUsage from the internal Usage type."""
    return ResponseUsage(
        input_tokens=usage.prompt_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.prompt_tokens_details.cached_tokens,
        ),
        output_tokens=usage.completion_tokens,
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=usage.completion_tokens_details.reasoning_tokens,
        ),
        total_tokens=usage.total_tokens,
    )


def _format_sse(event: ResponsesStreamEvent) -> str:
    """Format a streaming event as an SSE message."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _extract_content(content: str | list[ResponseContentPart]) -> str:
    """Extract plain text from a content field that may be a string or list of parts."""
    if isinstance(content, str):
        return content
    return "".join(
        part.text for part in content if not isinstance(part, ResponseInputImagePart)
    )


def _append_tool_call(
    chat_template_messages: list[dict[str, Any]], tool_call: dict[str, Any]
) -> None:
    if chat_template_messages:
        prev = chat_template_messages[-1]
        if prev.get("role") == "assistant" and isinstance(prev.get("content"), str):
            existing: list[dict[str, Any]] | None = prev.get("tool_calls")
            if existing is None:
                prev["tool_calls"] = [tool_call]
            else:
                existing.append(tool_call)
            return
    chat_template_messages.append(
        {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    )


def _custom_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON schema for Responses custom/freeform tools.

    Codex exposes some tools, notably apply_patch, as custom/freeform tools in
    the Responses API. MLX chat templates expect function-style JSON schemas,
    so preserve that freeform input as a single required string argument.
    """
    format_config = tool.get("format")
    description = "Freeform tool input."
    if isinstance(format_config, dict) and isinstance(
        format_config.get("description"), str
    ):
        description = format_config["description"]

    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }


def _normalise_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API tool definition into chat-completions shape."""
    if "function" in tool:
        return tool

    name = tool.get("name", "")
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        parameters = _custom_tool_parameters(tool) if tool.get("type") == "custom" else {}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": parameters,
            **({"strict": tool["strict"]} if "strict" in tool else {}),
        },
    }


async def responses_request_to_text_generation(
    request: ResponsesRequest,
) -> TextGenerationTaskParams:
    input_value: list[InputMessage]
    built_chat_template: list[dict[str, Any]] | None = None
    images: list[Base64Image] = []
    if isinstance(request.input, str):
        input_value = [
            InputMessage(role="user", content=InputMessageContent(request.input))
        ]
    else:
        input_messages: list[InputMessage] = []
        chat_template_messages: list[dict[str, Any]] = []
        has_images = False

        if request.instructions is not None:
            chat_template_messages.append(
                {"role": "system", "content": request.instructions}
            )

        for item in request.input:
            match item:
                case ResponseInputMessage():
                    content = _extract_content(item.content)
                    if isinstance(item.content, list):
                        for part in item.content:
                            if (
                                isinstance(part, ResponseInputImagePart)
                                and part.image_url
                            ):
                                url = part.image_url
                                if url.startswith(("http://", "https://")):
                                    images.append(await fetch_image_url(url))
                                else:
                                    images.append(extract_base64_from_data_url(url))
                                has_images = True
                    if item.role in ("user", "assistant", "developer"):
                        input_messages.append(
                            InputMessage(
                                role=item.role, content=InputMessageContent(content)
                            )
                        )
                    if item.role == "system":
                        chat_template_messages.append(
                            {"role": "system", "content": content}
                        )
                    elif has_images:
                        multimodal: list[dict[str, Any]] = []
                        if isinstance(item.content, list):
                            for part in item.content:
                                if isinstance(part, ResponseInputImagePart):
                                    multimodal.append({"type": "image"})
                                elif hasattr(part, "text"):
                                    multimodal.append(
                                        {"type": "text", "text": part.text}
                                    )
                        chat_template_messages.append(
                            {"role": item.role, "content": multimodal}
                        )
                        has_images = False
                    else:
                        chat_template_messages.append(
                            {"role": item.role, "content": content}
                        )
                case (
                    FunctionCallInputItem()
                    | McpCallInputItem()
                    | CustomToolCallInputItem()
                ):
                    _append_tool_call(
                        chat_template_messages,
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        },
                    )
                case (
                    LocalShellCallInputItem()
                    | ShellCallInputItem()
                    | ComputerCallInputItem()
                ):
                    _append_tool_call(
                        chat_template_messages,
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.type,
                                "arguments": json.dumps(item.action),
                            },
                        },
                    )
                case ApplyPatchCallInputItem():
                    _append_tool_call(
                        chat_template_messages,
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": json.dumps({"input": item.patch}),
                            },
                        },
                    )
                case (
                    WebSearchCallInputItem()
                    | FileSearchCallInputItem()
                    | CodeInterpreterCallInputItem()
                    | ImageGenerationCallInputItem()
                    | ToolSearchCallInputItem()
                ):
                    args: dict[str, Any] = {}
                    if isinstance(item, WebSearchCallInputItem):
                        args = {"query": item.query}
                    elif isinstance(item, FileSearchCallInputItem):
                        args = {"queries": item.queries}
                    elif isinstance(item, CodeInterpreterCallInputItem):
                        args = {"code": item.code}
                    elif isinstance(item, ImageGenerationCallInputItem):
                        args = {"prompt": item.prompt}
                    else:
                        args = {"query": item.query}
                    _append_tool_call(
                        chat_template_messages,
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.type,
                                "arguments": json.dumps(args),
                            },
                        },
                    )
                case (
                    FunctionCallOutputInputItem()
                    | LocalShellCallOutputInputItem()
                    | ShellCallOutputInputItem()
                    | ApplyPatchCallOutputInputItem()
                    | ComputerCallOutputInputItem()
                    | CustomToolCallOutputInputItem()
                    | ToolSearchOutputInputItem()
                ):
                    output = (
                        item.output
                        if isinstance(item.output, str)
                        else json.dumps(item.output)
                    )
                    chat_template_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item.call_id,
                            "content": output,
                        }
                    )
                case ReasoningInputItem():
                    # Reasoning items are internal assistant state. Replaying
                    # them as assistant messages can separate an assistant
                    # tool_call from its tool output in chat-template history.
                    continue
                case CompactionInputItem():
                    if item.summary:
                        chat_template_messages.append(
                            {"role": "system", "content": item.summary}
                        )
                case McpListToolsInputItem():
                    tools_desc = ", ".join(t.get("name", "") for t in item.tools)
                    if tools_desc:
                        chat_template_messages.append(
                            {
                                "role": "system",
                                "content": f"Available MCP tools ({item.server_label}): {tools_desc}",
                            }
                        )
                case McpApprovalRequestInputItem():
                    _append_tool_call(
                        chat_template_messages,
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        },
                    )
                case McpApprovalResponseInputItem():
                    chat_template_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item.approval_request_id,
                            "content": f"{'Approved' if item.approve else 'Denied'}{': ' + item.reason if item.reason else ''}",
                        }
                    )
                case ItemReferenceInputItem():
                    logger.info("Cannot handle ItemReferenceInputItem, skipping")

        input_value = (
            input_messages
            if input_messages
            else [InputMessage(role="user", content=InputMessageContent(""))]
        )
        built_chat_template = chat_template_messages if chat_template_messages else None

    effort_from_reasoning = request.reasoning.effort if request.reasoning else None
    resolved_effort, resolved_thinking = resolve_reasoning_params(
        effort_from_reasoning, request.enable_thinking
    )

    # The responses API often does not provide tool args nested under a "function" field.
    # Since we follow the chat completions format of tools in the backend (for MLX chat templates)
    # we need to normalise to this format.
    normalised_tools: list[dict[str, Any]] | None = None
    if request.tools:
        normalised_tools = [_normalise_responses_tool(tool) for tool in request.tools]

    return TextGenerationTaskParams(
        model=request.model,
        input=input_value,
        instructions=InputMessageContent(request.instructions)
        if request.instructions
        else None,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=normalised_tools,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        chat_template_messages=built_chat_template or request.chat_template_messages,
        reasoning_effort=resolved_effort,
        enable_thinking=resolved_thinking,
        images=images,
    )


async def collect_responses_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
    reasoning: Reasoning | None = None,
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
    usage = _build_response_usage(last_usage) if last_usage is not None else None

    output: list[ResponseItem] = []
    if thinking_parts:
        output.append(
            ResponseReasoningItem(
                id=reasoning_id,
                summary=[ResponseReasoningSummaryText(text="".join(thinking_parts))],
            )
        )
    if accumulated_text and not function_call_items:
        output.append(
            ResponseMessageItem(
                id=item_id,
                content=[ResponseOutputText(text=accumulated_text)],
                status="completed",
            )
        )
    elif not function_call_items:
        output.append(
            ResponseMessageItem(
                id=item_id,
                content=[ResponseOutputText(text="")],
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
        reasoning=reasoning,
    ).model_dump_json()
    return


async def generate_responses_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
    reasoning: Reasoning | None = None,
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
        reasoning=reasoning,
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
    reasoning_closed = False
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
        if reasoning_started and not reasoning_closed:
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
            reasoning_closed = True

        # Start message block on first visible text token.
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

        delta_event = ResponseTextDeltaEvent(
            sequence_number=next(seq),
            item_id=item_id,
            output_index=message_output_index,
            content_index=0,
            delta=chunk.text,
        )
        yield _format_sse(delta_event)

    # Close reasoning block if it was never followed by text
    if reasoning_started and not reasoning_closed:
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
        reasoning_closed = True

    # If this response has tool calls, do not also emit a pre-tool assistant
    # message. Codex replays streamed items in a way that can place that message
    # between the assistant tool_call and the tool output, which breaks local
    # chat-template continuations.
    tool_call_response = bool(function_call_items)
    if not message_started and tool_call_response:
        usage = _build_response_usage(last_usage) if last_usage is not None else None
        output: list[ResponseItem] = []
        if reasoning_started:
            output.append(
                ResponseReasoningItem(
                    id=reasoning_id,
                    summary=[ResponseReasoningSummaryText(text=accumulated_thinking)],
                )
            )
        output.extend(function_call_items)
        final_response = ResponsesResponse(
            id=response_id,
            model=model,
            status="completed",
            output=output,
            output_text=accumulated_text,
            usage=usage,
            reasoning=reasoning,
        )
        completed_event = ResponseCompletedEvent(
            sequence_number=next(seq), response=final_response
        )
        yield _format_sse(completed_event)
        return

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
    usage = _build_response_usage(last_usage) if last_usage is not None else None

    # response.completed
    output: list[ResponseItem] = []
    if reasoning_started:
        output.append(
            ResponseReasoningItem(
                id=reasoning_id,
                summary=[ResponseReasoningSummaryText(text=accumulated_thinking)],
            )
        )
    if not function_call_items:
        output.append(final_message_item)
    output.extend(function_call_items)
    final_response = ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=output,
        output_text=accumulated_text,
        usage=usage,
        reasoning=reasoning,
    )
    completed_event = ResponseCompletedEvent(
        sequence_number=next(seq), response=final_response
    )
    yield _format_sse(completed_event)
