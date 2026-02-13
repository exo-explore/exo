"""Claude Messages API adapter for converting requests/responses."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from exo.shared.types.api import FinishReason
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.claude_api import (
    ClaudeContentBlock,
    ClaudeContentBlockDeltaEvent,
    ClaudeContentBlockStartEvent,
    ClaudeContentBlockStopEvent,
    ClaudeInputJsonDelta,
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
    ClaudeToolResultBlock,
    ClaudeToolUseBlock,
    ClaudeUsage,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


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


def _extract_tool_result_text(block: ClaudeToolResultBlock) -> str:
    """Extract plain text from a tool_result content field."""
    if block.content is None:
        return ""
    if isinstance(block.content, str):
        return block.content
    return "".join(sub_block.text for sub_block in block.content)


def claude_request_to_text_generation(
    request: ClaudeMessagesRequest,
) -> TextGenerationTaskParams:
    # Handle system message
    instructions: str | None = None
    chat_template_messages: list[dict[str, Any]] = []

    if request.system:
        if isinstance(request.system, str):
            instructions = request.system
        else:
            instructions = "".join(block.text for block in request.system)
        chat_template_messages.append({"role": "system", "content": instructions})

    # Convert messages to input
    input_messages: list[InputMessage] = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            input_messages.append(InputMessage(role=msg.role, content=msg.content))
            chat_template_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Process structured content blocks
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[ClaudeToolResultBlock] = []

        for block in msg.content:
            if isinstance(block, ClaudeTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ClaudeToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            elif isinstance(block, ClaudeToolResultBlock):
                tool_results.append(block)

        content = "".join(text_parts)

        # Build InputMessage from text content
        if msg.role in ("user", "assistant"):
            input_messages.append(InputMessage(role=msg.role, content=content))

        # Build chat_template_messages preserving tool structure
        if tool_calls:
            chat_template_messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
        elif tool_results:
            for tr in tool_results:
                chat_template_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": _extract_tool_result_text(tr),
                    }
                )
        else:
            chat_template_messages.append({"role": msg.role, "content": content})

    # Convert Claude tool definitions to OpenAI-style function tools
    tools: list[dict[str, Any]] | None = None
    if request.tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]

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
        stop=request.stop_sequences,
        stream=request.stream,
        tools=tools,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
    )


async def collect_claude_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> ClaudeMessagesResponse:
    """Collect all token chunks and return a single ClaudeMessagesResponse."""
    text_parts: list[str] = []
    tool_use_blocks: list[ClaudeToolUseBlock] = []
    stop_reason: ClaudeStopReason | None = None
    last_stats = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        if isinstance(chunk, ToolCallChunk):
            for tool in chunk.tool_calls:
                tool_use_blocks.append(
                    ClaudeToolUseBlock(
                        id=f"toolu_{tool.id}",
                        name=tool.name,
                        input=json.loads(tool.arguments),  # pyright: ignore[reportAny]
                    )
                )
            last_stats = chunk.stats or last_stats
            stop_reason = "tool_use"
            continue

        text_parts.append(chunk.text)
        last_stats = chunk.stats or last_stats

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)

    # Build content blocks
    content: list[ClaudeContentBlock] = []
    if combined_text:
        content.append(ClaudeTextBlock(text=combined_text))
    content.extend(tool_use_blocks)

    # If no content at all, include empty text block
    if not content:
        content.append(ClaudeTextBlock(text=""))

    # Use actual usage data from stats if available
    input_tokens = last_stats.prompt_tokens if last_stats else 0
    output_tokens = last_stats.generation_tokens if last_stats else 0

    return ClaudeMessagesResponse(
        id=f"msg_{command_id}",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


async def generate_claude_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
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

    # content_block_start for text block at index 0
    block_start = ClaudeContentBlockStartEvent(
        index=0, content_block=ClaudeTextBlock(text="")
    )
    yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

    output_tokens = 0
    stop_reason: ClaudeStopReason | None = None
    last_stats = None
    next_block_index = 1  # text block is 0, tool blocks start at 1

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            # Close text block and bail
            break

        if isinstance(chunk, ToolCallChunk):
            last_stats = chunk.stats or last_stats
            stop_reason = "tool_use"

            # Emit tool_use content blocks
            for tool in chunk.tool_calls:
                tool_id = f"toolu_{tool.id}"
                tool_input_json = tool.arguments

                # content_block_start for tool_use
                tool_block_start = ClaudeContentBlockStartEvent(
                    index=next_block_index,
                    content_block=ClaudeToolUseBlock(
                        id=tool_id, name=tool.name, input={}
                    ),
                )
                yield f"event: content_block_start\ndata: {tool_block_start.model_dump_json()}\n\n"

                # content_block_delta with input_json_delta
                tool_delta_event = ClaudeContentBlockDeltaEvent(
                    index=next_block_index,
                    delta=ClaudeInputJsonDelta(partial_json=tool_input_json),
                )
                yield f"event: content_block_delta\ndata: {tool_delta_event.model_dump_json()}\n\n"

                # content_block_stop
                tool_block_stop = ClaudeContentBlockStopEvent(index=next_block_index)
                yield f"event: content_block_stop\ndata: {tool_block_stop.model_dump_json()}\n\n"

                next_block_index += 1
            continue

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

    # content_block_stop for text block
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
