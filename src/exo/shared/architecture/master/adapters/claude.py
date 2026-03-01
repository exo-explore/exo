"""Claude Messages API adapter for converting requests/responses."""

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from exo.shared.types.api import FinishReason, Usage
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
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
    ClaudeThinkingBlock,
    ClaudeThinkingDelta,
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


# Matches "x-anthropic-billing-header: ...;" (with optional trailing newline)
# or similar telemetry headers that change every request and break KV prefix caching.
_VOLATILE_HEADER_RE = re.compile(r"^x-anthropic-[^\n]*;\n?", re.MULTILINE)


def _strip_volatile_headers(text: str) -> str:
    """Remove Anthropic billing/telemetry headers from system prompt text.

    Claude Code prepends headers like 'x-anthropic-billing-header: cc_version=...;
    cc_entrypoint=...; cch=...;' that contain per-request content hashes. These
    change every request and break KV prefix caching (the prefix diverges at ~20
    tokens instead of matching thousands of conversation tokens).
    """
    return _VOLATILE_HEADER_RE.sub("", text)


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

        instructions = _strip_volatile_headers(instructions)
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
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[ClaudeToolResultBlock] = []

        for block in msg.content:
            if isinstance(block, ClaudeTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ClaudeThinkingBlock):
                thinking_parts.append(block.thinking)
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
        reasoning_content = "".join(thinking_parts) if thinking_parts else None

        # Build InputMessage from text content
        if msg.role in ("user", "assistant"):
            input_messages.append(InputMessage(role=msg.role, content=content))

        # Build chat_template_messages preserving tool structure
        if tool_calls:
            chat_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            if reasoning_content:
                chat_msg["reasoning_content"] = reasoning_content
            chat_template_messages.append(chat_msg)
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
            chat_msg = {"role": msg.role, "content": content}
            if reasoning_content:
                chat_msg["reasoning_content"] = reasoning_content
            chat_template_messages.append(chat_msg)

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

    enable_thinking: bool | None = None
    if request.thinking is not None:
        enable_thinking = request.thinking.type in ("enabled", "adaptive")

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
        enable_thinking=enable_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
    )


async def collect_claude_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ClaudeMessagesResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_use_blocks: list[ClaudeToolUseBlock] = []
    stop_reason: ClaudeStopReason | None = None
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
                tool_use_blocks.append(
                    ClaudeToolUseBlock(
                        id=f"toolu_{tool.id}",
                        name=tool.name,
                        input=json.loads(tool.arguments),  # pyright: ignore[reportAny]
                    )
                )
            stop_reason = "tool_use"
            continue

        if chunk.is_thinking:
            thinking_parts.append(chunk.text)
        else:
            text_parts.append(chunk.text)

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts)

    # Build content blocks
    content: list[ClaudeContentBlock] = []
    if combined_thinking:
        content.append(ClaudeThinkingBlock(thinking=combined_thinking))
    if combined_text:
        content.append(ClaudeTextBlock(text=combined_text))
    content.extend(tool_use_blocks)

    # If no content at all, include empty text block
    if not content:
        content.append(ClaudeTextBlock(text=""))

    # Use actual usage data if available
    input_tokens = last_usage.prompt_tokens if last_usage else 0
    output_tokens = last_usage.completion_tokens if last_usage else 0

    yield ClaudeMessagesResponse(
        id=f"msg_{command_id}",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    ).model_dump_json()
    return


async def generate_claude_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
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

    output_tokens = 0
    stop_reason: ClaudeStopReason | None = None
    last_usage: Usage | None = None
    next_block_index = 0

    # Track whether we've started thinking/text blocks
    thinking_block_started = False
    thinking_block_index = -1
    text_block_started = False
    text_block_index = -1

    async for chunk in chunk_stream:
        if isinstance(chunk, PrefillProgressChunk):
            continue

        if isinstance(chunk, ErrorChunk):
            # Close text block and bail
            break

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
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

        if chunk.is_thinking:
            # Start thinking block on first thinking token
            if not thinking_block_started:
                thinking_block_started = True
                thinking_block_index = next_block_index
                next_block_index += 1
                block_start = ClaudeContentBlockStartEvent(
                    index=thinking_block_index,
                    content_block=ClaudeThinkingBlock(thinking=""),
                )
                yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

            delta_event = ClaudeContentBlockDeltaEvent(
                index=thinking_block_index,
                delta=ClaudeThinkingDelta(thinking=chunk.text),
            )
            yield f"event: content_block_delta\ndata: {delta_event.model_dump_json()}\n\n"
        else:
            # Close thinking block when transitioning to text
            if thinking_block_started and text_block_index == -1:
                block_stop = ClaudeContentBlockStopEvent(index=thinking_block_index)
                yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

            # Start text block on first text token
            if not text_block_started:
                text_block_started = True
                text_block_index = next_block_index
                next_block_index += 1
                block_start = ClaudeContentBlockStartEvent(
                    index=text_block_index,
                    content_block=ClaudeTextBlock(text=""),
                )
                yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

            delta_event = ClaudeContentBlockDeltaEvent(
                index=text_block_index,
                delta=ClaudeTextDelta(text=chunk.text),
            )
            yield f"event: content_block_delta\ndata: {delta_event.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    # Use actual token count from usage if available
    if last_usage is not None:
        output_tokens = last_usage.completion_tokens

    # Close any open blocks
    if thinking_block_started and text_block_index == -1:
        block_stop = ClaudeContentBlockStopEvent(index=thinking_block_index)
        yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

    if text_block_started:
        block_stop = ClaudeContentBlockStopEvent(index=text_block_index)
        yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

    if not thinking_block_started and not text_block_started:
        empty_start = ClaudeContentBlockStartEvent(
            index=0, content_block=ClaudeTextBlock(text="")
        )
        yield f"event: content_block_start\ndata: {empty_start.model_dump_json()}\n\n"
        empty_stop = ClaudeContentBlockStopEvent(index=0)
        yield f"event: content_block_stop\ndata: {empty_stop.model_dump_json()}\n\n"

    # message_delta
    message_delta = ClaudeMessageDeltaEvent(
        delta=ClaudeMessageDelta(stop_reason=stop_reason),
        usage=ClaudeMessageDeltaUsage(output_tokens=output_tokens),
    )
    yield f"event: message_delta\ndata: {message_delta.model_dump_json()}\n\n"

    # message_stop
    message_stop = ClaudeMessageStopEvent()
    yield f"event: message_stop\ndata: {message_stop.model_dump_json()}\n\n"
