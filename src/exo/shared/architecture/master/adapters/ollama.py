from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaDoneReason,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaToolCall,
    OllamaToolFunction,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _map_done_reason(
    finish_reason: str | None,
) -> OllamaDoneReason | None:
    if finish_reason is None:
        return None
    if finish_reason == "stop":
        return "stop"
    if finish_reason == "length":
        return "length"
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_call"
    if finish_reason == "error":
        return "error"
    return "stop"


def _try_parse_json(value: str) -> dict[str, Any] | str:
    try:
        return json.loads(value)  # type: ignore
    except json.JSONDecodeError:
        return value


def _build_tool_calls(chunk: ToolCallChunk) -> list[OllamaToolCall]:
    tool_calls: list[OllamaToolCall] = []
    for index, tool in enumerate(chunk.tool_calls):
        # tool.arguments is always str; try to parse as JSON dict for Ollama format
        arguments: dict[str, Any] | str = _try_parse_json(tool.arguments)
        tool_calls.append(
            OllamaToolCall(
                id=tool.id,
                type="function",
                function=OllamaToolFunction(
                    name=tool.name, arguments=arguments, index=index
                ),
            )
        )
    return tool_calls


def _get_usage(
    chunk: TokenChunk | ToolCallChunk,
) -> tuple[int | None, int | None]:
    """Extract (prompt_eval_count, eval_count) from a chunk."""
    if chunk.usage is not None:
        return (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
    if chunk.stats is not None:
        return (chunk.stats.prompt_tokens, chunk.stats.generation_tokens)
    return (None, None)


def ollama_request_to_text_generation(
    request: OllamaChatRequest,
) -> TextGenerationTaskParams:
    """Convert Ollama chat request to exo's internal text generation format."""
    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []
    tool_message_index = 0

    for msg in request.messages:
        content = msg.content or ""

        if msg.role == "system":
            if instructions is None:
                instructions = content
            else:
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": "system", "content": content})
            continue

        if msg.role in ("user", "assistant") and (
            msg.content is not None or msg.thinking is not None or msg.tool_calls
        ):
            input_messages.append(InputMessage(role=msg.role, content=content))

        dumped: dict[str, Any] = {"role": msg.role, "content": content}
        if msg.thinking is not None:
            dumped["thinking"] = msg.thinking
        if msg.tool_calls is not None:
            tool_calls_list: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                function: dict[str, Any] = {
                    "name": tc.function.name,
                    "arguments": (
                        json.dumps(tc.function.arguments)
                        if isinstance(tc.function.arguments, dict)
                        else tc.function.arguments
                    ),
                }
                if tc.function.index is not None:
                    function["index"] = tc.function.index
                tool_call: dict[str, Any] = {"function": function}
                if tc.id is not None:
                    tool_call["id"] = tc.id
                if tc.type is not None:
                    tool_call["type"] = tc.type
                tool_calls_list.append(tool_call)
            dumped["tool_calls"] = tool_calls_list
        if msg.name is not None:
            dumped["name"] = msg.name
        if msg.role == "tool":
            tool_message_index += 1
            tool_call_id = msg.tool_name or msg.name or f"tool_{tool_message_index}"
            dumped["tool_call_id"] = tool_call_id
            if msg.tool_name is not None:
                dumped["tool_name"] = msg.tool_name
        chat_template_messages.append(dumped)

    options = request.options
    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content="")],
        instructions=instructions,
        max_output_tokens=options.num_predict if options else None,
        temperature=options.temperature if options else None,
        top_p=options.top_p if options else None,
        top_k=options.top_k if options else None,
        stop=options.stop if options else None,
        seed=options.seed if options else None,
        stream=request.stream,
        tools=request.tools,
        enable_thinking=request.think,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
    )


async def generate_ollama_chat_stream(
    _command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate streaming responses in Ollama format (newline-delimited JSON)."""
    thinking_parts: list[str] = []

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                error_response = OllamaChatResponse(
                    model=str(chunk.model),
                    message=OllamaMessage(
                        role="assistant", content=chunk.error_message
                    ),
                    done=True,
                    done_reason="error",
                )
                yield f"{error_response.model_dump_json(exclude_none=True)}\n"
                return

            case ToolCallChunk():
                prompt_eval, eval_count = _get_usage(chunk)
                response = OllamaChatResponse(
                    model=str(chunk.model),
                    message=OllamaMessage(
                        role="assistant",
                        content="",
                        tool_calls=_build_tool_calls(chunk),
                        thinking="".join(thinking_parts) if thinking_parts else None,
                    ),
                    done=True,
                    done_reason="tool_call",
                    prompt_eval_count=prompt_eval,
                    eval_count=eval_count,
                )
                yield f"{response.model_dump_json(exclude_none=True)}\n"
                return

            case TokenChunk():
                done = chunk.finish_reason is not None

                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                    response = OllamaChatResponse(
                        model=str(chunk.model),
                        message=OllamaMessage(
                            role="assistant", content="", thinking=chunk.text
                        ),
                        done=False,
                    )
                    yield f"{response.model_dump_json(exclude_none=True)}\n"
                elif done:
                    prompt_eval, eval_count = _get_usage(chunk)
                    response = OllamaChatResponse(
                        model=str(chunk.model),
                        message=OllamaMessage(
                            role="assistant",
                            content=chunk.text,
                        ),
                        done=True,
                        done_reason=_map_done_reason(chunk.finish_reason),
                        prompt_eval_count=prompt_eval,
                        eval_count=eval_count,
                    )
                    yield f"{response.model_dump_json(exclude_none=True)}\n"
                else:
                    response = OllamaChatResponse(
                        model=str(chunk.model),
                        message=OllamaMessage(role="assistant", content=chunk.text),
                        done=False,
                    )
                    yield f"{response.model_dump_json(exclude_none=True)}\n"

                if done:
                    return


async def collect_ollama_chat_response(
    _command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    """Collect streaming chunks into a single non-streaming Ollama response.

    Returns an AsyncGenerator[str] (single yield) for consistency with FastAPI
    StreamingResponse cancellation handling.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[OllamaToolCall] = []
    model: str | None = None
    finish_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                raise ValueError(chunk.error_message or "Internal server error")

            case TokenChunk():
                if model is None:
                    model = str(chunk.model)
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    prompt_eval_count, eval_count = _get_usage(chunk)

            case ToolCallChunk():
                if model is None:
                    model = str(chunk.model)
                tool_calls.extend(_build_tool_calls(chunk))
                finish_reason = chunk.finish_reason
                prompt_eval_count, eval_count = _get_usage(chunk)

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts) if thinking_parts else None
    assert model is not None

    yield OllamaChatResponse(
        model=model,
        message=OllamaMessage(
            role="assistant",
            content=combined_text,
            thinking=combined_thinking,
            tool_calls=tool_calls if tool_calls else None,
        ),
        done=True,
        done_reason=_map_done_reason(finish_reason),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    ).model_dump_json(exclude_none=True)
    return


# ── /api/generate ──


def ollama_generate_request_to_text_generation(
    request: OllamaGenerateRequest,
) -> TextGenerationTaskParams:
    """Convert Ollama generate request to exo's internal text generation format."""
    chat_template_messages: list[dict[str, Any]] = []
    if request.system:
        chat_template_messages.append({"role": "system", "content": request.system})
    chat_template_messages.append({"role": "user", "content": request.prompt})

    options = request.options
    return TextGenerationTaskParams(
        model=request.model,
        input=[InputMessage(role="user", content=request.prompt)],
        instructions=request.system,
        max_output_tokens=options.num_predict if options else None,
        temperature=options.temperature if options else None,
        top_p=options.top_p if options else None,
        top_k=options.top_k if options else None,
        stop=options.stop if options else None,
        seed=options.seed if options else None,
        stream=request.stream,
        enable_thinking=request.think,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
    )


async def generate_ollama_generate_stream(
    _command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate streaming responses for /api/generate in Ollama NDJSON format."""
    thinking_parts: list[str] = []

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                resp = OllamaGenerateResponse(
                    model=str(chunk.model),
                    response="",
                    done=True,
                    done_reason="error",
                )
                yield f"{resp.model_dump_json(exclude_none=True)}\n"
                return

            case ToolCallChunk():
                # generate endpoint doesn't support tools; emit as done
                prompt_eval, eval_count = _get_usage(chunk)
                resp = OllamaGenerateResponse(
                    model=str(chunk.model),
                    response="",
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=prompt_eval,
                    eval_count=eval_count,
                )
                yield f"{resp.model_dump_json(exclude_none=True)}\n"
                return

            case TokenChunk():
                done = chunk.finish_reason is not None

                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                    resp = OllamaGenerateResponse(
                        model=str(chunk.model),
                        response="",
                        thinking=chunk.text,
                        done=False,
                    )
                    yield f"{resp.model_dump_json(exclude_none=True)}\n"
                elif done:
                    prompt_eval, eval_count = _get_usage(chunk)
                    resp = OllamaGenerateResponse(
                        model=str(chunk.model),
                        response=chunk.text,
                        done=True,
                        done_reason=_map_done_reason(chunk.finish_reason),
                        prompt_eval_count=prompt_eval,
                        eval_count=eval_count,
                    )
                    yield f"{resp.model_dump_json(exclude_none=True)}\n"
                else:
                    resp = OllamaGenerateResponse(
                        model=str(chunk.model),
                        response=chunk.text,
                        done=False,
                    )
                    yield f"{resp.model_dump_json(exclude_none=True)}\n"

                if done:
                    return


async def collect_ollama_generate_response(
    _command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    """Collect chunks into a single non-streaming /api/generate response."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    model: str | None = None
    finish_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue
            case ErrorChunk():
                raise ValueError(chunk.error_message or "Internal server error")
            case TokenChunk():
                if model is None:
                    model = str(chunk.model)
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    prompt_eval_count, eval_count = _get_usage(chunk)
            case ToolCallChunk():
                if model is None:
                    model = str(chunk.model)
                finish_reason = chunk.finish_reason
                prompt_eval_count, eval_count = _get_usage(chunk)

    assert model is not None
    yield OllamaGenerateResponse(
        model=model,
        response="".join(text_parts),
        thinking="".join(thinking_parts) if thinking_parts else None,
        done=True,
        done_reason=_map_done_reason(finish_reason),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    ).model_dump_json(exclude_none=True)
    return
