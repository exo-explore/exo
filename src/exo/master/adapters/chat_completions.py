"""OpenAI Chat Completions API adapter for converting requests/responses."""

import base64
import io
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger
from PIL import Image

from exo.download.download_utils import create_http_session
from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
    ToolCall,
    Usage,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _extract_base64_from_data_url(data_url: str) -> str:
    """Extract raw base64 from a data URL or return raw base64 as-is."""
    match = re.match(r"data:[^;]+;base64,(.+)", data_url)
    if match:
        return match.group(1)
    return data_url  # Already raw base64


# Max pixel budget before resizing — keeps serialized payloads under the
# gossipsub max_transmit_size (8 MB).  2048×2048 ≈ 4 MP; a JPEG-85 at that
# resolution is well within budget while preserving plenty of detail for
# vision models.
_MAX_IMAGE_PIXELS = 2048 * 2048


def _resize_image_if_needed(b64_data: str) -> str:
    """Resize a base64-encoded image if it exceeds the pixel budget.

    Large images produce multi-MB base64 payloads that must travel through
    the entire command pipeline (gossipsub serialization, event log, IPC).
    This caps image size at the API entry point to prevent hangs.
    """
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w * h <= _MAX_IMAGE_PIXELS:
        return b64_data

    scale = (_MAX_IMAGE_PIXELS / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.info(f"Resizing image {w}×{h} → {new_w}×{new_h} for transport")

    img = img.convert("RGB")
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _fetch_image_url(url: str) -> str:
    """Fetch an image from an HTTP(S) URL and return base64-encoded data.

    Uses exo's aiohttp session with proper SSL, proxy, and timeout handling.
    """
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url) as resp,
    ):
        resp.raise_for_status()
        data = await resp.read()
        return base64.b64encode(data).decode("ascii")


async def chat_request_to_text_generation(
    request: ChatCompletionRequest,
) -> TextGenerationTaskParams:
    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []
    images: list[str] = []  # base64-encoded image data

    for msg in request.messages:
        # Normalize content to string, extracting images from multimodal parts
        content: str
        has_images = False
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        else:
            # List of content parts — may include text and image_url
            text_parts: list[str] = []
            for part in msg.content:
                if isinstance(part, ChatCompletionMessageText):
                    text_parts.append(part.text)
                elif isinstance(part, ChatCompletionMessageImageUrl):
                    url = part.image_url.get("url", "")
                    if url:
                        if url.startswith(("http://", "https://")):
                            b64 = await _fetch_image_url(url)
                        else:
                            b64 = _extract_base64_from_data_url(url)
                        images.append(_resize_image_if_needed(b64))
                        has_images = True
            content = "\n".join(text_parts)

        # Extract system message as instructions
        if msg.role == "system":
            if instructions is None:
                instructions = content
            else:
                # Append additional system messages
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": "system", "content": content})
        else:
            # Skip messages with no meaningful content
            if msg.content is None and msg.thinking is None and msg.tool_calls is None:
                continue

            if msg.role in ("user", "assistant", "developer"):
                input_messages.append(InputMessage(role=msg.role, content=content))

            # Build full message dict for chat template (preserves tool_calls etc.)
            if has_images:
                # Preserve multimodal content structure for the chat template
                # so the Jinja template can emit <image> placeholders
                multimodal_content: list[dict[str, Any]] = []
                assert isinstance(msg.content, list)
                for part in msg.content:
                    if isinstance(part, ChatCompletionMessageText):
                        multimodal_content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ChatCompletionMessageImageUrl):
                        multimodal_content.append({"type": "image"})
                dumped = msg.model_dump(exclude_none=True)
                dumped["content"] = multimodal_content
                chat_template_messages.append(dumped)
            else:
                # Normalize content for model_dump
                msg_copy = msg.model_copy(update={"content": content})
                dumped = msg_copy.model_dump(exclude_none=True)
                chat_template_messages.append(dumped)

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
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        enable_thinking=request.enable_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        logprobs=request.logprobs or False,
        top_logprobs=request.top_logprobs,
        images=images if images else None,
    )


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    # Build logprobs if available
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from chunks."""
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_response = ErrorResponse(
                error=ErrorInfo(
                    message=chunk.error_message or "Internal server error",
                    type="InternalServerError",
                    code=500,
                )
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
            tool_call_deltas = [
                ToolCall(
                    id=tool.id,
                    index=i,
                    function=tool,
                )
                for i, tool in enumerate(chunk.tool_calls)
            ]
            tool_response = ChatCompletionResponse(
                id=command_id,
                created=int(time.time()),
                model=chunk.model,
                choices=[
                    StreamingChoiceResponse(
                        index=0,
                        delta=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=tool_call_deltas,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=last_usage,
            )
            yield f"data: {tool_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        chunk_response = chunk_to_response(chunk, command_id)
        if chunk.finish_reason is not None:
            chunk_response = chunk_response.model_copy(update={"usage": last_usage})
        yield f"data: {chunk_response.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            yield "data: [DONE]\n\n"


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        if model is None:
            model = chunk.model

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, TokenChunk):
            text_parts.append(chunk.text)
            if chunk.logprob is not None:
                logprobs_content.append(
                    LogprobsContentItem(
                        token=chunk.text,
                        logprob=chunk.logprob,
                        top_logprobs=chunk.top_logprobs or [],
                    )
                )

        if isinstance(chunk, ToolCallChunk):
            tool_calls.extend(
                ToolCall(
                    id=tool.id,
                    index=i,
                    function=tool,
                )
                for i, tool in enumerate(chunk.tool_calls)
            )

        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    assert model is not None

    yield ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=last_usage,
    ).model_dump_json()
    return
