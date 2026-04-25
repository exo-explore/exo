"""FastAPI handlers for the reasoning proxy.

Two handlers, one shape: read body → resolve dialect → reattach cached
reasoning to designated history indices → forward → tee the response stream
→ capture emitted reasoning → cache under the emitted assistant's hash.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import Headers as StarletteHeaders

from exo.reasoning_proxy._helpers import (
    as_dict,
    as_list,
    dict_get_dict,
    dict_get_list,
    dict_get_str,
)
from exo.reasoning_proxy.accumulator import ClaudeAccumulator, OpenAIAccumulator
from exo.reasoning_proxy.cache import ReasoningCache
from exo.reasoning_proxy.dialects import get_dialect
from exo.reasoning_proxy.hashing import hash_claude_assistant, hash_openai_assistant
from exo.reasoning_proxy.registry import DialectRegistry

logger = logging.getLogger(__name__)


def _parse_body(raw_body: bytes) -> dict[str, object] | None:
    try:
        parsed = cast(object, json.loads(raw_body))
    except json.JSONDecodeError:
        return None
    return as_dict(parsed)


def _content_for_hash(value: object) -> str | list[object] | None:
    if isinstance(value, str):
        return value
    as_listed = as_list(value)
    if as_listed is not None:
        return as_listed
    return None


def _attach_openai_reasoning(
    messages: list[dict[str, object]],
    indices: set[int],
    cache: ReasoningCache,
) -> None:
    for i in indices:
        msg = messages[i]
        existing = dict_get_str(msg, "reasoning_content")
        if existing:
            continue
        content = _content_for_hash(msg.get("content"))
        tool_calls_raw = dict_get_list(msg, "tool_calls") or []
        tool_calls: list[dict[str, object]] = [
            d for d in (as_dict(t) for t in tool_calls_raw) if d is not None
        ]
        h = hash_openai_assistant(content, tool_calls or None)
        cached = cache.get(h)
        if cached is not None:
            msg["reasoning_content"] = cached


def _attach_claude_reasoning(
    messages: list[dict[str, object]],
    indices: set[int],
    cache: ReasoningCache,
) -> None:
    for i in indices:
        msg = messages[i]
        content = as_list(msg.get("content"))
        if content is None:
            continue
        has_thinking = False
        normalized: list[dict[str, object]] = []
        for raw in content:
            block = as_dict(raw)
            if block is None:
                continue
            if block.get("type") == "thinking":
                has_thinking = True
            normalized.append(block)
        if has_thinking:
            continue
        h = hash_claude_assistant(normalized)
        cached = cache.get(h)
        if cached is None:
            continue
        new_content: list[dict[str, object]] = [
            {"type": "thinking", "thinking": cached}
        ]
        new_content.extend(normalized)
        msg["content"] = new_content


async def _stream_and_capture_openai(
    upstream_resp: httpx.Response,
    cache: ReasoningCache,
) -> AsyncIterator[bytes]:
    accumulator = OpenAIAccumulator()
    try:
        async for chunk in upstream_resp.aiter_raw():
            accumulator.feed_bytes(chunk)
            yield chunk
    finally:
        await upstream_resp.aclose()
    reasoning = accumulator.reasoning
    if not reasoning:
        return
    h = hash_openai_assistant(accumulator.content, accumulator.tool_calls)
    cache.put(h, reasoning)


async def _stream_and_capture_claude(
    upstream_resp: httpx.Response,
    cache: ReasoningCache,
) -> AsyncIterator[bytes]:
    accumulator = ClaudeAccumulator()
    try:
        async for chunk in upstream_resp.aiter_raw():
            accumulator.feed_bytes(chunk)
            yield chunk
    finally:
        await upstream_resp.aclose()
    reasoning = accumulator.reasoning
    if not reasoning:
        return
    h = hash_claude_assistant(accumulator.content_blocks)
    cache.put(h, reasoning)


def _capture_openai_nonstream(body_text: str, cache: ReasoningCache) -> None:
    body = _parse_body(body_text.encode("utf-8"))
    if body is None:
        return
    choices = dict_get_list(body, "choices")
    if not choices:
        return
    first = as_dict(choices[0])
    if first is None:
        return
    message = dict_get_dict(first, "message")
    if message is None:
        return
    reasoning = dict_get_str(message, "reasoning_content")
    if not reasoning:
        return
    content = _content_for_hash(message.get("content"))
    tool_calls_raw = dict_get_list(message, "tool_calls") or []
    tool_calls: list[dict[str, object]] = [
        d for d in (as_dict(t) for t in tool_calls_raw) if d is not None
    ]
    h = hash_openai_assistant(content, tool_calls or None)
    cache.put(h, reasoning)


def _capture_claude_nonstream(body_text: str, cache: ReasoningCache) -> None:
    body = _parse_body(body_text.encode("utf-8"))
    if body is None:
        return
    content = as_list(body.get("content"))
    if content is None:
        return
    reasoning_parts: list[str] = []
    public_blocks: list[dict[str, object]] = []
    for raw in content:
        block = as_dict(raw)
        if block is None:
            continue
        if block.get("type") == "thinking":
            thinking = dict_get_str(block, "thinking")
            if thinking is not None:
                reasoning_parts.append(thinking)
        else:
            public_blocks.append(block)
    reasoning = "".join(reasoning_parts)
    if not reasoning:
        return
    h = hash_claude_assistant(public_blocks)
    cache.put(h, reasoning)


def _messages_from_body(body: dict[str, object]) -> list[dict[str, object]] | None:
    raw = as_list(body.get("messages"))
    if raw is None:
        return None
    result: list[dict[str, object]] = []
    for item in raw:
        m = as_dict(item)
        if m is None:
            return None
        result.append(m)
    return result


def register_routes(
    app: FastAPI,
    client: httpx.AsyncClient,
    upstream: str,
    cache: ReasoningCache,
    registry: DialectRegistry,
) -> None:
    async def handle_chat_completions(request: Request) -> Response:
        raw_body = await request.body()
        body = _parse_body(raw_body)
        if body is None:
            return _bad_request("invalid JSON body")

        model_id = dict_get_str(body, "model")
        if model_id is None:
            return _bad_request("missing or invalid 'model' field")

        dialect_name = await registry.resolve(model_id)
        dialect = get_dialect(dialect_name)

        messages = _messages_from_body(body)
        if messages is not None:
            has_tools = bool(body.get("tools"))
            indices = dialect.select_attach_indices(messages, has_tools=has_tools)
            if indices:
                _attach_openai_reasoning(messages, indices, cache)
                body["messages"] = messages

        forward_body = json.dumps(body).encode("utf-8")
        forward_headers = _copy_headers(request.headers)
        forward_headers["content-length"] = str(len(forward_body))

        is_stream = bool(body.get("stream"))

        try:
            if is_stream:
                req = client.build_request(
                    "POST",
                    f"{upstream}/v1/chat/completions",
                    content=forward_body,
                    headers=forward_headers,
                )
                upstream_resp = await client.send(req, stream=True)
                return StreamingResponse(
                    _stream_and_capture_openai(upstream_resp, cache),
                    status_code=upstream_resp.status_code,
                    media_type=_media_type(upstream_resp.headers, "text/event-stream"),
                    headers=_response_headers(upstream_resp.headers),
                )
            upstream_resp = await client.post(
                f"{upstream}/v1/chat/completions",
                content=forward_body,
                headers=forward_headers,
            )
            text = upstream_resp.text
            if upstream_resp.status_code == 200:
                _capture_openai_nonstream(text, cache)
            return Response(
                content=text,
                status_code=upstream_resp.status_code,
                media_type=_media_type(upstream_resp.headers, "application/json"),
                headers=_response_headers(upstream_resp.headers),
            )
        except httpx.RequestError as exc:
            logger.warning("Upstream request failed: %s", exc)
            return _bad_gateway(str(exc))

    async def handle_claude_messages(request: Request) -> Response:
        raw_body = await request.body()
        body = _parse_body(raw_body)
        if body is None:
            return _bad_request("invalid JSON body")

        model_id = dict_get_str(body, "model")
        if model_id is None:
            return _bad_request("missing or invalid 'model' field")

        dialect_name = await registry.resolve(model_id)
        dialect = get_dialect(dialect_name)

        messages = _messages_from_body(body)
        if messages is not None:
            has_tools = bool(body.get("tools"))
            indices = dialect.select_attach_indices(messages, has_tools=has_tools)
            if indices:
                _attach_claude_reasoning(messages, indices, cache)
                body["messages"] = messages

        forward_body = json.dumps(body).encode("utf-8")
        forward_headers = _copy_headers(request.headers)
        forward_headers["content-length"] = str(len(forward_body))

        is_stream = bool(body.get("stream"))

        try:
            if is_stream:
                req = client.build_request(
                    "POST",
                    f"{upstream}/v1/messages",
                    content=forward_body,
                    headers=forward_headers,
                )
                upstream_resp = await client.send(req, stream=True)
                return StreamingResponse(
                    _stream_and_capture_claude(upstream_resp, cache),
                    status_code=upstream_resp.status_code,
                    media_type=_media_type(upstream_resp.headers, "text/event-stream"),
                    headers=_response_headers(upstream_resp.headers),
                )
            upstream_resp = await client.post(
                f"{upstream}/v1/messages",
                content=forward_body,
                headers=forward_headers,
            )
            text = upstream_resp.text
            if upstream_resp.status_code == 200:
                _capture_claude_nonstream(text, cache)
            return Response(
                content=text,
                status_code=upstream_resp.status_code,
                media_type=_media_type(upstream_resp.headers, "application/json"),
                headers=_response_headers(upstream_resp.headers),
            )
        except httpx.RequestError as exc:
            logger.warning("Upstream request failed: %s", exc)
            return _bad_gateway(str(exc))

    async def health() -> dict[str, object]:
        return {"status": "ok", "cache_entries": cache.size()}

    _ = app.post("/v1/chat/completions")(handle_chat_completions)
    _ = app.post("/v1/messages")(handle_claude_messages)
    _ = app.get("/health")(health)


_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _copy_headers(headers: httpx.Headers | StarletteHeaders) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _HOP_BY_HOP:
            continue
        out[k] = v
    return out


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    return _copy_headers(headers)


def _media_type(headers: httpx.Headers, default: str) -> str:
    value = cast(object, headers.get("content-type", default))
    return value if isinstance(value, str) else default


def _bad_request(msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": {"message": msg, "type": "invalid_request_error"}},
    )


def _bad_gateway(msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {"message": f"upstream unreachable: {msg}", "type": "bad_gateway"}
        },
    )
