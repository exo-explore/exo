"""Non-streaming responses: a generation that fails part way gets an error status, not an empty 200."""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from exo.api.adapters.chat_completions import collect_chat_response
from exo.api.collected_response import CollectedResponse
from exo.api.request_logger import RequestLogger
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId

MODEL = ModelId("test-model")

type Chunk = ErrorChunk | TokenChunk | ToolCallChunk | PrefillProgressChunk


async def chunks(*items: Chunk) -> AsyncGenerator[Chunk, None]:
    for item in items:
        yield item


def client_for(body: AsyncIterator[str]) -> TestClient:
    app = FastAPI()
    app.get("/")(lambda: CollectedResponse(body, media_type="application/json"))
    return TestClient(app)


def test_complete_body_is_sent_with_its_length() -> None:
    async def body() -> AsyncIterator[str]:
        yield '{"answer":'
        yield "42}"

    reply = client_for(body()).get("/")

    assert reply.status_code == 200
    assert reply.json() == {"answer": 42}
    assert reply.headers["content-length"] == str(len('{"answer":42}'))


def test_failed_generation_is_an_error_not_an_empty_200() -> None:
    collector = collect_chat_response(
        CommandId("failing"),
        chunks(
            TokenChunk(model=MODEL, text="Hel", token_id=1, usage=None),
            ErrorChunk(model=MODEL, error_message="Runner shutdown"),
        ),
    )

    reply = client_for(collector).get("/")

    assert reply.status_code == 500
    error = cast(dict[str, dict[str, object]], reply.json())["error"]
    assert error["message"] == "Runner shutdown"
    assert error["code"] == 500


def test_successful_generation_is_sent_whole() -> None:
    collector = collect_chat_response(
        CommandId("working"),
        chunks(
            TokenChunk(model=MODEL, text="Hel", token_id=1, usage=None),
            TokenChunk(
                model=MODEL, text="lo", token_id=2, usage=None, finish_reason="stop"
            ),
        ),
    )

    reply = client_for(collector).get("/")

    assert reply.status_code == 200
    body = cast(dict[str, list[dict[str, dict[str, str]]]], reply.json())
    assert body["choices"][0]["message"]["content"] == "Hello"


async def test_client_disconnect_cancels_the_generation() -> None:
    cancelled = anyio.Event()

    async def endless() -> AsyncIterator[str]:
        try:
            await anyio.sleep_forever()
            yield ""
        finally:
            cancelled.set()

    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    with anyio.fail_after(5):
        await CollectedResponse(endless(), media_type="application/json")(
            {"type": "http", "asgi": {"spec_version": "2.1"}}, receive, send
        )
        await cancelled.wait()

    assert sent == []


async def test_a_client_that_gives_up_is_not_an_api_error() -> None:
    # Behind the API's request logging, as every route is
    cancelled = anyio.Event()

    async def endless() -> AsyncIterator[str]:
        try:
            await anyio.sleep_forever()
            yield ""
        finally:
            cancelled.set()

    app = FastAPI()
    app.add_middleware(RequestLogger)
    app.post("/")(lambda: CollectedResponse(endless(), media_type="application/json"))

    requested = False

    async def receive() -> Message:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await anyio.sleep(0.1)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("client", 1),
        "server": ("server", 80),
    }
    with anyio.fail_after(5):
        # Raised "No response returned." through FastAPI's @app.middleware("http")
        await app(scope, receive, send)
        await cancelled.wait()
