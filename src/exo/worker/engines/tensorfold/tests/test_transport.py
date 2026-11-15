import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from exo.shared.request_errors import generation_error_code
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.worker.engines.tensorfold.tests.fixture_server import frames
from exo.worker.engines.tensorfold.transport import Result, StreamTransport

MODEL = ModelId("native-transport-fixture")


class ResponseBytes(httpx.AsyncByteStream):
    def __init__(self, body: dict[str, Any], hold: bool, pace: float = 0.0):
        self.body = body
        self.hold = hold
        self.pace = pace
        self.started = threading.Event()
        self.closed = threading.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        for frame in frames(self.body, str(MODEL)):
            for offset in range(0, len(frame), 17):
                yield frame[offset : offset + 17]
            if self.hold:
                await asyncio.sleep(30)
            if self.pace:
                await asyncio.sleep(self.pace)

    async def aclose(self) -> None:
        self.closed.set()


def install_http_fixture(monkeypatch: pytest.MonkeyPatch) -> list[ResponseBytes]:
    streams: list[ResponseBytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "model": str(MODEL),
                    "warming": False,
                    "max_batch_size": 2,
                    "backend": "cuda",
                    "version": "fixture",
                    "memory": {"peak": 1234},
                },
            )
        if b"native-400" in request.content:
            return httpx.Response(
                400, json={"error": {"message": "Unsupported sampling option"}}
            )
        body = cast(dict[str, Any], json.loads(request.content))
        text = json.dumps(body)
        stream = ResponseBytes(body, "hold" in text, 0.3 if "steady" in text else 0.0)
        streams.append(stream)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=stream
        )

    def mock_transport(**_: object) -> httpx.MockTransport:
        return httpx.MockTransport(handle)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", mock_transport)
    return streams


def body(prompt: str) -> bytes:
    return json.dumps(
        {
            "model": str(MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3,
        }
    ).encode()


def received(transport: StreamTransport) -> list[Result]:
    result: list[Result] = []
    while True:
        try:
            result.append(transport.events.get_nowait())
        except queue.Empty:
            return result


def test_transport_consumes_fragmented_native_bytes_and_authoritative_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 2)
    try:
        task = TaskId("normal")
        transport.schedule(transport.generate(task, MODEL, body("normal"))).result(
            timeout=5
        )
        output = received(transport)
        assert all(identifier == task for identifier, _ in output)
        tokens = [chunk for _, chunk in output if isinstance(chunk, TokenChunk)]
        assert "".join(chunk.text for chunk in tokens) == "Answer café ✓"
        assert [chunk.token_id for chunk in tokens] == [301, 302, 303, 303]
        assert tokens[-1].finish_reason == "stop"
        assert tokens[-1].usage is not None
        assert tokens[-1].usage.completion_tokens == 3
        assert tokens[-1].stats is not None
        assert tokens[-1].stats.peak_memory_usage.in_bytes == 1234
        assert isinstance(output[-1][1], FinishedResponse)
        assert streams[0].closed.is_set()
    finally:
        transport.close()


def test_transport_idle_deadline_lets_a_steady_long_reply_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each event arrives 0.3 s after the last, so the reply takes over 2 s, four times the
    # 0.5 s deadline; every event restarts it
    install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 0.5)
    try:
        transport.schedule(
            transport.generate(TaskId("steady"), MODEL, body("steady"))
        ).result(timeout=10)
        output = received(transport)
        assert not any(isinstance(chunk, ErrorChunk) for _, chunk in output)
        tokens = [chunk for _, chunk in output if isinstance(chunk, TokenChunk)]
        assert tokens[-1].finish_reason == "stop"
        assert tokens[-1].usage is not None
        assert tokens[-1].usage.completion_tokens == 3
        assert isinstance(output[-1][1], FinishedResponse)
    finally:
        transport.close()


def test_native_bad_request_becomes_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 2)
    try:
        transport.schedule(
            transport.generate(TaskId("refused"), MODEL, body("native-400"))
        ).result(timeout=5)
        errors = [
            chunk for _, chunk in received(transport) if isinstance(chunk, ErrorChunk)
        ]
        assert len(errors) == 1
        assert generation_error_code(errors[0].error_message) == "invalid_request"
        assert errors[0].error_message.endswith("Unsupported sampling option.")
    finally:
        transport.close()


@pytest.mark.parametrize("fault", ["truncate", "malformed", "bad-offset", "bad-usage"])
def test_transport_refuses_corruption_then_serves_again(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 2)
    try:
        task = TaskId("fault")
        transport.schedule(
            transport.generate(task, MODEL, body(f"fixture:{fault}"))
        ).result(timeout=5)
        output = received(transport)
        assert any(isinstance(chunk, ErrorChunk) for _, chunk in output)
        assert not any(
            isinstance(chunk, TokenChunk) and chunk.finish_reason is not None
            for _, chunk in output
        )
        assert isinstance(output[-1][1], FinishedResponse)
        transport.schedule(
            transport.generate(TaskId("next"), MODEL, body("normal"))
        ).result(timeout=5)
        assert not any(
            isinstance(chunk, ErrorChunk) for _, chunk in received(transport)
        )
    finally:
        transport.close()


def test_transport_cancellation_closes_held_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 5)
    try:
        job = transport.schedule(
            transport.generate(TaskId("cancel"), MODEL, body("hold"))
        )
        for _ in range(100):
            if streams:
                break
            threading.Event().wait(0.01)
        assert streams and streams[0].started.wait(timeout=1)
        job.cancel()
        assert streams[0].closed.wait(timeout=2)
        assert job.cancelled()
        transport.schedule(
            transport.generate(TaskId("next"), MODEL, body("normal"))
        ).result(timeout=5)
        assert not any(
            isinstance(chunk, ErrorChunk) for _, chunk in received(transport)
        )
    finally:
        transport.close()


def test_transport_deadline_closes_held_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = install_http_fixture(monkeypatch)
    transport = StreamTransport(Path("/unused-test-transport"), 0.1)
    try:
        transport.schedule(
            transport.generate(TaskId("timeout"), MODEL, body("hold"))
        ).result(timeout=2)
        output = received(transport)
        assert any(
            isinstance(chunk, ErrorChunk) and "deadline" in chunk.error_message
            for _, chunk in output
        )
        assert streams[0].closed.is_set()
        assert isinstance(output[-1][1], FinishedResponse)
    finally:
        transport.close()
