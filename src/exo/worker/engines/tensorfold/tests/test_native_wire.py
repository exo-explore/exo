"""Real native HTTP server -> EXO parser, with scripted tokens and no GPU weights.

Set EXO_TENSORFOLD_TEST_NATIVE_BINARY to TensorFold's built ``fake_serve`` and
EXO_TENSORFOLD_TEST_NATIVE_FIXTURES to its ``zig/tests/server/fixtures`` directory.
This is a TCP protocol test. It does not replace the managed Unix-socket tests.
"""

import json
import os
import selectors
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from exo.shared.types.chunks import Chunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import ModelId
from exo.worker.engines.tensorfold.protocol import StreamProtocol
from exo.worker.engines.tensorfold.sse import EventDecoder, Heartbeat

MODEL = ModelId("native-wire-fixture")


@pytest.fixture(scope="module")
def native_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    binary = os.environ.get("EXO_TENSORFOLD_TEST_NATIVE_BINARY")
    fixtures = os.environ.get("EXO_TENSORFOLD_TEST_NATIVE_FIXTURES")
    if not binary or not fixtures:
        pytest.skip("Requires explicitly configured native fake_serve and fixtures")
    assert Path(binary).is_file()
    assert (Path(fixtures) / "scripts.json").is_file()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TENSORFOLD_", "TF_FAKE_"))
    }
    environment.update({"TENSORFOLD_NO_LIVE": "1", "TF_FAKE_CONTEXT": "8192"})
    scripts = cast(
        dict[str, object], json.loads((Path(fixtures) / "scripts.json").read_text())
    )
    scripts["exo_disconnect"] = {
        "delay_ms": 100,
        "chunks": [["Hello"], *[[" world"] for _ in range(80)], ["<|im_end|>"]],
    }
    script_path = tmp_path_factory.mktemp("native-wire") / "scripts.json"
    script_path.write_text(json.dumps(scripts))
    environment["TF_FAKE_SCRIPTS"] = str(script_path)
    process = subprocess.Popen(
        [
            binary,
            "serve",
            fixtures,
            "--name",
            str(MODEL),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--parallel",
            "2",
            "--keep-warm",
            "0",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 10
        startup: list[bytes] = []
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if not selector.select(max(0, deadline - time.monotonic())):
                    break
                line = process.stdout.readline()
                startup.append(line)
                if line.startswith(b"PORT "):
                    yield f"http://127.0.0.1:{int(line.split()[1])}"
                    return
                if not line:
                    break
        pytest.fail(f"Native fixture did not announce a listener: {startup!r}")
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


@dataclass
class NativeReply:
    chunks: list[Chunk]
    raw_ids: list[int]

    @property
    def text(self) -> str:
        return "".join(
            chunk.text
            for chunk in self.chunks
            if isinstance(chunk, TokenChunk) and not chunk.is_thinking
        )

    @property
    def reasoning(self) -> str:
        return "".join(
            chunk.text
            for chunk in self.chunks
            if isinstance(chunk, TokenChunk) and chunk.is_thinking
        )


def consume(base: str, script: str, **options: object) -> NativeReply:
    protocol = StreamProtocol(MODEL)
    decoder = EventDecoder()
    reply = NativeReply([], [])
    request = {
        "model": str(MODEL),
        "messages": [{"role": "user", "content": f"Hi @script={script}"}],
        "stream": True,
        "max_tokens": 256,
        "return_token_ids": True,
        "stream_options": {"tensorfold_token_stream_version": 1},
        "chat_template_kwargs": {"enable_thinking": False},
        **options,
    }
    with httpx.stream(
        "POST",
        base + "/v1/chat/completions",
        json=request,
        timeout=10,
        trust_env=False,
    ) as response:
        response.raise_for_status()
        for data in response.iter_bytes(chunk_size=17):
            for event in decoder.feed(data):
                if isinstance(event, Heartbeat):
                    continue
                if event != "[DONE]":
                    value = cast(dict[str, Any], json.loads(event))
                    if value.get("object") == "tensorfold.token.chunk":
                        assert value["offset"] == len(reply.raw_ids)
                        reply.raw_ids.extend(cast(list[int], value["token_ids"]))
                reply.chunks.extend(protocol.feed(event))
    decoder.finish()
    reply.chunks.append(protocol.finish(0))
    return reply


@pytest.mark.parametrize("separate_usage", [False, True])
def test_real_native_tokens_match_exo_content_and_usage(
    native_url: str, separate_usage: bool
) -> None:
    reply = consume(
        native_url,
        "plain",
        stream_options={
            "tensorfold_token_stream_version": 1,
            "include_usage": separate_usage,
        },
    )
    assert reply.text == "Hello world!"
    # Fixed vocabulary in TensorFold's parity fixture: Hello, world, !, EOS.
    assert reply.raw_ids == [284, 285, 33, 257]
    terminal = reply.chunks[-1]
    assert isinstance(terminal, TokenChunk) and terminal.finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.completion_tokens == 4
    assert terminal.stats is not None and terminal.stats.generation_tokens == 4
    assert all(
        chunk.token_id in reply.raw_ids
        for chunk in reply.chunks
        if isinstance(chunk, TokenChunk)
    )


def test_real_native_utf8_split_across_tokens_is_valid_exo_text(
    native_url: str,
) -> None:
    reply = consume(native_url, "split_utf8")
    assert reply.text == "café € ok😀"
    assert 195 in reply.raw_ids and 169 in reply.raw_ids
    assert 240 in reply.raw_ids and 128 in reply.raw_ids
    assert "\ufffd" not in reply.text


def test_real_native_reasoning_preserves_cache_usage(native_url: str) -> None:
    reply = consume(native_url, "think", chat_template_kwargs={"enable_thinking": True})
    assert reply.reasoning == "Let me think"
    assert reply.text == "The answer is fast."
    terminal = reply.chunks[-1]
    assert isinstance(terminal, TokenChunk) and terminal.usage is not None
    assert terminal.usage.prompt_tokens_details.cached_tokens == 3
    assert terminal.usage.completion_tokens == len(reply.raw_ids)


def test_real_native_tool_calls_survive_exo_aggregation(native_url: str) -> None:
    reply = consume(
        native_url,
        "qwen_tool",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    )
    terminal = reply.chunks[-1]
    assert isinstance(terminal, ToolCallChunk)
    assert len(terminal.tool_calls) == 1
    tool = terminal.tool_calls[0]
    assert tool.id and tool.name == "lookup"
    assert json.loads(tool.arguments) == {"query": "speed"}
    assert terminal.usage is not None
    assert terminal.usage.completion_tokens == len(reply.raw_ids)


def test_real_native_length_cut_preserves_exact_committed_ids(native_url: str) -> None:
    reply = consume(native_url, "long", max_tokens=2)
    terminal = reply.chunks[-1]
    assert reply.text == "Hello world"
    assert reply.raw_ids == [284, 285]
    assert isinstance(terminal, TokenChunk) and terminal.finish_reason == "length"
    assert terminal.usage is not None and terminal.usage.completion_tokens == 2


def running_requests(client: httpx.Client) -> int:
    response = client.get("/metrics")
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("tensorfold:requests_running "):
            return int(line.split()[1])
    raise AssertionError("Native server did not report its active request gauge")


def test_real_native_socket_disconnect_releases_admitted_request(
    native_url: str,
) -> None:
    request = {
        "model": str(MODEL),
        "messages": [{"role": "user", "content": "Hi @script=exo_disconnect"}],
        "stream": True,
        "max_tokens": 128,
        "return_token_ids": True,
        "stream_options": {"tensorfold_token_stream_version": 1},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    decoder = EventDecoder()
    protocol = StreamProtocol(MODEL)
    saw_output = False
    with httpx.Client(base_url=native_url, trust_env=False, timeout=3) as client:
        assert running_requests(client) == 0
        with client.stream("POST", "/v1/chat/completions", json=request) as response:
            response.raise_for_status()
            for data in response.iter_bytes(chunk_size=17):
                for event in decoder.feed(data):
                    if isinstance(event, Heartbeat):
                        continue
                    if any(
                        isinstance(chunk, TokenChunk) and chunk.text
                        for chunk in protocol.feed(event)
                    ):
                        saw_output = True
                if saw_output:
                    break
            assert saw_output and not protocol.done
            assert running_requests(client) == 1
        # The complete script takes >8 seconds. A release here proves early
        # cancellation rather than merely observing natural completion.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and running_requests(client):
            time.sleep(0.02)
        assert running_requests(client) == 0
    assert consume(native_url, "plain").text == "Hello world!"


def test_real_native_unknown_protocol_version_is_an_explicit_error(
    native_url: str,
) -> None:
    response = httpx.post(
        native_url + "/v1/chat/completions",
        json={
            "model": str(MODEL),
            "messages": [{"role": "user", "content": "Hi @script=plain"}],
            "stream": True,
            "stream_options": {"tensorfold_token_stream_version": 2},
        },
        timeout=3,
        trust_env=False,
    )
    assert response.status_code == 400
    assert "tensorfold_token_stream_version" in response.json()["error"]["message"]
    assert consume(native_url, "plain").text == "Hello world!"
