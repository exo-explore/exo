import json
import time
from collections.abc import Iterator

import httpx
import pytest
from pydantic import BaseModel, Field, JsonValue

from exo.api.types import ChatCompletionChoice, ChatCompletionResponse, Usage
from exo.testkit.tensorfold import MODEL, NativeCluster
from exo.testkit.tests.conftest import report_cluster

pytestmark = pytest.mark.cluster


@pytest.fixture(scope="module")
def shared_native_cluster(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[NativeCluster]:
    """Two real nodes with the model placed on node one; the scenarios share them."""
    cluster = NativeCluster(2, tmp_path_factory.mktemp("tensorfold-cluster"))
    try:
        cluster.start(timeout=90)
        cluster.launch(on=(1,), via=0, model=MODEL, timeout=60)
        yield cluster
    finally:
        cluster.close()
        assert not cluster.fallbacks, cluster.fallbacks


@pytest.fixture
def native_cluster(
    shared_native_cluster: NativeCluster, request: pytest.FixtureRequest
) -> NativeCluster:
    report_cluster(request, shared_native_cluster)
    return shared_native_cluster


class _Delta(BaseModel):
    content: str | None = None


class _StreamChoice(BaseModel):
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _Frame(BaseModel):
    """One streamed event as a client reads it; fields a check doesn't use are ignored."""

    object: str | None = None
    choices: list[_StreamChoice] = Field(default_factory=list)
    usage: Usage | None = None
    error: dict[str, JsonValue] | None = None


def stream(
    cluster: NativeCluster, prompt: str, **options: object
) -> list[_Frame | str]:
    received: list[_Frame | str] = []
    with httpx.stream(
        "POST",
        cluster.nodes[0].url + "/v1/chat/completions",
        json={
            "model": str(MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 8,
            **options,
        },
        timeout=30,
        trust_env=False,
    ) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if line.startswith("data: "):
                value = line.removeprefix("data: ")
                received.append(
                    value if value == "[DONE]" else _Frame.model_validate_json(value)
                )
    return received


def reply(response: httpx.Response) -> tuple[ChatCompletionChoice, Usage]:
    assert response.status_code == 200, response.text
    result = ChatCompletionResponse.model_validate_json(response.content)
    choice = result.choices[0]
    assert isinstance(choice, ChatCompletionChoice), result
    assert result.usage is not None, result
    return choice, result.usage


def success(cluster: NativeCluster, prompt: str = "fixture:normal") -> Usage:
    choice, usage = reply(cluster.request(prompt))
    assert choice.message.content == "Answer café ✓", choice
    assert choice.finish_reason == "stop", choice
    assert usage.prompt_tokens == 11, usage
    assert usage.completion_tokens == 3, usage
    assert usage.total_tokens == 14, usage
    return usage


def test_request_crosses_real_nodes_and_native_subprocess(
    native_cluster: NativeCluster,
) -> None:
    cluster = native_cluster
    pair = (cluster.nodes[0].name, cluster.nodes[1].name)
    connections_before = cluster.network.api_connections.get(pair, 0)
    usage = success(cluster, "fixture:normal preserved-sampling")
    assert cluster.network.api_connections.get(pair, 0) > connections_before
    assert usage.prompt_tokens_details.cached_tokens == 4
    requests = [
        entry["request"]
        for entry in cluster.native_events(1)
        if entry["event"] == "request" and "preserved-sampling" in json.dumps(entry)
    ]
    assert len(requests) == 1
    assert requests[0]["stream"] is True
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["stream_options"]["tensorfold_token_stream_version"] == 1
    assert requests[0]["temperature"] == 0
    started = [entry for entry in cluster.native_events(1) if entry["event"] == "start"]
    assert started[-1]["context"] == 4096
    assert started[-1]["parallel"] == 2


def test_streaming_usage_unicode_and_length(native_cluster: NativeCluster) -> None:
    received = stream(native_cluster, "fixture:normal", max_tokens=2)
    assert received[-1] == "[DONE]", received
    frames = [item for item in received if isinstance(item, _Frame)]
    assert all(frame.object != "tensorfold.token.chunk" for frame in frames)
    content = "".join(
        choice.delta.content or "" for frame in frames for choice in frame.choices
    )
    assert content == "Answer café "
    finishes = [
        choice.finish_reason
        for frame in frames
        for choice in frame.choices
        if choice.finish_reason is not None
    ]
    assert finishes == ["length"]
    usage = [frame.usage for frame in frames if frame.usage is not None]
    assert usage[-1].completion_tokens == 2
    assert usage[-1].prompt_tokens_details.cached_tokens == 4


def test_reasoning_and_structured_tool_calls(native_cluster: NativeCluster) -> None:
    choice, usage = reply(native_cluster.request("fixture:reasoning"))
    assert choice.message.reasoning_content == "Checking the result."
    assert choice.message.content == "café ✓"
    assert usage.completion_tokens_details.reasoning_tokens == 1
    response = native_cluster.request(
        "fixture:tool",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    choice, _ = reply(response)
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    tool = choice.message.tool_calls[0]
    assert tool.id == "call_fixture"
    assert tool.function.name == "weather"
    assert json.loads(tool.function.arguments) == {"city": "London"}


def test_cancelled_stream_releases_native_request_and_preserves_peer(
    native_cluster: NativeCluster,
) -> None:
    cluster = native_cluster
    prompt = "fixture:slow cancel-client"
    with httpx.stream(
        "POST",
        cluster.nodes[0].url + "/v1/chat/completions",
        json={
            "model": str(MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 8,
        },
        timeout=30,
        trust_env=False,
    ) as response:
        assert response.status_code == 200
        first = next(
            line
            for line in response.iter_lines()
            if line.startswith("data: ") and '"content"' in line
        )
        assert "Answer" in first
        started = time.monotonic()
        success(cluster, "fixture:normal simultaneous-peer")
        assert time.monotonic() - started < 5
    cluster.wait_for(
        lambda: any(
            entry["event"] == "cancelled" and entry.get("prompt") == prompt
            for entry in cluster.native_events(1)
        ),
        timeout=8,
        what="native request cancellation after client disconnect",
    )
    success(cluster, "fixture:normal after-cancel")


@pytest.mark.parametrize(
    "fault", ["truncate", "malformed", "bad-offset", "bad-usage", "http-error"]
)
def test_broken_native_response_is_an_error_and_next_request_works(
    native_cluster: NativeCluster, fault: str
) -> None:
    received = stream(native_cluster, f"fixture:{fault}")
    frames = [item for item in received if isinstance(item, _Frame)]
    assert any(frame.error for frame in frames), received
    assert not any(
        choice.finish_reason == "stop" for frame in frames for choice in frame.choices
    ), received
    success(native_cluster, f"fixture:normal after-{fault}")


def test_native_crash_is_reported_and_supervisor_restarts_runtime(
    native_cluster: NativeCluster,
) -> None:
    cluster = native_cluster
    previous = {
        entry["pid"] for entry in cluster.native_events(1) if entry["event"] == "start"
    }
    assert previous
    received = stream(cluster, "fixture:crash")
    assert any(isinstance(frame, _Frame) and frame.error for frame in received), (
        received
    )
    cluster.wait_for(
        lambda: any(
            entry["event"] == "start" and entry["pid"] not in previous
            for entry in cluster.native_events(1)
        ),
        timeout=60,
        what="EXO supervisor to replace the crashed native runtime",
    )
    cluster.wait_serving(model=MODEL, timeout=30)
    success(cluster, "fixture:normal after-native-crash")
