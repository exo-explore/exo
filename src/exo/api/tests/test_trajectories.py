"""Tests for ATIF trajectory collection."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from exo.api.trajectories import (
    ATIF_SCHEMA_VERSION,
    TrajectoryCollector,
    tap_chunk_stream,
)
from exo.api.types import (
    ChatCompletionMessage,
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    ToolCall,
    ToolCallItem,
    Usage,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def _stats(
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none",
) -> GenerationStats:
    return GenerationStats(
        prompt_tps=100.0,
        generation_tps=50.0,
        prompt_tokens=10,
        generation_tokens=5,
        peak_memory_usage=Memory.from_bytes(1024),
        prefix_cache_hit=prefix_cache_hit,
    )


def _usage(cached: int = 0) -> Usage:
    return Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
        completion_tokens_details=CompletionTokensDetails(),
    )


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], traj["steps"])


def test_single_turn_produces_user_and_agent_step(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path)
    messages = [ChatCompletionMessage(role="user", content="hello")]
    session_id = collector.record_request(
        messages=messages, client_key="ip:1.1.1.1", model="m"
    )
    assert session_id is not None
    collector.record_response(
        session_id=session_id,
        client_key="ip:1.1.1.1",
        request_messages=messages,
        assistant_text="hi",
        reasoning_content=None,
        tool_calls=[],
        stats=_stats(),
        model="m",
    )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    traj = _load(files[0])
    steps = _steps(traj)
    assert traj["schema_version"] == ATIF_SCHEMA_VERSION
    assert traj["session_id"] == session_id
    assert len(steps) == 2
    assert steps[0]["source"] == "user"
    assert steps[0]["message"] == "hello"
    assert steps[1]["source"] == "agent"
    assert steps[1]["message"] == "hi"
    assert steps[1]["model_name"] == "m"
    final_metrics = cast(dict[str, Any], traj["final_metrics"])
    assert final_metrics["total_steps"] == 2
    assert final_metrics["total_prompt_tokens"] == 10
    assert final_metrics["total_completion_tokens"] == 5


def test_multi_turn_stitches_into_one_trajectory(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path)
    client = "ip:2.2.2.2"
    m1 = [ChatCompletionMessage(role="user", content="one")]
    sid = collector.record_request(messages=m1, client_key=client, model="m")
    assert sid is not None
    collector.record_response(
        session_id=sid,
        client_key=client,
        request_messages=m1,
        assistant_text="A1",
        reasoning_content=None,
        tool_calls=[],
        stats=_stats(),
        model="m",
    )

    m2 = [
        ChatCompletionMessage(role="user", content="one"),
        ChatCompletionMessage(role="assistant", content="A1"),
        ChatCompletionMessage(role="user", content="two"),
    ]
    sid2 = collector.record_request(messages=m2, client_key=client, model="m")
    assert sid2 == sid, "continuation must reuse session id"
    assert sid2 is not None
    collector.record_response(
        session_id=sid2,
        client_key=client,
        request_messages=m2,
        assistant_text="A2",
        reasoning_content=None,
        tool_calls=[],
        stats=_stats(),
        model="m",
    )

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    traj = _load(files[0])
    steps = _steps(traj)
    sources = [s["source"] for s in steps]
    messages_out = [s["message"] for s in steps]
    assert sources == ["user", "agent", "user", "agent"]
    assert messages_out == ["one", "A1", "two", "A2"]
    step_ids = [s["step_id"] for s in steps]
    assert step_ids == [1, 2, 3, 4]


def test_tool_call_and_observation_folding(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path)
    client = "ip:3.3.3.3"
    m1 = [ChatCompletionMessage(role="user", content="compute")]
    sid = collector.record_request(messages=m1, client_key=client, model="m")
    assert sid is not None
    tool_call = ToolCallItem(id="tc_1", name="add", arguments='{"a": 1, "b": 2}')
    collector.record_response(
        session_id=sid,
        client_key=client,
        request_messages=m1,
        assistant_text="",
        reasoning_content=None,
        tool_calls=[tool_call],
        stats=_stats(),
        model="m",
    )

    assistant_with_tool = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="tc_1", function=tool_call),
        ],
    )
    m2 = [
        ChatCompletionMessage(role="user", content="compute"),
        assistant_with_tool,
        ChatCompletionMessage(role="tool", tool_call_id="tc_1", content="3"),
    ]
    sid2 = collector.record_request(messages=m2, client_key=client, model="m")
    assert sid2 == sid
    assert sid2 is not None
    collector.record_response(
        session_id=sid2,
        client_key=client,
        request_messages=m2,
        assistant_text="result is 3",
        reasoning_content=None,
        tool_calls=[],
        stats=_stats(),
        model="m",
    )

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    traj = _load(files[0])
    steps = _steps(traj)
    agent_steps = [s for s in steps if s["source"] == "agent"]
    first_agent = agent_steps[0]
    tcs = cast(list[dict[str, Any]], first_agent["tool_calls"])
    assert tcs is not None
    assert tcs[0]["function_name"] == "add"
    assert tcs[0]["arguments"] == {"a": 1, "b": 2}
    obs = cast(dict[str, Any], first_agent["observation"])
    assert obs is not None
    results = cast(list[dict[str, Any]], obs["results"])
    assert results[0] == {
        "source_call_id": "tc_1",
        "content": "3",
    }


def test_different_clients_get_separate_trajectories(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path)
    messages = [ChatCompletionMessage(role="user", content="identical")]
    sid_a = collector.record_request(messages=messages, client_key="ip:A", model="m")
    sid_b = collector.record_request(messages=messages, client_key="ip:B", model="m")
    assert sid_a is not None and sid_b is not None
    assert sid_a != sid_b
    collector.record_response(
        session_id=sid_a,
        client_key="ip:A",
        request_messages=messages,
        assistant_text="a",
        reasoning_content=None,
        tool_calls=[],
        stats=None,
        model="m",
    )
    collector.record_response(
        session_id=sid_b,
        client_key="ip:B",
        request_messages=messages,
        assistant_text="b",
        reasoning_content=None,
        tool_calls=[],
        stats=None,
        model="m",
    )
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_lru_eviction_drops_oldest(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path, capacity=2)
    for i in range(3):
        msgs = [ChatCompletionMessage(role="user", content=f"msg-{i}")]
        sid = collector.record_request(messages=msgs, client_key=f"ip:{i}", model="m")
        assert sid is not None
        collector.record_response(
            session_id=sid,
            client_key=f"ip:{i}",
            request_messages=msgs,
            assistant_text=f"r-{i}",
            reasoning_content=None,
            tool_calls=[],
            stats=None,
            model="m",
        )
    assert len(collector._sessions) <= 2  # pyright: ignore[reportPrivateUsage]


def test_disabled_writes_nothing(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=False, directory=tmp_path)
    messages = [ChatCompletionMessage(role="user", content="hi")]
    sid = collector.record_request(messages=messages, client_key="ip:x", model="m")
    assert sid is None
    collector.record_response(
        session_id="nonexistent",
        client_key="ip:x",
        request_messages=messages,
        assistant_text="x",
        reasoning_content=None,
        tool_calls=[],
        stats=None,
        model="m",
    )
    assert list(tmp_path.glob("*.json")) == []


async def _chunks_to_stream(
    chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk],
) -> AsyncGenerator[
    PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
]:
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_tap_chunk_stream_accumulates_text_stats_and_usage() -> None:
    model = ModelId("m")
    final_usage = _usage(cached=4)
    chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
        TokenChunk(model=model, text="hel", token_id=1, usage=None),
        TokenChunk(model=model, text="lo", token_id=2, usage=None),
        TokenChunk(
            model=model,
            text="",
            token_id=3,
            usage=final_usage,
            finish_reason="stop",
            stats=_stats(),
        ),
    ]
    captured: dict[str, Any] = {}

    def on_complete(
        text: str,
        reasoning: str | None,
        tool_calls: list[ToolCallItem],
        stats: GenerationStats | None,
        usage: Usage | None,
        ttft_ms: float | None,
    ) -> None:
        captured["text"] = text
        captured["reasoning"] = reasoning
        captured["tool_calls"] = tool_calls
        captured["stats"] = stats
        captured["usage"] = usage
        captured["ttft_ms"] = ttft_ms

    out: list[Any] = []
    async for chunk in tap_chunk_stream(_chunks_to_stream(chunks), on_complete):
        out.append(chunk)
    assert len(out) == 3
    assert captured["text"] == "hello"
    assert cast(list[Any], captured["tool_calls"]) == []
    assert captured["stats"] is not None
    captured_usage = cast(Usage, captured["usage"])
    assert captured_usage.prompt_tokens_details.cached_tokens == 4
    assert captured["ttft_ms"] is not None
    assert cast(float, captured["ttft_ms"]) >= 0.0


@pytest.mark.asyncio
async def test_tap_chunk_stream_captures_tool_calls() -> None:
    model = ModelId("m")
    tool = ToolCallItem(id="t1", name="foo", arguments="{}")
    chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
        ToolCallChunk(model=model, tool_calls=[tool], usage=None, stats=_stats()),
    ]
    captured: dict[str, Any] = {}

    def on_complete(
        text: str,
        reasoning: str | None,
        tool_calls: list[ToolCallItem],
        stats: GenerationStats | None,
        usage: Usage | None,
        ttft_ms: float | None,
    ) -> None:
        captured["tool_calls"] = tool_calls

    async for _ in tap_chunk_stream(_chunks_to_stream(chunks), on_complete):
        pass
    captured_tools = cast(list[ToolCallItem], captured["tool_calls"])
    assert len(captured_tools) == 1
    assert captured_tools[0].id == "t1"


def test_cached_tokens_and_ttft_recorded(tmp_path: Path) -> None:
    collector = TrajectoryCollector(enabled=True, directory=tmp_path)
    messages = [ChatCompletionMessage(role="user", content="hello")]
    sid = collector.record_request(messages=messages, client_key="ip:1", model="m")
    assert sid is not None
    collector.record_response(
        session_id=sid,
        client_key="ip:1",
        request_messages=messages,
        assistant_text="hi",
        reasoning_content=None,
        tool_calls=[],
        stats=_stats("partial"),
        model="m",
        usage=_usage(cached=7),
        ttft_ms=42.5,
    )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    traj = _load(files[0])
    steps = _steps(traj)
    agent_step = steps[1]
    metrics = cast(dict[str, Any], agent_step["metrics"])
    assert metrics["cached_tokens"] == 7
    ext = cast(dict[str, Any], metrics["_exo_extensions"])
    assert ext["prefix_cache_hit"] == "partial"
    assert ext["ttft_ms"] == 42.5
