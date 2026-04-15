"""ATIF-v1.4 trajectory collection for live chat completions.

Emits one JSON file per trajectory in Harbor Framework's Agent Trajectory
Interchange Format. Multi-turn chat histories are stitched into a single
trajectory by hashing the expected-next-prefix of messages after each response.
exo is self-hosted, so `metrics.cost` is always 0.0; exo-specific observability
lives under `metrics._exo_extensions` so strict ATIF consumers still parse.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast, final
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    GenerationStats,
    ToolCall,
    ToolCallItem,
    Usage,
)
from exo.shared.constants import EXO_TRAJECTORIES_DIR
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.utils.pydantic_ext import CamelCaseModel

ATIF_SCHEMA_VERSION = "ATIF-v1.4"

StepSource = Literal["user", "agent", "system"]


class AtifToolCall(BaseModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


class AtifObservationResult(BaseModel):
    source_call_id: str
    content: str


class AtifObservation(BaseModel):
    results: list[AtifObservationResult]


class AtifExoExtensions(BaseModel):
    prompt_tps: float | None = None
    generation_tps: float | None = None
    peak_memory_bytes: int | None = None
    prefix_cache_hit: Literal["none", "partial", "exact"] | None = None
    reasoning_content: str | None = None
    ttft_ms: float | None = None


class AtifStepMetrics(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    exo_extensions: AtifExoExtensions | None = Field(
        default=None, serialization_alias="_exo_extensions"
    )

    model_config = {"populate_by_name": True}


class AtifStep(BaseModel):
    step_id: int
    timestamp: str
    source: StepSource
    message: str = ""
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    observation: AtifObservation | None = None
    metrics: AtifStepMetrics | None = None
    model_name: str | None = None


class AtifAgent(BaseModel):
    name: str = "exo"
    model: str
    provider: str = "exo"
    exo_extensions: dict[str, Any] | None = Field(
        default=None, serialization_alias="_exo_extensions"
    )

    model_config = {"populate_by_name": True}


class AtifFinalMetrics(BaseModel):
    total_steps: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0


class AtifTrajectory(BaseModel):
    schema_version: str = ATIF_SCHEMA_VERSION
    session_id: str
    agent: AtifAgent
    steps: list[AtifStep] = Field(default_factory=list)
    final_metrics: AtifFinalMetrics = Field(default_factory=AtifFinalMetrics)


@final
@dataclass
class SessionState:
    session_id: str
    file_path: Path
    trajectory: AtifTrajectory
    last_prefix_len: int = 0
    last_activity: float = field(default_factory=time.monotonic)
    hash_keys: set[tuple[str, str]] = field(default_factory=set)


def _extract_text(
    content: str | ChatCompletionMessageText | list[Any] | Any | None,
) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, ChatCompletionMessageText):
        return content.text
    if isinstance(content, ChatCompletionMessageImageUrl):
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(p, ChatCompletionMessageText):
                parts.append(p.text)
        return "\n".join(parts)
    return ""


def _canonicalize(m: ChatCompletionMessage) -> dict[str, Any]:
    """Strip everything that can differ between client echo and server record.

    Keeps only role, normalized content (empty string ≡ None), tool_call_id,
    and semantic tool_calls (id + function name + parsed arguments — no index,
    no type since it's always "function"). Drops reasoning_content, name,
    function_call, logprobs-adjacent fields. This is what two honest clients
    would agree on about a message.
    """
    out: dict[str, Any] = {"role": m.role}
    text = _extract_text(m.content)
    if text:
        out["content"] = text
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        normalized_tcs: list[dict[str, Any]] = []
        for tc in m.tool_calls:
            try:
                args = cast(dict[str, Any], json.loads(tc.function.arguments))
                args_canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                args_canon = tc.function.arguments
            normalized_tcs.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args_canon,
                }
            )
        out["tool_calls"] = normalized_tcs
    return out


def _hash_messages(messages: list[ChatCompletionMessage]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(
            json.dumps(
                _canonicalize(m), sort_keys=True, separators=(",", ":")
            ).encode()
        )
        h.update(b"\x1f")
    return h.hexdigest()


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _canonical_agent_step_key(step: AtifStep) -> tuple[str, tuple[tuple[str, str], ...]]:
    tc_key: tuple[tuple[str, str], ...] = ()
    if step.tool_calls:
        tc_key = tuple(
            sorted(
                (tc.tool_call_id, tc.function_name) for tc in step.tool_calls
            )
        )
    return (step.message or "", tc_key)


def _matches_last_agent_step(
    trajectory: AtifTrajectory, msg: ChatCompletionMessage
) -> bool:
    """Return True if `msg` is an echo of the most recent agent step we stored.

    Handles opencode-style replay where the client repeats our prior assistant
    response (with minor formatting differences we've already canonicalized).
    """
    last_agent = next(
        (s for s in reversed(trajectory.steps) if s.source == "agent"), None
    )
    if last_agent is None:
        return False
    text = _extract_text(msg.content)
    client_tc_key: tuple[tuple[str, str], ...] = ()
    if msg.tool_calls:
        client_tc_key = tuple(
            sorted((tc.id, tc.function.name) for tc in msg.tool_calls)
        )
    return _canonical_agent_step_key(last_agent) == (text, client_tc_key)


def _message_to_step(
    msg: ChatCompletionMessage, step_id: int
) -> tuple[AtifStep | None, tuple[str, str] | None]:
    """Convert an input message to a step (or a tool-result to be folded)."""
    text = _extract_text(msg.content)
    if msg.role == "tool":
        call_id = msg.tool_call_id or ""
        return None, (call_id, text)
    source: StepSource
    if msg.role == "system" or msg.role == "developer":
        source = "system"
    elif msg.role == "assistant":
        source = "agent"
    else:
        source = "user"
    tool_calls: list[AtifToolCall] | None = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            try:
                args = cast(dict[str, Any], json.loads(tc.function.arguments))
            except (json.JSONDecodeError, TypeError):
                args = {"raw": tc.function.arguments}
            tool_calls.append(
                AtifToolCall(
                    tool_call_id=tc.id,
                    function_name=tc.function.name,
                    arguments=args,
                )
            )
    return (
        AtifStep(
            step_id=step_id,
            timestamp=_iso_now(),
            source=source,
            message=text,
            reasoning_content=msg.reasoning_content,
            tool_calls=tool_calls,
        ),
        None,
    )


@final
class TrajectoryCollector:
    """Stitches multi-turn chat histories into ATIF trajectory files.

    Keyed by (client_key, prefix_hash); matches the longest prefix of the
    incoming request to the hash stored when the previous response was written.
    Missing match mints a new session_id.
    """

    def __init__(
        self,
        enabled: bool,
        directory: Path,
        capacity: int = 1024,
        idle_seconds: int = 1800,
    ) -> None:
        self.enabled = enabled
        self.directory = directory
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._by_hash: dict[tuple[str, str], SessionState] = {}
        self._lock = threading.Lock()
        if enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("could not create trajectories dir {}: {}", directory, e)

    def _drop_session(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return
        for key in state.hash_keys:
            self._by_hash.pop(key, None)

    def _evict(self) -> None:
        now = time.monotonic()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_activity > self.idle_seconds
        ]
        for sid in stale:
            self._drop_session(sid)
        while len(self._sessions) > self.capacity:
            oldest_sid, _ = next(iter(self._sessions.items()))
            self._drop_session(oldest_sid)

    def _find_continuation(
        self, client_key: str, messages: list[ChatCompletionMessage]
    ) -> SessionState | None:
        for k in range(len(messages), 0, -1):
            prefix_hash = _hash_messages(messages[:k])
            state = self._by_hash.get((client_key, prefix_hash))
            if state is not None:
                state.last_prefix_len = k
                state.last_activity = time.monotonic()
                self._sessions.move_to_end(state.session_id)
                return state
        return None

    def record_request(
        self,
        messages: list[ChatCompletionMessage],
        client_key: str,
        model: str,
        cluster_info: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            self._evict()
            state = self._find_continuation(client_key, messages)
            if state is None:
                session_id = str(uuid4())
                state = SessionState(
                    session_id=session_id,
                    file_path=self.directory / f"{session_id}.json",
                    trajectory=AtifTrajectory(
                        session_id=session_id,
                        agent=AtifAgent(
                            model=model,
                            exo_extensions={"cluster": cluster_info}
                            if cluster_info
                            else None,
                        ),
                    ),
                    last_prefix_len=0,
                )
                self._sessions[session_id] = state
            self._append_new_input_messages(state, messages, client_key)
            self._evict()
            return state.session_id

    def _append_new_input_messages(
        self,
        state: SessionState,
        messages: list[ChatCompletionMessage],
        client_key: str,
    ) -> None:
        new_messages = messages[state.last_prefix_len :]
        if len(new_messages) > 3 and state.last_prefix_len > 0:
            logger.warning(
                "trajectory {}: prefix match fell back — {} new messages "
                "(request len={}, last known prefix len={}). Likely system "
                "prompt or message format changed between calls.",
                state.session_id,
                len(new_messages),
                len(messages),
                state.last_prefix_len,
            )
        trajectory = state.trajectory
        for msg in new_messages:
            if msg.role == "assistant" and _matches_last_agent_step(trajectory, msg):
                continue
            next_id = len(trajectory.steps) + 1
            step, tool_result = _message_to_step(msg, next_id)
            if step is not None:
                trajectory.steps.append(step)
            elif tool_result is not None:
                call_id, content = tool_result
                prev_agent = next(
                    (s for s in reversed(trajectory.steps) if s.source == "agent"),
                    None,
                )
                if prev_agent is not None:
                    if prev_agent.observation is None:
                        prev_agent.observation = AtifObservation(results=[])
                    prev_agent.observation.results.append(
                        AtifObservationResult(
                            source_call_id=call_id, content=content
                        )
                    )
        state.last_prefix_len = len(messages)
        self._reindex_state(state, client_key, messages)

    def _reindex_state(
        self,
        state: SessionState,
        client_key: str,
        messages: list[ChatCompletionMessage],
    ) -> None:
        prefix_hash = _hash_messages(messages)
        key = (client_key, prefix_hash)
        self._by_hash[key] = state
        state.hash_keys.add(key)
        self._sessions.move_to_end(state.session_id)

    def record_response(
        self,
        session_id: str,
        client_key: str,
        request_messages: list[ChatCompletionMessage],
        assistant_text: str,
        reasoning_content: str | None,
        tool_calls: list[ToolCallItem],
        stats: GenerationStats | None,
        model: str,
        usage: Usage | None = None,
        ttft_ms: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        is_empty = (
            not assistant_text
            and not tool_calls
            and not reasoning_content
            and stats is None
            and usage is None
            and ttft_ms is None
        )
        if is_empty:
            return
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            trajectory = state.trajectory
            atif_tool_calls: list[AtifToolCall] | None = None
            if tool_calls:
                atif_tool_calls = []
                for tc in tool_calls:
                    try:
                        args = cast(dict[str, Any], json.loads(tc.arguments))
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": tc.arguments}
                    atif_tool_calls.append(
                        AtifToolCall(
                            tool_call_id=tc.id,
                            function_name=tc.name,
                            arguments=args,
                        )
                    )
            metrics: AtifStepMetrics | None = None
            cached_tokens = (
                usage.prompt_tokens_details.cached_tokens if usage is not None else 0
            )
            if stats is not None:
                metrics = AtifStepMetrics(
                    prompt_tokens=stats.prompt_tokens,
                    completion_tokens=stats.generation_tokens,
                    cached_tokens=cached_tokens,
                    cost=0.0,
                    exo_extensions=AtifExoExtensions(
                        prompt_tps=stats.prompt_tps,
                        generation_tps=stats.generation_tps,
                        peak_memory_bytes=stats.peak_memory_usage.in_bytes,
                        prefix_cache_hit=stats.prefix_cache_hit,
                        reasoning_content=reasoning_content,
                        ttft_ms=ttft_ms,
                    ),
                )
            elif reasoning_content or ttft_ms is not None:
                metrics = AtifStepMetrics(
                    cached_tokens=cached_tokens,
                    exo_extensions=AtifExoExtensions(
                        reasoning_content=reasoning_content,
                        ttft_ms=ttft_ms,
                    ),
                )
            step = AtifStep(
                step_id=len(trajectory.steps) + 1,
                timestamp=_iso_now(),
                source="agent",
                message=assistant_text,
                reasoning_content=reasoning_content,
                tool_calls=atif_tool_calls,
                metrics=metrics,
                model_name=model,
            )
            trajectory.steps.append(step)
            trajectory.final_metrics = _recompute_final_metrics(trajectory.steps)

            assistant_msg_tool_calls: list[ToolCall] | None = None
            if tool_calls:
                assistant_msg_tool_calls = [
                    ToolCall(id=tc.id, function=tc) for tc in tool_calls
                ]
            assistant_msg = ChatCompletionMessage(
                role="assistant",
                content=assistant_text or None,
                reasoning_content=reasoning_content,
                tool_calls=assistant_msg_tool_calls,
            )
            expected_prefix = request_messages + [assistant_msg]
            self._reindex_state(state, client_key, expected_prefix)
            state.last_prefix_len = len(expected_prefix)
            state.last_activity = time.monotonic()
            _atomic_write(state.file_path, trajectory)


def _recompute_final_metrics(steps: list[AtifStep]) -> AtifFinalMetrics:
    total_prompt = sum(s.metrics.prompt_tokens for s in steps if s.metrics is not None)
    total_completion = sum(
        s.metrics.completion_tokens for s in steps if s.metrics is not None
    )
    total_cost = sum(s.metrics.cost for s in steps if s.metrics is not None)
    return AtifFinalMetrics(
        total_steps=len(steps),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cost=total_cost,
    )


def _atomic_write(path: Path, trajectory: AtifTrajectory) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid4().hex[:8]}")
        tmp.write_text(
            trajectory.model_dump_json(by_alias=True, exclude_none=True, indent=2)
        )
        tmp.replace(path)
    except OSError as e:
        logger.warning("failed to write trajectory {}: {}", path, e)


async def tap_chunk_stream(
    chunk_stream: AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ],
    on_complete: Callable[
        [
            str,
            str | None,
            list[ToolCallItem],
            GenerationStats | None,
            Usage | None,
            float | None,
        ],
        None,
    ],
) -> AsyncGenerator[
    PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
]:
    """Passes chunks through verbatim while accumulating the assistant response.

    Invokes `on_complete(text, reasoning, tool_calls, stats, usage, ttft_ms)`
    once when the underlying stream terminates (normal or exceptional). TTFT is
    measured from tap entry to the first produced token/tool-call chunk.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCallItem] = []
    stats: GenerationStats | None = None
    last_usage: Usage | None = None
    tap_start = time.perf_counter()
    ttft_ms: float | None = None
    chunk_idx = 0
    try:
        async for chunk in chunk_stream:
            chunk_idx += 1
            match chunk:
                case TokenChunk():
                    if ttft_ms is None and chunk.text:
                        ttft_ms = (time.perf_counter() - tap_start) * 1000.0
                    if chunk.is_thinking:
                        thinking_parts.append(chunk.text)
                    else:
                        text_parts.append(chunk.text)
                    if chunk.stats is not None:
                        stats = chunk.stats
                    if chunk.usage is not None:
                        last_usage = chunk.usage
                    logger.debug(
                        "[tap] C#{} TokenChunk finish={} stats={} usage={}",
                        chunk_idx,
                        chunk.finish_reason,
                        "Y" if chunk.stats else "N",
                        "Y" if chunk.usage else "N",
                    )
                case ToolCallChunk():
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - tap_start) * 1000.0
                    tool_calls.extend(chunk.tool_calls)
                    if chunk.stats is not None:
                        stats = chunk.stats
                    if chunk.usage is not None:
                        last_usage = chunk.usage
                    logger.info(
                        "[tap] C#{} ToolCallChunk tools={} stats={} usage={}",
                        chunk_idx,
                        [tc.name for tc in chunk.tool_calls],
                        "Y" if chunk.stats else "N",
                        "Y" if chunk.usage else "N",
                    )
                case _:
                    pass
            yield chunk
    finally:
        logger.info(
            "[tap] stream ended: {} chunks, text_len={}, tool_calls={}, "
            "stats={}, usage={}, ttft={}",
            chunk_idx,
            sum(len(p) for p in text_parts),
            len(tool_calls),
            "Y" if stats else "N",
            "Y" if last_usage else "N",
            f"{ttft_ms:.0f}ms" if ttft_ms else "None",
        )
        try:
            on_complete(
                "".join(text_parts),
                "".join(thinking_parts) if thinking_parts else None,
                tool_calls,
                stats,
                last_usage,
                ttft_ms,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("trajectory record_response failed: {}", e)


class TrajectoryListItem(CamelCaseModel):
    session_id: str
    created_at: str
    updated_at: str
    total_steps: int
    model: str


class TrajectoryListResponse(CamelCaseModel):
    trajectories: list[TrajectoryListItem]


class DeleteTrajectoriesRequest(CamelCaseModel):
    session_ids: list[str]


class DeleteTrajectoriesResponse(CamelCaseModel):
    deleted: list[str]
    not_found: list[str]


_global_collector: TrajectoryCollector | None = None
_collector_lock = threading.Lock()


def _env_enabled() -> bool:
    return os.environ.get("EXO_TRAJECTORIES", "false").lower() == "true"


def get_collector() -> TrajectoryCollector:
    global _global_collector
    with _collector_lock:
        if _global_collector is None:
            _global_collector = TrajectoryCollector(
                enabled=_env_enabled(),
                directory=EXO_TRAJECTORIES_DIR,
            )
        else:
            _global_collector.enabled = _env_enabled()
            if _global_collector.enabled:
                try:
                    _global_collector.directory.mkdir(
                        parents=True, exist_ok=True
                    )
                except OSError as e:
                    logger.warning(
                        "could not create trajectories dir {}: {}",
                        _global_collector.directory,
                        e,
                    )
        return _global_collector


def set_collector(collector: TrajectoryCollector) -> None:
    global _global_collector
    with _collector_lock:
        _global_collector = collector
