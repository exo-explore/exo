"""Typed wrapper around ``/bench/chat/completions`` for benchmarks.

The bench endpoint disables EOS suppression and KV prefix caching by
default (see ``bench/METHODOLOGY.md``). This module exposes a single
function :func:`run_one_completion` that:

  1. Builds an exact-token-length prompt via :class:`PromptSizer`.
  2. POSTs to ``/bench/chat/completions``.
  3. Returns a ``(BenchRow, prompt_tokens)`` pair where ``BenchRow`` is a
     :class:`typing.TypedDict` with the fields the caller needs.

Streaming is supported but rarely needed for context-scaling — the
non-streaming path is the default.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Literal, NotRequired, TypedDict, cast

from exo_tools.client import ExoClient

from .prompt import PromptSizer

PrefixCacheHit = Literal["none", "partial", "exact"]


class GenerationStats(TypedDict, total=False):
    """Server-reported per-task timing stats."""

    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: dict[str, int]
    prefix_cache_hit: PrefixCacheHit


class BenchRow(TypedDict):
    """Per-request result row returned to callers."""

    elapsed_s: float
    output_text_preview: str
    stats: GenerationStats
    error: NotRequired[str]


def _as_dict(value: Any) -> dict[str, Any]:  # type: ignore[reportAny]
    """Narrow an arbitrary JSON value to a typed ``dict[str, Any]``."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _as_list(value: Any) -> list[Any]:  # type: ignore[reportAny]
    if isinstance(value, list):
        return cast("list[Any]", value)
    return []


def _extract_stats(raw_response: dict[str, Any]) -> GenerationStats:
    stats_obj = raw_response.get("generation_stats")
    if not isinstance(stats_obj, dict):
        return {}
    return cast("GenerationStats", cast("object", stats_obj))


def _extract_preview(raw_response: dict[str, Any], limit: int = 200) -> str:
    choices = _as_list(raw_response.get("choices"))
    if not choices:
        return ""
    first = _as_dict(choices[0])
    message = _as_dict(first.get("message"))
    content_obj = message.get("content")
    if isinstance(content_obj, str):
        return content_obj[:limit]
    return ""


def run_one_completion(
    client: ExoClient,
    model_id: str,
    pp_hint: int,
    tg: int,
    prompt_sizer: PromptSizer,
    *,
    use_prefix_cache: bool = False,
    stream: bool = False,
) -> tuple[BenchRow, int]:
    """Send one request to ``/bench/chat/completions`` and return its row.

    ``pp_hint`` is the *target* prompt-token count; the actual prompt is
    sized via :class:`PromptSizer` and the verified value is returned as
    the second element of the tuple.
    """
    content, pp_tokens = prompt_sizer.build(pp_hint)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": tg,
        "logprobs": False,
        "use_prefix_cache": use_prefix_cache,
    }

    if not stream:
        payload["stream"] = False
        t0 = time.perf_counter()
        raw_obj = client.post_bench_chat_completions(payload)
        elapsed = time.perf_counter() - t0
        raw = _as_dict(raw_obj)
        return (
            BenchRow(
                elapsed_s=elapsed,
                output_text_preview=_extract_preview(raw),
                stats=_extract_stats(raw),
            ),
            pp_tokens,
        )

    return _run_streaming(client, payload, pp_tokens)


def _run_streaming(
    client: ExoClient,
    payload: dict[str, Any],
    pp_tokens: int,
) -> tuple[BenchRow, int]:
    """Streaming variant: parse SSE lines, recover ``GenerationStats``."""
    payload = {**payload, "stream": True}

    tokens = 0
    first_token_time: float | None = None
    t0 = time.perf_counter()
    text_parts: list[str] = []
    stats: GenerationStats = {}

    for raw_line in client.stream_bench_chat_completions(payload):
        line = raw_line.strip()
        if line.startswith(": generation_stats "):
            with contextlib.suppress(json.JSONDecodeError):
                parsed_obj: Any = json.loads(  # type: ignore[reportAny]
                    line[len(": generation_stats ") :]
                )
                if isinstance(parsed_obj, dict):
                    stats = cast("GenerationStats", cast("object", parsed_obj))
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk_obj: Any = json.loads(data)  # type: ignore[reportAny]
        except json.JSONDecodeError:
            continue
        chunk = _as_dict(chunk_obj)
        choices = _as_list(chunk.get("choices"))
        if not choices:
            continue
        first = _as_dict(choices[0])
        delta = _as_dict(first.get("delta"))
        delta_content_obj = delta.get("content")
        if isinstance(delta_content_obj, str) and delta_content_obj:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens += 1
            text_parts.append(delta_content_obj)

    elapsed = time.perf_counter() - t0
    preview = "".join(text_parts)[:200]

    if not stats:
        ttft = (first_token_time - t0) if first_token_time is not None else elapsed
        gen_time = elapsed - ttft if tokens > 1 else elapsed
        gen_tps = (tokens - 1) / gen_time if tokens > 1 and gen_time > 0 else 0.0
        prompt_tps = pp_tokens / ttft if ttft > 0 else 0.0
        stats = GenerationStats(
            prompt_tokens=pp_tokens,
            generation_tokens=tokens,
            prompt_tps=round(prompt_tps, 2),
            generation_tps=round(gen_tps, 2),
            peak_memory_usage={"inBytes": 0},
        )

    return (
        BenchRow(
            elapsed_s=elapsed,
            output_text_preview=preview,
            stats=stats,
        ),
        pp_tokens,
    )
