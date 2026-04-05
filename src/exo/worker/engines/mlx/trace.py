"""Microsecond-resolution request tracing for the MLX inference pipeline.

Usage:
    from exo.worker.engines.mlx.trace import request_trace, T

    # Start a new trace for each request
    request_trace.reset()

    # Record spans (context manager)
    with T("prefill.chunk.forward"):
        ...

    # Record spans (manual)
    t0 = time.perf_counter()
    ...
    request_trace.record("decode.tolist", t0)

    # Dump at end of request
    request_trace.dump()

All tracing is gated on EXO_TRACING_ENABLED=1. When disabled, all operations
are no-ops with near-zero overhead.
"""

import os
import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger

ENABLED = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")


class _Span:
    __slots__ = ("name", "start", "end")

    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start
        self.end = end


class RequestTrace:
    """Accumulates spans for a single request lifecycle."""

    def __init__(self) -> None:
        self._spans: list[_Span] = []
        self._t0: float = 0.0
        self._active: bool = False

    def reset(self) -> None:
        """Start a new trace. Call at the beginning of each request."""
        self._spans.clear()
        self._t0 = time.perf_counter()
        self._active = ENABLED

    @contextmanager
    def span(self, name: str) -> Generator[None]:
        """Context manager to record a named span."""
        if not self._active:
            yield
            return
        start = time.perf_counter()
        yield
        self._spans.append(_Span(name, start, time.perf_counter()))

    def record(self, name: str, start: float, end: float | None = None) -> None:
        """Record a span from explicit start/end perf_counter values."""
        if not self._active:
            return
        self._spans.append(_Span(name, start, end if end is not None else time.perf_counter()))

    def mark(self, name: str) -> None:
        """Record a zero-duration marker at the current time."""
        if not self._active:
            return
        now = time.perf_counter()
        self._spans.append(_Span(name, now, now))

    def dump(self) -> None:
        """Log the full trace timeline, sorted by start time."""
        if not self._active or not self._spans:
            return

        lines: list[str] = ["[TRACE] Request timeline:"]
        for s in sorted(self._spans, key=lambda x: x.start):
            offset_us = (s.start - self._t0) * 1_000_000
            dur_us = (s.end - s.start) * 1_000_000

            # Format offset
            if offset_us >= 1_000_000:
                offset_str = f"{offset_us / 1_000_000:.3f}s"
            elif offset_us >= 1000:
                offset_str = f"{offset_us / 1000:.1f}ms"
            else:
                offset_str = f"{offset_us:.0f}µs"

            # Format duration
            if dur_us < 1:
                dur_str = "---"  # marker
            elif dur_us >= 1_000_000:
                dur_str = f"+{dur_us / 1_000_000:.3f}s"
            elif dur_us >= 1000:
                dur_str = f"+{dur_us / 1000:.3f}ms"
            else:
                dur_str = f"+{dur_us:.0f}µs"

            lines.append(f"  {offset_str:>12s}  {dur_str:>12s}  {s.name}")

        logger.info("\n".join(lines))

    @property
    def active(self) -> bool:
        return self._active


# Module-level singleton — one trace per runner process.
request_trace = RequestTrace()


@contextmanager
def T(name: str) -> Generator[None]:
    """Shorthand for request_trace.span(name)."""
    if not request_trace._active:
        yield
        return
    start = time.perf_counter()
    yield
    request_trace._spans.append(_Span(name, start, time.perf_counter()))
