from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final

from exo.shared.constants import EXO_TRACING_ENABLED
from exo.worker.runner.bootstrap import logger

# Context variable to track the current trace category for hierarchical nesting
_current_category: ContextVar[str | None] = ContextVar("current_category", default=None)


@final
@dataclass(frozen=True)
class TraceEvent:
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


@final
@dataclass
class CategoryStats:
    total_us: int = 0
    count: int = 0
    min_us: int = 0
    max_us: int = 0

    def add(self, duration_us: int) -> None:
        if self.count == 0:
            self.min_us = duration_us
            self.max_us = duration_us
        else:
            self.min_us = min(self.min_us, duration_us)
            self.max_us = max(self.max_us, duration_us)
        self.total_us += duration_us
        self.count += 1

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count > 0 else 0.0


@final
@dataclass
class TraceStats:
    total_wall_time_us: int = 0
    by_category: dict[str, CategoryStats] = field(default_factory=dict)
    by_rank: dict[int, dict[str, CategoryStats]] = field(default_factory=dict)


# Global trace buffer - each rank accumulates traces here
_trace_buffer: list[TraceEvent] = []


def _record_span(
    name: str, start_us: int, duration_us: int, rank: int, category: str
) -> None:
    _trace_buffer.append(
        TraceEvent(
            name=name,
            start_us=start_us,
            duration_us=duration_us,
            rank=rank,
            category=category,
        )
    )


@contextmanager
def trace(
    name: str,
    rank: int,
    category: str = "compute",
) -> Generator[None, None, None]:
    """Context manager to trace any operation.

    Nested traces automatically inherit the parent category, creating hierarchical
    categories like "sync/compute" or "async/comms".

    Args:
        name: Name of the operation (e.g., "recv 0", "send 1", "joint_blocks")
        rank: This rank's ID
        category: Category for grouping in trace viewer ("comm", "compute", "step")

    Example:
        with trace(f"sync {t}", rank, "sync"):
            with trace("joint_blocks", rank, "compute"):
                # Recorded with category "sync/compute"
                hidden_states = some_computation(...)
    """
    if not EXO_TRACING_ENABLED:
        yield
        return

    # Combine with parent category if nested
    parent = _current_category.get()
    full_category = f"{parent}/{category}" if parent else category

    # Set as current for nested traces
    token = _current_category.set(full_category)

    try:
        start_us = int(time.time() * 1_000_000)
        start_perf = time.perf_counter()
        yield
        duration_us = int((time.perf_counter() - start_perf) * 1_000_000)
        _record_span(name, start_us, duration_us, rank, full_category)
    finally:
        _current_category.reset(token)


def get_trace_buffer() -> list[TraceEvent]:
    return list(_trace_buffer)


def clear_trace_buffer() -> None:
    _trace_buffer.clear()


def export_trace(traces: list[TraceEvent], output_path: Path) -> None:
    trace_events: list[dict[str, object]] = []

    for event in traces:
        # Chrome trace format uses "X" for complete events (with duration)
        chrome_event: dict[str, object] = {
            "name": event.name,
            "cat": event.category,
            "ph": "X",
            "ts": event.start_us,
            "dur": event.duration_us,
            "pid": 0,
            "tid": event.rank,
            "args": {"rank": event.rank},
        }
        trace_events.append(chrome_event)

    ranks_seen = set(t.rank for t in traces)
    for rank in ranks_seen:
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",  # Metadata event
                "pid": 0,
                "tid": rank,
                "args": {"name": f"Rank {rank}"},
            }
        )

    chrome_trace = {"traceEvents": trace_events}

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(chrome_trace, f, indent=2)
    except OSError as e:
        logger.warning("Failed to export trace to %s: %s", output_path, e)


def load_trace_file(path: Path) -> list[TraceEvent]:
    with open(path) as f:
        data = cast(dict[str, list[dict[str, object]]], json.load(f))

    events = data.get("traceEvents", [])
    traces: list[TraceEvent] = []

    for event in events:
        # Skip metadata events
        if event.get("ph") == "M":
            continue

        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        ts_value = event.get("ts", 0)
        dur_value = event.get("dur", 0)
        tid_value = event.get("tid", 0)
        start_us = int(ts_value) if isinstance(ts_value, (int, float, str)) else 0
        duration_us = int(dur_value) if isinstance(dur_value, (int, float, str)) else 0

        # Get rank from tid or args
        rank = int(tid_value) if isinstance(tid_value, (int, float, str)) else 0
        args = event.get("args")
        if isinstance(args, dict):
            args_dict = cast(dict[str, object], args)
            rank_from_args = args_dict.get("rank")
            if isinstance(rank_from_args, (int, float, str)):
                rank = int(rank_from_args)

        traces.append(
            TraceEvent(
                name=name,
                start_us=start_us,
                duration_us=duration_us,
                rank=rank,
                category=category,
            )
        )

    return traces


def compute_stats(traces: list[TraceEvent]) -> TraceStats:
    stats = TraceStats()

    if not traces:
        return stats

    # Calculate wall time from earliest start to latest end
    min_start = min(t.start_us for t in traces)
    max_end = max(t.start_us + t.duration_us for t in traces)
    stats.total_wall_time_us = max_end - min_start

    # Initialize nested dicts
    by_category: dict[str, CategoryStats] = defaultdict(CategoryStats)
    by_rank: dict[int, dict[str, CategoryStats]] = defaultdict(
        lambda: defaultdict(CategoryStats)
    )

    for event in traces:
        # By category
        by_category[event.category].add(event.duration_us)

        # By rank and category
        by_rank[event.rank][event.category].add(event.duration_us)

    stats.by_category = dict(by_category)
    stats.by_rank = {k: dict(v) for k, v in by_rank.items()}

    return stats
