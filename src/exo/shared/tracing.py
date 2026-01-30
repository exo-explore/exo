from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final

from exo.shared.constants import EXO_TRACING_ENABLED

logger = logging.getLogger(__name__)

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


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return EXO_TRACING_ENABLED


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
    if not is_tracing_enabled():
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


def export_local_traces(rank: int) -> None:
    if not is_tracing_enabled():
        return

    local_traces = get_trace_buffer()
    if local_traces:
        output_path = Path.home() / ".exo" / "traces" / f"trace_{rank}.json"
        try:
            export_trace(local_traces, output_path)
        except Exception as e:
            logger.warning("Failed to export local traces for rank %d: %s", rank, e)

    clear_trace_buffer()


def merge_trace_files(trace_dir: Path | None = None) -> Path | None:
    if trace_dir is None:
        trace_dir = Path.home() / ".exo" / "traces"

    if not trace_dir.exists():
        return None

    trace_files = sorted(trace_dir.glob("trace_*.json"))

    if not trace_files:
        return None

    merged_events: list[dict[str, object]] = []
    for trace_file in trace_files:
        file_rank = int(trace_file.stem.split("_")[1])

        with open(trace_file) as f:
            raw = f.read()
            data = cast(dict[str, list[dict[str, object]]], json.loads(raw))
            events: list[dict[str, object]] = data.get("traceEvents", [])
            for event in events:
                event["tid"] = file_rank
                if "args" in event and isinstance(event["args"], dict):
                    event["args"]["rank"] = file_rank
            merged_events.extend(events)

    output_path = Path.home() / ".exo" / "traces" / "merged_trace.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"traceEvents": merged_events}, f, indent=2)
    except OSError as e:
        logger.warning("Failed to write merged trace to %s: %s", output_path, e)
        return None

    return output_path


def _format_duration(us: int | float) -> str:
    if us < 1000:
        return f"{us:.0f}µs"
    elif us < 1_000_000:
        return f"{us / 1000:.2f}ms"
    else:
        return f"{us / 1_000_000:.2f}s"


def load_trace_file(path: Path) -> list[TraceEvent]:
    """Load a Chrome Trace Format JSON file into TraceEvent objects."""
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
    """Compute comprehensive statistics from trace events."""
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


def print_stats(stats: TraceStats) -> None:
    """Print formatted trace statistics."""
    print("=== Trace Statistics ===")
    print()
    print(f"Wall Time: {_format_duration(stats.total_wall_time_us)}")
    print()

    # Parse hierarchical categories (e.g., "sync/compute" -> phase="sync", subcat="compute")
    if stats.by_category:
        phases: dict[str, dict[str, CategoryStats]] = defaultdict(dict)
        has_hierarchical = False

        for cat, cat_stats in stats.by_category.items():
            if "/" in cat:
                phase, subcat = cat.split("/", 1)
                phases[phase][subcat] = cat_stats
                has_hierarchical = True
            else:
                phases[cat]["_total"] = cat_stats

        if has_hierarchical:
            print("By Phase:")
            for phase in sorted(phases.keys()):
                subcats = phases[phase]
                # Skip phases that only have _total (non-hierarchical top-level categories)
                non_total_subcats = {k: v for k, v in subcats.items() if k != "_total"}
                if not non_total_subcats:
                    continue

                phase_total = sum(s.total_us for s in non_total_subcats.values())
                print(f"  {phase}:")
                for subcat, subcat_stats in sorted(
                    non_total_subcats.items(),
                    key=lambda x: x[1].total_us,
                    reverse=True,
                ):
                    pct = (
                        subcat_stats.total_us / phase_total * 100 if phase_total else 0
                    )
                    # Use parent phase's step count for per-step average
                    phase_step_count = subcats.get("_total", CategoryStats()).count
                    if phase_step_count > 0:
                        avg_per_step = subcat_stats.total_us / phase_step_count
                    else:
                        avg_per_step = subcat_stats.avg_us  # fallback
                    print(
                        f"    {subcat:12s} {_format_duration(subcat_stats.total_us):>10s} "
                        f"({pct:5.1f}%)  avg: {_format_duration(avg_per_step)}"
                    )
            print()
        else:
            # Fall back to flat category display if no hierarchical categories
            print("By Category:")
            total_time = sum(c.total_us for c in stats.by_category.values())
            for category, cat_stats in sorted(
                stats.by_category.items(), key=lambda x: x[1].total_us, reverse=True
            ):
                pct = (cat_stats.total_us / total_time * 100) if total_time > 0 else 0
                print(
                    f"  {category:12s} {_format_duration(cat_stats.total_us):>10s} "
                    f"({pct:5.1f}%)  avg: {_format_duration(cat_stats.avg_us):>8s}  "
                    f"count: {cat_stats.count}"
                )
            print()

    # By Rank
    if stats.by_rank:
        print("By Rank:")
        for rank in sorted(stats.by_rank.keys()):
            rank_stats = stats.by_rank[rank]
            print(f"  Rank {rank}:")

            # Parse hierarchical categories for this rank
            rank_phases: dict[str, dict[str, CategoryStats]] = defaultdict(dict)
            has_hierarchical = False
            for cat, cat_stats in rank_stats.items():
                if "/" in cat:
                    phase, subcat = cat.split("/", 1)
                    rank_phases[phase][subcat] = cat_stats
                    has_hierarchical = True
                else:
                    rank_phases[cat]["_total"] = cat_stats

            if has_hierarchical:
                for phase in sorted(rank_phases.keys()):
                    subcats = rank_phases[phase]
                    non_total_subcats = {
                        k: v for k, v in subcats.items() if k != "_total"
                    }
                    if not non_total_subcats:
                        continue

                    phase_total = sum(s.total_us for s in non_total_subcats.values())
                    print(f"    {phase}:")
                    for subcat, subcat_stats in sorted(
                        non_total_subcats.items(),
                        key=lambda x: x[1].total_us,
                        reverse=True,
                    ):
                        pct = (
                            subcat_stats.total_us / phase_total * 100
                            if phase_total
                            else 0
                        )
                        # Use parent phase's step count for per-step average
                        phase_step_count = subcats.get("_total", CategoryStats()).count
                        if phase_step_count > 0:
                            avg_per_step = subcat_stats.total_us / phase_step_count
                        else:
                            avg_per_step = subcat_stats.avg_us  # fallback
                        print(
                            f"      {subcat:12s} {_format_duration(subcat_stats.total_us):>10s} "
                            f"({pct:5.1f}%)  avg: {_format_duration(avg_per_step)}"
                        )
            else:
                # Flat display fallback
                for category, cat_stats in sorted(
                    rank_stats.items(), key=lambda x: x[1].total_us, reverse=True
                ):
                    print(f"    {category}: {_format_duration(cat_stats.total_us)}")
        print()


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trace.json")

    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    traces = load_trace_file(path)
    if not traces:
        print("No trace events found in file.")
        sys.exit(0)

    computed_stats = compute_stats(traces)
    print_stats(computed_stats)
