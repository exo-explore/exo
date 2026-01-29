from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

_TRACING_ENV_VAR: Final[str] = "EXO_TRACING_ENABLED"


@dataclass(frozen=True)
class TraceEvent:
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


# Global trace buffer - each rank accumulates traces here
_trace_buffer: list[TraceEvent] = []


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return os.environ.get(_TRACING_ENV_VAR, "").lower() in ("1", "true", "yes")


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

    Args:
        name: Name of the operation (e.g., "recv 0", "send 1", "joint_blocks")
        rank: This rank's ID
        category: Category for grouping in trace viewer ("comm", "compute", "step")

    Example:
        with trace_span(f"recv {from_rank}", rank, "comm"):
            hidden_states = mx.distributed.recv(...)
            mx.eval(hidden_states)

        with trace_span("joint_blocks", rank):
            hidden_states = some_computation(...)
    """
    if not is_tracing_enabled():
        yield
        return

    start_us = int(time.time() * 1_000_000)
    yield
    duration_us = int(time.time() * 1_000_000) - start_us
    _record_span(name, start_us, duration_us, rank, category)


def get_trace_buffer() -> list[TraceEvent]:
    return list(_trace_buffer)


def clear_trace_buffer() -> None:
    _trace_buffer.clear()


def export_trace(traces: list[TraceEvent], output_path: Path) -> None:
    trace_events: list[dict[str, object]] = []

    for trace in traces:
        # Chrome trace format uses "X" for complete events (with duration)
        event: dict[str, object] = {
            "name": trace.name,
            "cat": trace.category,
            "ph": "X",
            "ts": trace.start_us,
            "dur": trace.duration_us,
            "pid": 0,
            "tid": trace.rank,
            "args": {"rank": trace.rank},
        }
        trace_events.append(event)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chrome_trace, f, indent=2)


def export_local_traces(rank: int) -> None:
    if not is_tracing_enabled():
        return

    local_traces = get_trace_buffer()
    if local_traces:
        output_path = Path.home() / ".exo" / "traces" / f"trace_{rank}.json"
        export_trace(local_traces, output_path)

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

    output_path = Path("trace.json")
    with open(output_path, "w") as f:
        json.dump({"traceEvents": merged_events}, f, indent=2)

    return output_path
