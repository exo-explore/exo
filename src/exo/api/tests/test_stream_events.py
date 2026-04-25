# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the GET /events endpoint cursor support.

The handler at exo/api/main.py:894 supports two query parameters:
  - since: int (default 0) — start index, inclusive
  - limit: int | None (default None) — max events to return

The response sets `X-EXO-Last-Idx` to the upper bound consumed, allowing
clients to chain reads without a separate /state round-trip.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.events import TestEvent
from exo.utils.disk_event_log import DiskEventLog


def _make_api(log_dir: Path, n_events: int) -> Any:
    """Create a minimal API with a DiskEventLog containing n_events records
    and only the GET /events route mounted."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    log = DiskEventLog(log_dir)
    for _ in range(n_events):
        log.append(TestEvent())
    api._event_log = log  # pyright: ignore[reportPrivateUsage]

    app.get("/events")(api.stream_events)
    return api


def test_stream_events_full_dump_backward_compatible(tmp_path: Path) -> None:
    """No params -> full ledger; X-EXO-Last-Idx equals count."""
    api = _make_api(tmp_path / "log_full", n_events=5)
    client = TestClient(api.app)

    resp = client.get("/events")
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 5
    assert resp.headers["X-EXO-Last-Idx"] == "5"


def test_stream_events_with_since_and_limit(tmp_path: Path) -> None:
    """since=N&limit=M returns events in [since, since+M); header reflects bound."""
    api = _make_api(tmp_path / "log_cursor", n_events=10)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 3, "limit": 4})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 4
    assert resp.headers["X-EXO-Last-Idx"] == "7"


def test_stream_events_since_only_reads_to_end(tmp_path: Path) -> None:
    """since with no limit returns [since, count); header equals count."""
    api = _make_api(tmp_path / "log_tail", n_events=8)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 5})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 3
    assert resp.headers["X-EXO-Last-Idx"] == "8"


def test_stream_events_since_beyond_count_returns_empty(tmp_path: Path) -> None:
    """since past end yields []; header reflects clamped end."""
    api = _make_api(tmp_path / "log_overshoot", n_events=4)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 99})
    assert resp.status_code == 200
    assert resp.json() == []
    # end is clamped to log_count (4) since limit is None and since > count
    assert resp.headers["X-EXO-Last-Idx"] == "4"


def test_stream_events_limit_larger_than_remaining(tmp_path: Path) -> None:
    """limit > remaining is clamped to log_count; no error."""
    api = _make_api(tmp_path / "log_clamp", n_events=10)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 7, "limit": 100})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 3
    assert resp.headers["X-EXO-Last-Idx"] == "10"


def test_stream_events_negative_since_rejected(tmp_path: Path) -> None:
    """FastAPI Query(ge=0) rejects negative since with 422."""
    api = _make_api(tmp_path / "log_neg", n_events=3)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": -1})
    assert resp.status_code == 422


def test_stream_events_chained_cursor_reads(tmp_path: Path) -> None:
    """Two sequential reads using returned cursor cover full ledger without overlap."""
    api = _make_api(tmp_path / "log_chain", n_events=6)
    client = TestClient(api.app)

    first = client.get("/events", params={"since": 0, "limit": 4})
    assert first.status_code == 200
    cursor = int(first.headers["X-EXO-Last-Idx"])
    assert cursor == 4
    first_data: list[dict[str, Any]] = first.json()
    assert len(first_data) == 4

    second = client.get("/events", params={"since": cursor})
    assert second.status_code == 200
    second_data: list[dict[str, Any]] = second.json()
    assert len(second_data) == 2
    assert second.headers["X-EXO-Last-Idx"] == "6"

    # Two reads cover the full ledger with no gap and no overlap.
    assert len(first_data) + len(second_data) == 6
