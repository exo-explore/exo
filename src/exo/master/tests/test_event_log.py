from pathlib import Path

import pytest

from exo.master.event_log import DiskEventLog
from exo.shared.types.events import TestEvent


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "event_log"


def test_append_and_read_back(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(5)]
    for e in events:
        log.append(e)

    assert len(log) == 5

    result = list(log.read_all())
    assert len(result) == 5
    for original, restored in zip(events, result, strict=True):
        assert original.event_id == restored.event_id

    log.close()


def test_read_range(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(10)]
    for e in events:
        log.append(e)

    result = list(log.read_range(3, 7))
    assert len(result) == 4
    for i, restored in enumerate(result):
        assert events[3 + i].event_id == restored.event_id

    log.close()


def test_read_range_bounds(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(3)]
    for e in events:
        log.append(e)

    # Start beyond count
    assert list(log.read_range(5, 10)) == []
    # Negative start
    assert list(log.read_range(-1, 2)) == []
    # End beyond count is clamped
    result = list(log.read_range(1, 100))
    assert len(result) == 2

    log.close()


def test_empty_log(log_dir: Path):
    log = DiskEventLog(log_dir)
    assert len(log) == 0
    assert list(log.read_all()) == []
    assert list(log.read_range(0, 10)) == []
    log.close()


def test_rotation_on_close(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()

    active = log_dir / "events.bin"
    archive = log_dir / "events.1.bin.zst"
    assert not active.exists()
    assert archive.exists()
    assert archive.stat().st_size > 0


def test_rotation_on_construction_with_stale_file(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    active = log_dir / "events.bin"
    active.write_bytes(b"stale data")

    log = DiskEventLog(log_dir)
    archive = log_dir / "events.1.bin.zst"
    assert archive.exists()
    assert len(log) == 0

    log.close()


def test_empty_log_no_archive(log_dir: Path):
    """Closing an empty log should not leave an archive."""
    log = DiskEventLog(log_dir)
    log.close()

    active = log_dir / "events.bin"
    archive = log_dir / "events.1.bin.zst"
    assert not active.exists()
    assert not archive.exists()


def test_close_is_idempotent(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()
    log.close()  # should not raise

    archive = log_dir / "events.1.bin.zst"
    assert archive.exists()


def test_successive_sessions(log_dir: Path):
    """Simulate two master sessions: both archives should be kept."""
    log1 = DiskEventLog(log_dir)
    log1.append(TestEvent())
    log1.close()

    assert (log_dir / "events.1.bin.zst").exists()

    log2 = DiskEventLog(log_dir)
    log2.append(TestEvent())
    log2.append(TestEvent())
    log2.close()

    # Session 1 archive shifted to slot 2, session 2 in slot 1
    assert (log_dir / "events.1.bin.zst").exists()
    assert (log_dir / "events.2.bin.zst").exists()


def test_rotation_keeps_at_most_5_archives(log_dir: Path):
    """After 7 sessions, only the 5 most recent archives should remain."""
    for _ in range(7):
        log = DiskEventLog(log_dir)
        log.append(TestEvent())
        log.close()

    for i in range(1, 6):
        assert (log_dir / f"events.{i}.bin.zst").exists()
    assert not (log_dir / "events.6.bin.zst").exists()
    assert not (log_dir / "events.7.bin.zst").exists()
