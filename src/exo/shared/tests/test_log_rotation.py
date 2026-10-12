"""exo's log file is rotated when exo starts and whenever it grows past a size, so a node that runs
for weeks keeps a bounded amount of log on disk."""

import sys
from pathlib import Path

import pytest
from loguru import logger

from exo.shared import logging as exo_logging


@pytest.fixture
def small_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(exo_logging, "_MAX_LOG_BYTES", 20_000)
    log_file = tmp_path / "exo.log"
    exo_logging.logger_setup(log_file, verbosity=0)
    yield log_file
    logger.remove()
    logger.add(sys.stderr)


def test_the_log_is_rotated_once_it_passes_the_size_limit(small_logs: Path):
    for i in range(2_000):
        logger.info(f"request {i}: " + "x" * 100)
    exo_logging.logger_cleanup()

    archives = list(small_logs.parent.glob("exo.*.log.zst"))
    assert len(archives) > 1, "the log was only rotated when exo started"
    # Old logs are kept compressed, up to the archive limit
    assert len(archives) <= exo_logging._MAX_LOG_ARCHIVES  # pyright: ignore[reportPrivateUsage]
    # The current log file stays near the limit
    assert small_logs.stat().st_size <= 20_000 + 1_000
    # Nothing written last is lost
    assert "request 1999" in small_logs.read_text()


def test_a_small_log_is_only_rotated_when_exo_starts(small_logs: Path):
    for i in range(10):
        logger.info(f"request {i}")
    exo_logging.logger_cleanup()

    # The one archive is the previous run's log, set aside when this run started
    assert len(list(small_logs.parent.glob("exo.*.log.zst"))) == 1
    assert "request 9" in small_logs.read_text()
