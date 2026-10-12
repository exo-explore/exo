"""-q shows only warnings and errors on the console, -v shows debug logs, and the default shows info."""

import sys
from collections.abc import Iterator

import pytest
from loguru import logger

from exo.main import Args
from exo.shared import logging as exo_logging


@pytest.fixture(autouse=True)
def restore_default_logger() -> Iterator[None]:
    yield
    logger.remove()
    logger.add(sys.stderr)


def console_output_at(verbosity: int, capfd: pytest.CaptureFixture[str]) -> str:
    exo_logging.logger_setup(None, verbosity)
    logger.debug("a debug line")
    logger.info("an info line")
    logger.warning("a warning line")
    logger.error("an error line")
    exo_logging.logger_cleanup()
    return capfd.readouterr().err


def test_quiet_shows_only_warnings_and_errors(capfd: pytest.CaptureFixture[str]):
    output = console_output_at(-1, capfd)

    assert "a warning line" in output
    assert "an error line" in output
    assert "an info line" not in output
    assert "a debug line" not in output


def test_default_shows_info(capfd: pytest.CaptureFixture[str]):
    output = console_output_at(0, capfd)

    assert "an info line" in output
    assert "a debug line" not in output


def test_verbose_shows_debug(capfd: pytest.CaptureFixture[str]):
    output = console_output_at(1, capfd)

    assert "a debug line" in output
    assert "an info line" in output


@pytest.mark.parametrize(
    ("argv", "expected_level"),
    [
        (["exo", "-q"], "WARNING"),
        (["exo", "--quiet"], "WARNING"),
        (["exo"], "INFO"),
        (["exo", "-v"], "DEBUG"),
        (["exo", "-vv"], "DEBUG"),
    ],
)
def test_flags_choose_the_console_level(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_level: str
):
    monkeypatch.setattr(sys, "argv", argv)
    assert exo_logging.console_log_level(Args.parse().verbosity) == expected_level
