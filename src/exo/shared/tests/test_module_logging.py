"""Unit tests for per-module log filters and default DEBUG log levels (TODO #27)."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from exo.main import Args
from exo.shared import logging as exo_logging


@pytest.fixture(autouse=True)
def restore_default_logger() -> Iterator[None]:
    yield
    logger.remove()
    logger.add(sys.stderr)


def test_parse_module_log_filters():
    assert exo_logging.parse_module_log_filters(None) == {}
    assert exo_logging.parse_module_log_filters("") == {}
    assert exo_logging.parse_module_log_filters("   ") == {}

    parsed = exo_logging.parse_module_log_filters(
        "exo.routing=DEBUG,exo.master=INFO,exo.download=WARNING"
    )
    assert parsed == {
        "exo.routing": "DEBUG",
        "exo.master": "INFO",
        "exo.download": "WARNING",
    }

    # Test whitespace and case-insensitivity
    parsed_whitespace = exo_logging.parse_module_log_filters(
        "  exo.routing = debug , exo.master = Info , invalid_entry , exo.bad = INVALID "
    )
    assert parsed_whitespace == {
        "exo.routing": "DEBUG",
        "exo.master": "INFO",
    }


def test_create_module_log_filter_prefix_matching():
    filters = {
        "exo.routing": "DEBUG",
        "exo.routing.zenoh": "WARNING",
        "exo.master": "INFO",
    }
    log_filter = exo_logging.create_module_log_filter(filters, base_level="INFO")

    # 1. Exact match
    rec_routing_debug = {"name": "exo.routing", "level": logger.level("DEBUG")}
    assert log_filter(rec_routing_debug) is True

    # 2. Submodule prefix match
    rec_router_debug = {"name": "exo.routing.router", "level": logger.level("DEBUG")}
    assert log_filter(rec_router_debug) is True

    # 3. Longest prefix takes precedence (zenoh is WARNING)
    rec_zenoh_debug = {"name": "exo.routing.zenoh", "level": logger.level("DEBUG")}
    assert log_filter(rec_zenoh_debug) is False

    rec_zenoh_warn = {"name": "exo.routing.zenoh", "level": logger.level("WARNING")}
    assert log_filter(rec_zenoh_warn) is True

    # 4. Fallback to base_level (INFO) for unmatched modules
    rec_download_debug = {"name": "exo.download", "level": logger.level("DEBUG")}
    assert log_filter(rec_download_debug) is False

    rec_download_info = {"name": "exo.download", "level": logger.level("INFO")}
    assert log_filter(rec_download_info) is True


def test_console_logging_with_module_filters(capfd: pytest.CaptureFixture[str]):
    # Default verbosity 0 (base level INFO), but enable DEBUG specifically for exo.routing
    module_filters = {"exo.routing": "DEBUG"}
    exo_logging.logger_setup(None, verbosity=0, module_levels=module_filters)

    logger.bind(name="exo.routing.router").debug("routing debug message")
    logger.bind(name="exo.master.placement").debug("master debug message")
    logger.bind(name="exo.master.placement").info("master info message")
    exo_logging.logger_cleanup()

    captured = capfd.readouterr().err
    # routing debug message should be shown because of per-module filter
    assert "routing debug message" in captured
    # master debug message should NOT be shown (base level is INFO)
    assert "master debug message" not in captured
    # master info message should be shown
    assert "master info message" in captured


def test_file_logging_defaults_to_debug(tmp_path: Path):
    log_file = tmp_path / "test_debug.log"
    # Even at default verbosity 0 (no -v), file log defaults to DEBUG
    exo_logging.logger_setup(log_file, verbosity=0)

    logger.debug("detailed debug diagnostic")
    logger.info("general operational info")
    exo_logging.logger_cleanup()

    content = log_file.read_text(encoding="utf-8")
    assert "detailed debug diagnostic" in content
    assert "general operational info" in content


def test_cli_and_env_log_modules(monkeypatch: pytest.MonkeyPatch):
    # Test CLI flag --log-modules
    monkeypatch.setattr(
        sys,
        "argv",
        ["exo", "--log-modules", "exo.routing=DEBUG,exo.master=WARNING"],
    )
    args = Args.parse()
    assert args.log_modules == "exo.routing=DEBUG,exo.master=WARNING"
    parsed = exo_logging.parse_module_log_filters(args.log_modules)
    assert parsed == {"exo.routing": "DEBUG", "exo.master": "WARNING"}

    # Test environment variable fallback
    monkeypatch.setattr(sys, "argv", ["exo"])
    monkeypatch.setenv("EXO_LOG_MODULES", "exo.download=TRACE")
    args_env = Args.parse()
    assert args_env.log_modules == "exo.download=TRACE"
    assert exo_logging.parse_module_log_filters(args_env.log_modules) == {
        "exo.download": "TRACE"
    }
