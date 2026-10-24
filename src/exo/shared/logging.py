import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message

_MAX_LOG_ARCHIVES = 5
# A node that runs for weeks would otherwise keep one ever-growing log file: at default
# verbosity each request logs a few lines, ~200 MB a day under steady load
_MAX_LOG_BYTES = 50 * 1024 * 1024


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _once_then_never() -> Iterator[bool]:
    yield True
    while True:
        yield False


def _rotate_at_start_and_by_size(
    max_bytes: int,
) -> "Callable[[Message, TextIO], bool]":
    """Start a new log file when exo starts, and whenever the current one passes max_bytes."""
    at_start = _once_then_never()

    def should_rotate(message: "Message", file: TextIO) -> bool:
        return next(at_start) or file.tell() + len(message) > max_bytes

    return should_rotate


class InterceptLogger(HypercornLogger):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.error_logger
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


type ConsoleLogLevel = Literal["WARNING", "INFO", "DEBUG"]

VALID_LOG_LEVELS: set[str] = {
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def parse_module_log_filters(filter_str: str | None) -> dict[str, str]:
    """Parse comma-separated module log filters, e.g.
    'exo.routing=DEBUG,exo.master=INFO,exo.download=WARNING'.
    Returns a dict mapping module name prefix to log level name in uppercase.
    """
    if not filter_str:
        return {}
    filters: dict[str, str] = {}
    for part in filter_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        module, level = part.split("=", 1)
        module = module.strip()
        level = level.strip().upper()
        if module and level in VALID_LOG_LEVELS:
            filters[module] = level
    return filters


def create_module_log_filter(
    module_levels: Mapping[str, str] | None,
    base_level: str,
) -> Callable[[dict[str, Any]], bool]:
    """Create a filter function that checks if a log record meets the per-module or base level threshold.
    Hierarchical prefix matching is applied: 'exo.routing' matches 'exo.routing.router'.
    The most specific (longest) matching prefix wins.
    """
    normalized_filters: dict[str, int] = {}
    if module_levels:
        for mod, lvl in module_levels.items():
            lvl_name = lvl.upper()
            if lvl_name in VALID_LOG_LEVELS:
                normalized_filters[mod] = logger.level(lvl_name).no

    base_level_no = logger.level(base_level.upper()).no

    def log_filter(record: dict[str, Any]) -> bool:
        extra = record.get("extra", {})
        record_name = (
            extra.get("name") or extra.get("module") or record.get("name") or ""
        )
        level_obj = record.get("level")
        record_level_no = getattr(level_obj, "no", base_level_no)

        # Find longest matching prefix
        matched_level_no: int | None = None
        longest_prefix_len = -1
        for mod, min_lvl_no in normalized_filters.items():
            if (record_name == mod or record_name.startswith(mod + ".")) and len(
                mod
            ) > longest_prefix_len:
                longest_prefix_len = len(mod)
                matched_level_no = min_lvl_no

        threshold = matched_level_no if matched_level_no is not None else base_level_no
        return record_level_no >= threshold

    return log_filter


def console_log_level(verbosity: int) -> ConsoleLogLevel:
    """The lowest level shown on the console: -q (verbosity -1) shows warnings and errors only,
    the default shows info, and -v or more shows debug."""
    if verbosity < 0:
        return "WARNING"
    if verbosity == 0:
        return "INFO"
    return "DEBUG"


def logger_setup(
    log_file: Path | None,
    verbosity: int = 0,
    module_levels: Mapping[str, str] | None = None,
    file_log_level: str = "DEBUG",
):
    """Set up logging for this process - formatting, file handles, verbosity and output"""
    if module_levels is None:
        module_levels = parse_module_log_filters(os.getenv("EXO_LOG_MODULES"))

    # Suppress noisy stdlib / third-party loggers
    for noisy_module in (
        "exo_rs",
        "networking",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "anyio",
        "zenoh",
        "hpack",
    ):
        logging.getLogger(noisy_module).setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    console_base = console_log_level(verbosity)
    console_filter = create_module_log_filter(module_levels, console_base)
    file_filter = create_module_log_filter(module_levels, file_log_level)

    console_format = (
        "[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>"
        if verbosity <= 0
        else "[ {time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>"
    )

    logger.add(
        sys.__stderr__,  # type: ignore
        format=console_format,
        level="DEBUG",
        filter=console_filter,
        colorize=True,
        enqueue=True,
    )

    if log_file:
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="DEBUG",
            filter=file_filter,
            colorize=False,
            enqueue=True,
            rotation=_rotate_at_start_and_by_size(_MAX_LOG_BYTES),
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


""" --- TODO: Capture MLX Log output:
import contextlib
import sys
from loguru import logger

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

logger.remove()
logger.add(sys.__stdout__)

stream = StreamToLogger()
with contextlib.redirect_stdout(stream):
    print("Standard output is sent to added handlers.")
"""
