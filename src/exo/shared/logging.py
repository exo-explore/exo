import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

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


def logger_setup(log_file: Path | None, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""

    logging.getLogger("exo_rs").setLevel(logging.INFO)
    logging.getLogger("networking").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    if verbosity == 0:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )
    if log_file:
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="DEBUG" if verbosity > 0 else "INFO",
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
