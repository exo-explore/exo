import logging
import sys
from pathlib import Path
from typing import Protocol

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger


class _TellableFile(Protocol):
    def tell(self) -> int: ...


_MAX_LOG_ARCHIVES = 5
_ROTATION_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _rotation_policy(max_bytes: int):
    """Rotate once on process start (fresh log per run), then rotate again
    whenever the file crosses ``max_bytes``.

    A previous version used a generator that returned True exactly once
    and False forever after. loguru calls the rotation callable on every
    message, so that rotated on startup as intended but then silently
    disabled every later rotation for the rest of the process's life,
    letting the log file grow unbounded even though retention and
    compression were configured as if ongoing rotation was happening.
    """
    started = False

    def _should_rotate(message: str, file: _TellableFile) -> bool:
        nonlocal started
        if not started:
            started = True
            return True
        try:
            return file.tell() + len(message) > max_bytes
        except OSError:
            return False

    return _should_rotate


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
            rotation=_rotation_policy(_ROTATION_MAX_BYTES),
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
