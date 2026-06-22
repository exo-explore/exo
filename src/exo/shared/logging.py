import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import zstandard
from exo_rs import VerbosityFilter
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

_MAX_LOG_ARCHIVES = 5


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


def _loguru_log_level(verbosity: VerbosityFilter):
    match verbosity:
        case VerbosityFilter.Off:
            raise ValueError(
                "VerbosityFilter.Off does not translate to a loguru log-level"
            )
        case VerbosityFilter.Error:
            return "ERROR"
        case VerbosityFilter.Warn:
            return "WARNING"
        case VerbosityFilter.Info:
            return "INFO"
        case VerbosityFilter.Debug:
            return "DEBUG"
        case VerbosityFilter.Trace:
            return "TRACE"


def logger_setup(
    log_file: Path | None, verbosity: VerbosityFilter = VerbosityFilter.Info
):
    """Set up logging for this process - formatting, file handles, verbosity and output"""

    logging.getLogger("exo_rs").setLevel(logging.INFO)
    logging.getLogger("networking").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    # if Off then no logging - return early
    if verbosity == VerbosityFilter.Off:
        return

    # info (or less verbose than info) gets a different formatter
    level = _loguru_log_level(verbosity)
    if verbosity <= VerbosityFilter.Info:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
            level=level,
            colorize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level=level,
            colorize=True,
            enqueue=True,
        )
    if log_file:
        rotate_once = _once_then_never()
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level=level,
            colorize=False,
            enqueue=True,
            rotation=lambda _, __: next(rotate_once),
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()
