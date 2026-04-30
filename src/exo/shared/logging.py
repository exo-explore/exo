import logging
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

_MAX_LOG_ARCHIVES = 5
_DEFAULT_LOG_ROTATION = "100 MB"
_DEFAULT_LOG_RETENTION = "7 days"

_LOG_CONTEXT: dict[str, object] = {
    "run_id": os.environ.get("EXO_RUN_ID", str(uuid4())),
    "node_id": "unknown",
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
    "role": "startup",
    "session_id": "unknown",
    "git_commit": os.environ.get("EXO_GIT_COMMIT", "unknown"),
}


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _context_format(message_format: str) -> str:
    return (
        "[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "run={extra[run_id]} node={extra[node_id]} host={extra[hostname]} "
        "pid={extra[pid]} role={extra[role]} session={extra[session_id]} "
        "git={extra[git_commit]} | {name}:{function}:{line} ] "
        f"{message_format}"
    )


def logger_set_context(**updates: object) -> None:
    """Update process-wide log context for subsequent log records."""
    _LOG_CONTEXT.update(
        {key: value for key, value in updates.items() if value is not None}
    )
    logger.configure(extra=_LOG_CONTEXT)


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

    logging.getLogger("exo_pyo3_bindings").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()
    logger.configure(extra=_LOG_CONTEXT)

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    console_level = "INFO" if verbosity == 0 else "DEBUG"
    if verbosity == 0:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
            level=console_level,
            colorize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.__stderr__,  # type: ignore
            format=_context_format("<level>{message}</level>"),
            level=console_level,
            colorize=True,
            enqueue=True,
        )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        run_dir = log_file.parent / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = (
            f"{timestamp}-{_LOG_CONTEXT['hostname']}-{_LOG_CONTEXT['pid']}-"
            f"{_LOG_CONTEXT['run_id']}"
        )
        run_text_log = run_dir / f"{run_name}.log"
        run_json_log = run_dir / f"{run_name}.jsonl"
        rotation = os.environ.get("EXO_LOG_ROTATION", _DEFAULT_LOG_ROTATION)
        retention = os.environ.get("EXO_LOG_RETENTION", _DEFAULT_LOG_RETENTION)

        logger.add(
            log_file,
            format=_context_format("{message}"),
            level="DEBUG" if verbosity > 0 else "INFO",
            colorize=False,
            enqueue=True,
            rotation=rotation,
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )
        for destination, serialize in ((run_text_log, False), (run_json_log, True)):
            logger.add(
                destination,
                format=_context_format("{message}"),
                level="DEBUG",
                colorize=False,
                enqueue=True,
                rotation=rotation,
                retention=retention,
                compression=_zstd_compress,
                serialize=serialize,
            )
        logger.info(
            f"Per-run logs enabled text_log={run_text_log} json_log={run_json_log} "
            f"rotation={rotation} retention={retention}"
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
