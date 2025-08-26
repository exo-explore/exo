from __future__ import annotations

import sys
from logging import Logger
from pathlib import Path

import loguru
from loguru import logger

from exo.shared.constants import EXO_TEST_LOG


def is_user_facing(record: loguru.Record) -> bool:
    return ("user_facing" in record["extra"]) and record["extra"]["user_facing"]


def logger_setup(log_file: Path, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""
    logger.remove()
    if verbosity == 0:
        _ = logger.add(  # type: ignore
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mmA} | <level>{level: <8}</level>] <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=True,
            filter=is_user_facing,
        )
    elif verbosity == 1:
        _ = logger.add(  # type: ignore
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mmA} | <level>{level: <8}</level>] <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=True,
        )
    else:
        _ = logger.add(  # type: ignore
            sys.__stderr__,  # type: ignore
            format="[ {time:HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )
    _ = logger.add(
        log_file,
        format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
        level="DEBUG",
        enqueue=True,
    )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


def logger_test_install(py_logger: Logger):
    """Installs a default python logger into the Loguru environment by capturing all its handlers - intended to be used for pytest compatibility, not within the main codebase"""
    logger_setup(EXO_TEST_LOG, 3)
    for handler in py_logger.handlers:
        logger.add(handler)
