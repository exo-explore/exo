import sys
from pathlib import Path

from loguru import logger


def logger_setup(log_file: Path | None, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""
    logger.remove()
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
            format="[ {time:HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )
    if log_file:
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="INFO",
            colorize=False,
            enqueue=True,
            rotation="1 week",
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()
