import sys
from pathlib import Path

from loguru import logger


def logger_setup(log_file: Path, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""
    logger.remove()
    if verbosity == 0:
        _ = logger.add(  # type: ignore
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
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
            enqueue=True,
        )
    _ = logger.add(
        log_file,
        format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
        level="INFO",
        enqueue=True,
    )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()
