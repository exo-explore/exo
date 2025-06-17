import logging
import logging.handlers
from collections.abc import Sequence
from queue import Queue

from rich.logging import RichHandler


def configure_logger(
    logger_name: str,
    log_level: int = logging.INFO,
    effect_handlers: Sequence[logging.Handler] | None = None,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    # If the named logger already has handlers, we assume it has been configured.
    if logger.hasHandlers():
        return logger

    console_handler = RichHandler(
        rich_tracebacks=True,
    )
    console_handler.setLevel(log_level)

    logger.addHandler(console_handler)
    if effect_handlers is None:
        effect_handlers = []
    for effect_handler in effect_handlers:
        logger.addHandler(effect_handler)

    return logger


def attach_to_queue(logger: logging.Logger, queue: Queue[logging.LogRecord]) -> None:
    logger.addHandler(logging.handlers.QueueHandler(queue))


def create_queue_listener(
    log_queue: Queue[logging.LogRecord],
    effect_handlers: list[logging.Handler],
) -> logging.handlers.QueueListener:
    listener = logging.handlers.QueueListener(
        log_queue, *effect_handlers, respect_handler_level=True
    )
    return listener
