import logging
import logging.handlers
from collections.abc import Sequence, Set
from queue import Queue
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter
from rich.logging import RichHandler

from master.logging import MasterLogEntries
from shared.logging.common import LogEntryType
from worker.logging import WorkerLogEntries

LogEntries = Annotated[
    MasterLogEntries | WorkerLogEntries, Field(discriminator="entry_type")
]
LogParser: TypeAdapter[LogEntries] = TypeAdapter(LogEntries)


class FilterLogByType(logging.Filter):
    def __init__(self, log_types: Set[LogEntryType]):
        super().__init__()
        self.log_types = log_types

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        LogParser.validate_json(message)
        return True


class LogEntry(BaseModel):
    event_type: Set[LogEntryType]


class LogFilterByType(logging.Filter):
    def __init__(self, log_types: Set[LogEntryType]):
        super().__init__()
        self.log_types = log_types

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        LogEntry.model_validate_json(message)
        return True


def configure_logger(
    logger_name: str,
    log_level: int = logging.INFO,
    effect_handlers: Sequence[logging.Handler] | None = None,
) -> logging.Logger:
    existing_logger = logging.Logger.manager.loggerDict.get(logger_name)
    if existing_logger is not None:
        raise RuntimeError(f"Logger with name '{logger_name}' already exists.")

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False
    logging.raiseExceptions = True

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


def attach_to_queue(
    logger: logging.Logger,
    filter_with: Sequence[logging.Filter],
    queue: Queue[logging.LogRecord],
) -> None:
    handler = logging.handlers.QueueHandler(queue)
    for log_filter in filter_with:
        handler.addFilter(log_filter)
    logger.addHandler(handler)


def create_queue_listener(
    log_queue: Queue[logging.LogRecord],
    effect_handlers: Sequence[logging.Handler],
) -> logging.handlers.QueueListener:
    listener = logging.handlers.QueueListener(
        log_queue, *effect_handlers, respect_handler_level=True
    )
    return listener


def log(
    logger: logging.Logger, log_entry: LogEntries, log_level: int = logging.INFO
) -> None:
    logger.log(log_level, log_entry.model_dump_json())
