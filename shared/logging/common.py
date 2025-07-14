from enum import Enum
from typing import Generic, TypeVar
from pydantic import BaseModel

from collections.abc import Set

LogEntryTypeT = TypeVar("LogEntryTypeT", bound=str)


class LogEntryType(str, Enum):
    telemetry = "telemetry"
    metrics = "metrics"
    cluster = "cluster"


class LogEntry(BaseModel, Generic[LogEntryTypeT]):
    entry_destination: Set[LogEntryType]
    entry_type: LogEntryTypeT
