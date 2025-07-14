from typing import Literal
from collections.abc import Set

from shared.logging.common import LogEntry, LogEntryType


class WorkerUninitialized(LogEntry[Literal["master_uninitialized"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["master_uninitialized"] = "master_uninitialized"
    message: str = "No master state found, creating new one."


WorkerLogEntries = WorkerUninitialized