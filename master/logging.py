from typing import Literal
from collections.abc import Set

from shared.logging.common import LogEntry, LogEntryType


class MasterUninitializedLogEntry(LogEntry[Literal["master_uninitialized"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["master_uninitialized"] = "master_uninitialized"
    message: str = "No master state found, creating new one."


class MasterCommandReceivedLogEntry(LogEntry[Literal["master_command_received"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["master_command_received"] = "master_command_received"
    command_name: str


class MasterInvalidCommandReceivedLogEntry(
    LogEntry[Literal["master_invalid_command_received"]]
):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["master_invalid_command_received"] = (
        "master_invalid_command_received"
    )
    command_name: str


class EventCategoryUnknownLogEntry(LogEntry[Literal["event_category_unknown"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["event_category_unknown"] = "event_category_unknown"
    event_category: str
    message: str = "Event Category Unknown, Skipping Event."


class StateUpdateLoopAlreadyRunningLogEntry(
    LogEntry[Literal["state_update_loop_already_running"]]
):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_loop_already_running"] = (
        "state_update_loop_already_running"
    )
    message: str = "State Update Loop Already Running"


class StateUpdateLoopStartedLogEntry(LogEntry[Literal["state_update_loop_started"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_loop_started"] = "state_update_loop_started"
    message: str = "State Update Loop Started"


class StateUpdateLoopNotRunningLogEntry(
    LogEntry[Literal["state_update_loop_not_running"]]
):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_loop_not_running"] = (
        "state_update_loop_not_running"
    )
    message: str = "State Update Loop Not Running"


class StateUpdateLoopStoppedLogEntry(LogEntry[Literal["state_update_loop_stopped"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_loop_stopped"] = "state_update_loop_stopped"
    message: str = "State Update Loop Stopped"


class StateUpdateErrorLogEntry(LogEntry[Literal["state_update_error"]]):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_error"] = "state_update_error"
    error: Exception


class StateUpdateEffectHandlerErrorLogEntry(
    LogEntry[Literal["state_update_effect_handler_error"]]
):
    entry_destination: Set[LogEntryType] = {LogEntryType.cluster}
    entry_type: Literal["state_update_effect_handler_error"] = (
        "state_update_effect_handler_error"
    )
    error: Exception


MasterLogEntries = (
    MasterUninitializedLogEntry
    | MasterCommandReceivedLogEntry
    | MasterInvalidCommandReceivedLogEntry
    | EventCategoryUnknownLogEntry
    | StateUpdateLoopAlreadyRunningLogEntry
    | StateUpdateLoopStartedLogEntry
    | StateUpdateLoopNotRunningLogEntry
    | StateUpdateLoopStoppedLogEntry
    | StateUpdateErrorLogEntry
    | StateUpdateEffectHandlerErrorLogEntry
)
