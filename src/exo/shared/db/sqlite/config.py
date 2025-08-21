from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from exo.shared.constants import EXO_GLOBAL_EVENT_DB, EXO_WORKER_EVENT_DB


class EventLogType(str, Enum):
    """Types of event logs in the system"""
    WORKER_EVENTS = "worker_events"
    GLOBAL_EVENTS = "global_events"


class EventLogConfig(BaseModel):
    """Configuration for the event log system"""
    
    # Batch processing settings
    batch_size: int = 100
    batch_timeout_ms: int = 100
    debounce_ms: int = 10
    max_age_ms: int = 100
    
    def get_db_path(self, log_type: EventLogType) -> Path:
        """Get the full path for a specific event log type"""
        if log_type == EventLogType.WORKER_EVENTS:
            return EXO_WORKER_EVENT_DB
        elif log_type == EventLogType.GLOBAL_EVENTS:
            return EXO_GLOBAL_EVENT_DB
        else:
            raise ValueError(f"Unknown log type: {log_type}")