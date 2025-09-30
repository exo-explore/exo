from pathlib import Path

from pydantic import BaseModel

from exo.shared.constants import EXO_GLOBAL_EVENT_DB


class EventLogConfig(BaseModel):
    """Configuration for the event log system"""

    # Batch processing settings
    batch_size: int = 100
    batch_timeout_ms: int = 100
    debounce_ms: int = 10
    max_age_ms: int = 100

    def get_db_path(self) -> Path:
        """Get the full path for a specific event log type"""
        return EXO_GLOBAL_EVENT_DB
