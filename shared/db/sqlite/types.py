from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from sqlalchemy import DateTime, Index
from sqlmodel import JSON, Column, Field, SQLModel

from shared.types.common import NodeId
from shared.types.events import Event
from shared.types.events.components import EventFromEventLog


class StoredEvent(SQLModel, table=True):
    """SQLite representation of an event in the event log.
    
    The rowid serves as the global sequence number (idx_in_log) for ordering.
    """
    __tablename__ = "events"  # type: ignore[assignment]
    
    # SQLite's rowid as primary key - we alias it but don't actually use it in queries
    rowid: int | None = Field(default=None, primary_key=True, alias="rowid")
    origin: str = Field(index=True)
    event_type: str = Field(index=True)
    event_id: str = Field(index=True)
    event_data: dict[str, Any] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        sa_column=Column(DateTime, index=True)
    )
    
    __table_args__ = (
        Index("idx_events_origin_created", "origin", "created_at"),
    )

class EventStorageProtocol(Protocol):
    """Protocol for event storage implementations."""
    
    async def append_events(
        self, 
        events: Sequence[Event],
        origin: NodeId
    ) -> None:
        """Append events to the log (fire-and-forget).
        
        Events are queued for batched writing and assigned idx_in_log
        when committed to storage.
        """
        ...
    
    async def get_events_since(
        self, 
        last_idx: int
    ) -> Sequence[EventFromEventLog[Event]]:
        """Retrieve events after a specific index.
        
        Returns events in idx_in_log order.
        """
        ...
    
    async def close(self) -> None:
        """Close the storage connection and cleanup resources."""
        ...