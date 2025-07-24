import asyncio
import contextlib
import json
from asyncio import Queue, Task
from collections.abc import Sequence
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel

from shared.types.events import Event, EventParser, NodeId
from shared.types.events.components import EventFromEventLog

from .types import StoredEvent


class AsyncSQLiteEventStorage:
    """High-performance SQLite event storage with async batching.
    
    Features:
    - Non-blocking writes via adaptive async batching with debouncing
    - Automatic sequence numbering using SQLite rowid
    - Type-safe event serialization/deserialization
    - Efficient indexing for common query patterns
    
    Batching behavior:
    - Low load: Minimal latency via short debounce windows
    - High load: Efficient batching up to batch_size limit
    - Max age constraint prevents indefinite delays
    """
    
    def __init__(
        self, 
        db_path: str | Path, 
        batch_size: int,
        batch_timeout_ms: int,
        debounce_ms: int,
        max_age_ms: int,
        logger: Logger | None = None
    ):
        self._db_path = Path(db_path)
        self._batch_size = batch_size
        self._batch_timeout_s = batch_timeout_ms / 1000.0
        self._debounce_s = debounce_ms / 1000.0
        self._max_age_s = max_age_ms / 1000.0
        self._logger = logger or getLogger(__name__)
        
        self._write_queue: Queue[tuple[Event, NodeId]] = Queue()
        self._batch_writer_task: Task[None] | None = None
        self._engine = None
        self._closed = False
    
    async def start(self) -> None:
        """Initialize the storage and start the batch writer."""
        if self._batch_writer_task is not None:
            raise RuntimeError("Storage already started")
        
        # Create database and tables
        await self._initialize_database()
        
        # Start batch writer
        self._batch_writer_task = asyncio.create_task(self._batch_writer())
        self._logger.info(f"Started SQLite event storage: {self._db_path}")
    
    async def append_events(
        self, 
        events: Sequence[Event], 
        origin: NodeId
    ) -> None:
        """Append events to the log (fire-and-forget). The writes are batched and committed 
        in the background so readers don't have a guarantee of seeing events immediately."""
        if self._closed:
            raise RuntimeError("Storage is closed")
        
        for event in events:
            await self._write_queue.put((event, origin))
    
    async def get_events_since(
        self, 
        last_idx: int
    ) -> Sequence[EventFromEventLog[Event]]:
        """Retrieve events after a specific index."""
        if self._closed:
            raise RuntimeError("Storage is closed")
        
        assert self._engine is not None
        
        async with AsyncSession(self._engine) as session:
            # Use raw SQL to get rowid along with the stored event data
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events WHERE rowid > :last_idx ORDER BY rowid"),
                {"last_idx": last_idx}
            )
            rows = result.fetchall()
        
        events: list[EventFromEventLog[Event]] = []
        for row in rows:
            rowid: int = cast(int, row[0])
            origin: str = cast(str, row[1])
            # Parse JSON string to dict
            raw_event_data = row[2]  # type: ignore[reportAny] - SQLAlchemy result is Any
            if isinstance(raw_event_data, str):
                event_data: dict[str, Any] = cast(dict[str, Any], json.loads(raw_event_data))
            else:
                event_data = cast(dict[str, Any], raw_event_data)
            events.append(EventFromEventLog(
                event=EventParser.validate_python(event_data),
                origin=NodeId(uuid=UUID(origin)),
                idx_in_log=rowid  # rowid becomes idx_in_log
            ))
        
        return events

    async def get_last_idx(self) -> int:
        if self._closed:
            raise RuntimeError("Storaged is closed")
    
        assert self._engine is not None

        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events ORDER BY rowid DESC LIMIT 1"),
                {}
            )
            rows = result.fetchall()

        if len(rows) == 0:
            return 0
        if len(rows) == 1:
            row = rows[0]
            return cast(int, row[0])
        else:
            raise AssertionError("There should have been at most 1 row returned from this SQL query.")
    
    async def close(self) -> None:
        """Close the storage connection and cleanup resources."""
        if self._closed:
            return
        
        self._closed = True
        
        # Stop batch writer
        if self._batch_writer_task is not None:
            self._batch_writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_writer_task
        
        # Close database
        if self._engine is not None:
            await self._engine.dispose()
        
        self._logger.info("Closed SQLite event storage")
    
    async def _initialize_database(self) -> None:
        """Initialize database connection and create tables."""
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,
            connect_args={
                "check_same_thread": False,
            }
        )
        
        # Create tables using SQLModel
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            
            # Enable WAL mode and other optimizations
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA cache_size=10000"))
    
    async def _batch_writer(self) -> None:
        """Background task that drains the queue and commits batches.
        
        Uses adaptive batching with debouncing:
        - Blocks waiting for first item (no CPU waste when idle)
        - Opens debounce window to collect more items
        - Respects max age to prevent stale batches
        - Resets debounce timer with each new item
        """
        loop = asyncio.get_event_loop()
        
        while not self._closed:
            batch: list[tuple[Event, NodeId]] = []
            
            try:
                # Block waiting for first item
                event, origin = await self._write_queue.get()
                batch.append((event, origin))
                first_ts = loop.time()  # monotonic seconds
                
                # Open debounce window
                while True:
                    # How much longer can we wait?
                    age_left = self._max_age_s - (loop.time() - first_ts)
                    if age_left <= 0:
                        break  # max age reached → flush
                    
                    # Shrink the wait to honour both debounce and max-age
                    try:
                        event, origin = await asyncio.wait_for(
                            self._write_queue.get(),
                            timeout=min(self._debounce_s, age_left)
                        )
                        batch.append((event, origin))
                        
                        if len(batch) >= self._batch_size:
                            break  # size cap reached → flush
                        # else: loop again, resetting debounce timer
                    except asyncio.TimeoutError:
                        break  # debounce window closed → flush
                
            except asyncio.CancelledError:
                # Drain any remaining items before exiting
                if batch:
                    await self._commit_batch(batch)
                raise
            
            if batch:
                await self._commit_batch(batch)
    
    async def _commit_batch(self, batch: list[tuple[Event, NodeId]]) -> None:
        """Commit a batch of events to SQLite."""
        assert self._engine is not None
        
        try:
            async with AsyncSession(self._engine) as session:
                for event, origin in batch:
                    stored_event = StoredEvent(
                        origin=str(origin.uuid),
                        event_type=event.event_type,
                        event_id=str(event.event_id),
                        event_data=event.model_dump(mode='json')  # Serialize UUIDs and other objects to JSON-compatible strings
                    )
                    session.add(stored_event)
                
                await session.commit()
            
            self._logger.debug(f"Committed batch of {len(batch)} events")
        
        except Exception as e:
            self._logger.error(f"Failed to commit batch: {e}")
            raise
