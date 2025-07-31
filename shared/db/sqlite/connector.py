import asyncio
import contextlib
import json
import random
from asyncio import Queue, Task
from collections.abc import Sequence
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine

from shared.types.events import Event, EventParser, NodeId
from shared.types.events._events import Heartbeat
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
        last_idx: int,
        ignore_no_op_events: bool = False
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
            event = EventParser.validate_python(event_data)
            if ignore_no_op_events and event.__no_apply__:
                continue
            events.append(EventFromEventLog(
                event=event,
                origin=NodeId(origin),
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
    
    async def delete_all_events(self) -> None:
        """Delete all events from the database."""
        assert self._engine is not None
        async with AsyncSession(self._engine) as session:
            await session.execute(text("DELETE FROM events"))
            await session.commit()
    
    async def _initialize_database(self) -> None:
        """Initialize database connection and create tables."""
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 30.0,  # Connection timeout in seconds
            },
            pool_pre_ping=True,  # Test connections before using them
            pool_size=5,
            max_overflow=10
        )
        
        # Create tables with proper race condition handling
        async with self._engine.begin() as conn:
            # First check if the table exists using SQLite's master table
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            )
            table_exists = result.fetchone() is not None
            
            if not table_exists:
                try:
                    # Use CREATE TABLE IF NOT EXISTS as a more atomic operation
                    # This avoids race conditions between check and create
                    await conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS events (
                            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                            origin TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            event_id TEXT NOT NULL,
                            event_data TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    
                    # Create indexes if they don't exist
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_origin ON events(origin)"))
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)"))
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id)"))
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)"))
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_origin_created ON events(origin, created_at)"))
                    
                    self._logger.info("Events table and indexes created successfully")
                except OperationalError as e:
                    # Even with IF NOT EXISTS, log any unexpected errors
                    self._logger.error(f"Error creating table: {e}")
                    # Re-check if table exists now
                    result = await conn.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
                    )
                    if result.fetchone() is None:
                        raise RuntimeError(f"Failed to create events table: {e}") from e
                    else:
                        self._logger.info("Events table exists (likely created by another process)")
            else:
                self._logger.debug("Events table already exists")
            
            # Enable WAL mode and other optimizations with retry logic
            await self._execute_pragma_with_retry(conn, [
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL",
                "PRAGMA cache_size=10000",
                "PRAGMA busy_timeout=30000"  # 30 seconds busy timeout
            ])
    
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
                        origin=origin,
                        event_type=event.event_type,
                        event_id=str(event.event_id),
                        event_data=event.model_dump(mode='json')  # Serialize UUIDs and other objects to JSON-compatible strings
                    )
                    session.add(stored_event)
                
                await session.commit()
            if len([ev for ev in batch if not isinstance(ev[0], Heartbeat)]) > 0:
                self._logger.debug(f"Committed batch of {len(batch)} events")
        
        except OperationalError as e:
            if "database is locked" in str(e):
                self._logger.warning(f"Database locked during batch commit, will retry: {e}")
                # Retry with exponential backoff
                await self._commit_batch_with_retry(batch)
            else:
                self._logger.error(f"Failed to commit batch: {e}")
                raise
        except Exception as e:
            self._logger.error(f"Failed to commit batch: {e}")
            raise
    
    async def _execute_pragma_with_retry(self, conn: AsyncConnection, pragmas: list[str], max_retries: int = 5) -> None:
        """Execute PRAGMA statements with retry logic for database lock errors."""
        for pragma in pragmas:
            retry_count = 0
            base_delay: float = 0.1  # 100ms
            
            while retry_count < max_retries:
                try:
                    await conn.execute(text(pragma))
                    break
                except OperationalError as e:
                    if "database is locked" in str(e) and retry_count < max_retries - 1:
                        delay = cast(float, base_delay * (2 ** retry_count) + random.uniform(0, 0.1))
                        self._logger.warning(f"Database locked on '{pragma}', retry {retry_count + 1}/{max_retries} after {delay:.2f}s")
                        await asyncio.sleep(delay)
                        retry_count += 1
                    else:
                        self._logger.error(f"Failed to execute '{pragma}' after {retry_count + 1} attempts: {e}")
                        raise
    
    async def _commit_batch_with_retry(self, batch: list[tuple[Event, NodeId]], max_retries: int = 5) -> None:
        """Commit a batch with retry logic for database lock errors."""
        retry_count = 0
        base_delay: float = 0.1  # 100ms
        
        while retry_count < max_retries:
            try:
                assert self._engine is not None
                
                async with AsyncSession(self._engine) as session:
                    for event, origin in batch:
                        stored_event = StoredEvent(
                            origin=origin,
                            event_type=event.event_type,
                            event_id=str(event.event_id),
                            event_data=event.model_dump(mode='json')
                        )
                        session.add(stored_event)
                    
                    await session.commit()
                
                if len([ev for ev in batch if not isinstance(ev[0], Heartbeat)]) > 0:
                    self._logger.debug(f"Committed batch of {len(batch)} events after {retry_count} retries")
                return
                
            except OperationalError as e:
                if "database is locked" in str(e) and retry_count < max_retries - 1:
                    delay = cast(float, base_delay * (2 ** retry_count) + random.uniform(0, 0.1))
                    self._logger.warning(f"Database locked on batch commit, retry {retry_count + 1}/{max_retries} after {delay:.2f}s")
                    await asyncio.sleep(delay)
                    retry_count += 1
                else:
                    self._logger.error(f"Failed to commit batch after {retry_count + 1} attempts: {e}")
                    raise
