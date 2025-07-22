import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Generator, cast
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.sqlite import AsyncSQLiteEventStorage, EventLogConfig
from shared.types.common import NodeId
from shared.types.events.chunks import ChunkType, TokenChunk
from shared.types.events.events import (
    ChunkGenerated,
    EventType,
)
from shared.types.tasks.request import RequestId

# Type ignore comment for all protected member access in this test file
# pyright: reportPrivateUsage=false


def _load_json_data(raw_data: str) -> dict[str, Any]:
    """Helper function to load JSON data with proper typing."""
    return cast(dict[str, Any], json.loads(raw_data))


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_node_id() -> NodeId:
    """Create a sample NodeId for testing."""
    return NodeId(uuid=uuid4())


class TestAsyncSQLiteEventStorage:
    """Test suite for AsyncSQLiteEventStorage focused on storage functionality."""

    @pytest.mark.asyncio
    async def test_initialization_creates_tables(self, temp_db_path: Path) -> None:
        """Test that database initialization creates the events table."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Verify table exists by querying directly
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='events'"))
            tables = result.fetchall()
            assert len(tables) == 1
            assert tables[0][0] == "events"
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self, temp_db_path: Path) -> None:
        """Test that starting storage twice raises an error."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        with pytest.raises(RuntimeError, match="Storage already started"):
            await storage.start()
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_direct_database_operations(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test direct database operations without event parsing."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Insert test data directly
        test_data = {
            "event_type": "test_event",
            "test_field": "test_value",
            "number": 42
        }
        
        async with AsyncSession(storage._engine) as session:
            await session.execute(
                text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                {
                    "origin": str(sample_node_id.uuid),
                    "event_type": "test_event",
                    "event_id": str(uuid4()),
                    "event_data": json.dumps(test_data)
                }
            )
            await session.commit()
        
        # Query data back
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events ORDER BY rowid")
            )
            rows = result.fetchall()
        
        assert len(rows) == 1
        assert rows[0][0] == 1  # rowid
        assert rows[0][1] == str(sample_node_id.uuid)  # origin
        raw_json = cast(str, rows[0][2])
        retrieved_data = _load_json_data(raw_json)
        assert retrieved_data == test_data
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_rowid_auto_increment(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test that rowid auto-increments correctly."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Insert multiple records
        test_records = [
            {"event_type": "test_event_1", "data": "first"},
            {"event_type": "test_event_2", "data": "second"},
            {"event_type": "test_event_3", "data": "third"}
        ]
        
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            for record in test_records:
                await session.execute(
                    text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                    {
                        "origin": str(sample_node_id.uuid),
                        "event_type": record["event_type"],
                        "event_id": str(uuid4()),
                        "event_data": json.dumps(record)
                    }
                )
            await session.commit()
        
        # Query back and verify rowid sequence
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, event_data FROM events ORDER BY rowid")
            )
            rows = result.fetchall()
        
        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert row[0] == i + 1  # rowid starts at 1
            raw_json = cast(str, row[1])
            retrieved_data = _load_json_data(raw_json)
            assert retrieved_data == test_records[i]
        
        await storage.close()



    @pytest.mark.asyncio
    async def test_get_last_idx(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test that rowid returns correctly from db."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Insert multiple records
        test_records = [
            {"event_type": "test_event_1", "data": "first"},
            {"event_type": "test_event_2", "data": "second"},
            {"event_type": "test_event_3", "data": "third"}
        ]
        
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            for record in test_records:
                await session.execute(
                    text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                    {
                        "origin": str(sample_node_id.uuid),
                        "event_type": record["event_type"],
                        "event_id": str(uuid4()),
                        "event_data": json.dumps(record)
                    }
                )
            await session.commit()
        
        last_idx = await storage.get_last_idx()
        assert last_idx == 3
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_rowid_with_multiple_origins(self, temp_db_path: Path) -> None:
        """Test rowid sequence across multiple origins."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        origin1 = NodeId(uuid=uuid4())
        origin2 = NodeId(uuid=uuid4())
        
        # Insert interleaved records from different origins
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            # Origin 1 - record 1
            await session.execute(
                text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                {"origin": str(origin1.uuid), "event_type": "event_1", "event_id": str(uuid4()), "event_data": json.dumps({"from": "origin1", "seq": 1})}
            )
            # Origin 2 - record 2
            await session.execute(
                text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                {"origin": str(origin2.uuid), "event_type": "event_2", "event_id": str(uuid4()), "event_data": json.dumps({"from": "origin2", "seq": 2})}
            )
            # Origin 1 - record 3
            await session.execute(
                text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                {"origin": str(origin1.uuid), "event_type": "event_3", "event_id": str(uuid4()), "event_data": json.dumps({"from": "origin1", "seq": 3})}
            )
            await session.commit()
        
        # Verify sequential rowid regardless of origin
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events ORDER BY rowid")
            )
            rows = result.fetchall()
        
        assert len(rows) == 3
        assert rows[0][0] == 1  # First rowid
        assert rows[1][0] == 2  # Second rowid
        assert rows[2][0] == 3  # Third rowid
        
        # Verify data integrity
        raw_json1 = cast(str, rows[0][2])
        raw_json2 = cast(str, rows[1][2])
        raw_json3 = cast(str, rows[2][2])
        data1 = _load_json_data(raw_json1)
        data2 = _load_json_data(raw_json2)
        data3 = _load_json_data(raw_json3)
        
        assert data1["from"] == "origin1" and data1["seq"] == 1
        assert data2["from"] == "origin2" and data2["seq"] == 2
        assert data3["from"] == "origin1" and data3["seq"] == 3
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_query_events_since_index(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test querying events after a specific rowid."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Insert 10 test records
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            for i in range(10):
                await session.execute(
                    text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                    {
                        "origin": str(sample_node_id.uuid),
                        "event_type": f"event_{i}",
                        "event_id": str(uuid4()),
                        "event_data": json.dumps({"index": i})
                    }
                )
            await session.commit()
        
        # Query events after index 5
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, event_data FROM events WHERE rowid > :last_idx ORDER BY rowid"),
                {"last_idx": 5}
            )
            rows = result.fetchall()
        
        assert len(rows) == 5  # Should get records 6-10
        for i, row in enumerate(rows):
            assert row[0] == i + 6  # rowid 6, 7, 8, 9, 10
            raw_json = cast(str, row[1])
            data = _load_json_data(raw_json)
            assert data["index"] == i + 5  # index 5, 6, 7, 8, 9
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_empty_query(self, temp_db_path: Path) -> None:
        """Test querying when no events exist."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events WHERE rowid > :last_idx ORDER BY rowid"),
                {"last_idx": 0}
            )
            rows = result.fetchall()
        
        assert len(rows) == 0
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_operations_after_close_raise_error(self, temp_db_path: Path) -> None:
        """Test that operations after close work properly."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        await storage.close()
        
        # These should not raise errors since we're not using the public API
        assert storage._closed is True
        assert storage._engine is not None  # Engine should still exist but be disposed

    @pytest.mark.asyncio
    async def test_multiple_close_calls_safe(self, temp_db_path: Path) -> None:
        """Test that multiple close calls are safe."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        await storage.close()
        await storage.close()  # Should not raise an error

    @pytest.mark.asyncio
    async def test_json_data_types(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test that various JSON data types are handled correctly."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Test various JSON data types
        test_data = {
            "string": "test string",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3, "four"],
            "object": {"nested": "value", "deep": {"deeper": "nested"}},
            "unicode": "测试 🚀"
        }
        
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            await session.execute(
                text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                {
                    "origin": str(sample_node_id.uuid),
                    "event_type": "complex_event",
                    "event_id": str(uuid4()),
                    "event_data": json.dumps(test_data)
                }
            )
            await session.commit()
        
        # Query back and verify data integrity
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT event_data FROM events WHERE event_type = :event_type"),
                {"event_type": "complex_event"}
            )
            rows = result.fetchall()
        
        assert len(rows) == 1
        raw_json = cast(str, rows[0][0])
        retrieved_data = _load_json_data(raw_json)
        assert retrieved_data == test_data
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, temp_db_path: Path) -> None:
        """Test concurrent inserts maintain rowid ordering."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        async def insert_batch(origin_id: str, batch_id: int, count: int) -> None:
            assert storage._engine is not None
            async with AsyncSession(storage._engine) as session:
                for i in range(count):
                    await session.execute(
                        text("INSERT INTO events (origin, event_type, event_id, event_data) VALUES (:origin, :event_type, :event_id, :event_data)"),
                        {
                            "origin": origin_id,
                            "event_type": f"batch_{batch_id}_event_{i}",
                            "event_id": str(uuid4()),
                            "event_data": json.dumps({"batch": batch_id, "item": i})
                        }
                    )
                await session.commit()
        
        # Run multiple concurrent insert batches
        origin1 = str(uuid4())
        origin2 = str(uuid4())
        origin3 = str(uuid4())
        
        await asyncio.gather(
            insert_batch(origin1, 1, 5),
            insert_batch(origin2, 2, 5),
            insert_batch(origin3, 3, 5)
        )
        
        # Verify all records were inserted and rowid is sequential
        assert storage._engine is not None
        async with AsyncSession(storage._engine) as session:
            result = await session.execute(
                text("SELECT rowid, origin, event_data FROM events ORDER BY rowid")
            )
            rows = result.fetchall()
        
        assert len(rows) == 15  # 3 batches * 5 records each
        
        # Verify rowid sequence is maintained
        for i, row in enumerate(rows):
            assert row[0] == i + 1  # rowid should be sequential
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_chunk_generated_event_serialization(self, temp_db_path: Path, sample_node_id: NodeId) -> None:
        """Test that ChunkGenerated event with nested types can be serialized and deserialized correctly."""
        default_config = EventLogConfig()
        storage = AsyncSQLiteEventStorage(db_path=temp_db_path, batch_size=default_config.batch_size, batch_timeout_ms=default_config.batch_timeout_ms, debounce_ms=default_config.debounce_ms, max_age_ms=default_config.max_age_ms)
        await storage.start()
        
        # Create a ChunkGenerated event with nested TokenChunk
        request_id = RequestId(uuid=uuid4())
        token_chunk = TokenChunk(
            text="Hello, world!",
            token_id=42,
            finish_reason="stop",
            chunk_type=ChunkType.token,
            request_id=request_id,
            idx=0,
            model="test-model"
        )
        
        chunk_generated_event = ChunkGenerated(
            request_id=request_id,
            chunk=token_chunk
        )
        
        # Store the event using the storage API
        await storage.append_events([chunk_generated_event], sample_node_id)
        
        # Wait for batch to be written
        await asyncio.sleep(0.5)
        
        # Retrieve the event
        events = await storage.get_events_since(0)
        
        # Verify we got the event back
        assert len(events) == 1
        retrieved_event_wrapper = events[0]
        assert retrieved_event_wrapper.origin == sample_node_id
        
        # Verify the event was deserialized correctly
        retrieved_event = retrieved_event_wrapper.event
        assert isinstance(retrieved_event, ChunkGenerated)
        assert retrieved_event.event_type == EventType.ChunkGenerated
        assert retrieved_event.request_id == request_id
        
        # Verify the nested chunk was deserialized correctly
        retrieved_chunk = retrieved_event.chunk
        assert isinstance(retrieved_chunk, TokenChunk)
        assert retrieved_chunk.chunk_type == ChunkType.token
        assert retrieved_chunk.request_id == request_id
        assert retrieved_chunk.idx == 0
        assert retrieved_chunk.model == "test-model"
        
        # Verify the chunk data
        assert retrieved_chunk.text == "Hello, world!"
        assert retrieved_chunk.token_id == 42
        assert retrieved_chunk.finish_reason == "stop"
        
        await storage.close()