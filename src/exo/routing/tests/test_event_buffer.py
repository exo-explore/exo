import pytest

from exo.shared.types.events import Event, TestEvent
from exo.utils.event_buffer import OrderedBuffer


def make_indexed_event(idx: int) -> tuple[int, Event]:
    """Factory function to create a unique ForwarderEvent for a given index."""
    return (idx, TestEvent())


@pytest.fixture
def buffer() -> OrderedBuffer[Event]:
    """Provides a clean instance of OrderedBuffer[Event] for each test."""
    return OrderedBuffer[Event]()


@pytest.mark.asyncio
async def test_initial_state(buffer: OrderedBuffer[Event]):
    """Tests that a new buffer is empty and starts at index 1."""
    assert buffer.next_idx_to_release == 0
    assert not buffer.store
    assert buffer.drain() == []


@pytest.mark.asyncio
async def test_ingest_and_drain_sequential_events(buffer: OrderedBuffer[Event]):
    """Tests ingesting and draining a simple, ordered sequence of events."""
    events = [make_indexed_event(0), make_indexed_event(1), make_indexed_event(2)]
    [buffer.ingest(*ev) for ev in events]

    drained_events = buffer.drain_indexed()
    assert drained_events == events
    assert buffer.next_idx_to_release == 3
    assert not buffer.store


@pytest.mark.asyncio
async def test_ingest_out_of_order_events(buffer: OrderedBuffer[Event]):
    """Tests that out-of-order events are buffered and drained in the correct sequence."""
    event1 = make_indexed_event(0)
    event2 = make_indexed_event(1)
    event3 = make_indexed_event(2)

    buffer.ingest(*event3)
    buffer.ingest(*event1)
    buffer.ingest(*event2)

    drained_events = buffer.drain_indexed()
    assert drained_events == [event1, event2, event3]
    assert buffer.next_idx_to_release == 3


@pytest.mark.asyncio
async def test_drain_with_gap_in_sequence(buffer: OrderedBuffer[Event]):
    """Tests that draining stops when there is a gap in the event indices."""
    event1 = make_indexed_event(0)
    event3 = make_indexed_event(2)

    buffer.ingest(*event1)
    buffer.ingest(*event3)

    drained_events = buffer.drain_indexed()
    assert drained_events == [event1]
    assert buffer.next_idx_to_release == 1

    assert buffer.drain() == []
    assert 2 in buffer.store


@pytest.mark.asyncio
async def test_fill_gap_and_drain_remaining(buffer: OrderedBuffer[Event]):
    """Tests that once a gap is filled, the rest of the sequence is drained."""
    event0 = make_indexed_event(0)
    event2 = make_indexed_event(2)
    buffer.ingest(*event0)
    buffer.ingest(*event2)

    buffer.drain()
    assert buffer.next_idx_to_release == 1

    event1 = make_indexed_event(1)
    buffer.ingest(*event1)

    drained_events = buffer.drain_indexed()
    assert [e[0] for e in drained_events] == [1, 2]
    assert buffer.next_idx_to_release == 3


@pytest.mark.asyncio
async def test_ingest_drops_duplicate_indices(buffer: OrderedBuffer[Event]):
    """Tests that if multiple events for the same index are ingested, the first one wins."""
    event2_first = make_indexed_event(1)
    event2_second = (1, TestEvent())

    buffer.ingest(*make_indexed_event(0))
    buffer.ingest(*event2_first)

    with pytest.raises(AssertionError):
        buffer.ingest(*event2_second)  # This duplicate should be ignored

    drained = buffer.drain_indexed()
    assert len(drained) == 2

    assert drained[1][1].event_id == event2_first[1].event_id
    assert drained[1][1].event_id != event2_second[1].event_id


@pytest.mark.asyncio
async def test_ingest_drops_stale_events(buffer: OrderedBuffer[Event]):
    """Tests that events with an index lower than next_idx_to_release are dropped."""
    buffer.ingest(*make_indexed_event(0))
    buffer.ingest(*make_indexed_event(1))
    buffer.drain()

    assert buffer.next_idx_to_release == 2

    stale_event1 = make_indexed_event(0)
    stale_event2 = make_indexed_event(1)
    buffer.ingest(*stale_event1)
    buffer.ingest(*stale_event2)

    assert not buffer.store
    assert buffer.drain() == []


@pytest.mark.asyncio
async def test_drain_and_ingest_with_new_sequence(buffer: OrderedBuffer[Event]):
    """Tests reusing the buffer after it has been fully drained."""
    buffer.ingest(*make_indexed_event(0))
    buffer.ingest(*make_indexed_event(1))
    buffer.drain()

    assert buffer.next_idx_to_release == 2
    assert not buffer.store

    buffer.ingest(*make_indexed_event(4))
    buffer.ingest(*make_indexed_event(2))

    drained = buffer.drain_indexed()
    assert [e[0] for e in drained] == [2]
    assert buffer.next_idx_to_release == 3
    assert 4 in buffer.store
