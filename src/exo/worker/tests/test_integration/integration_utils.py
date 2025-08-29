import asyncio
from typing import Callable, Optional, Tuple, TypeVar

from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.types.events import ChunkGenerated, TaskStateUpdated
from exo.shared.types.events.chunks import TokenChunk
from exo.shared.types.tasks import TaskId, TaskStatus


async def read_streaming_response(
    global_events: AsyncSQLiteEventStorage, filter_task: Optional[TaskId] = None
) -> Tuple[bool, bool, str, int]:
    # Read off all events - these should be our GenerationChunk events
    seen_task_started, seen_task_finished = 0, 0
    response_string = ""
    finish_reason: str | None = None
    token_count = 0

    if not filter_task:
        idx = await global_events.get_last_idx()
    else:
        found = False
        idx = 0
        while not found:
            events = await global_events.get_events_since(idx)

            for event in events:
                if (
                    isinstance(event.event, TaskStateUpdated)
                    and event.event.task_status == TaskStatus.RUNNING
                    and event.event.task_id == filter_task
                ):
                    found = True
                    idx = event.idx_in_log - 1
                    break

    print(f"START IDX {idx}")

    while not finish_reason:
        events = await global_events.get_events_since(idx)
        if len(events) == 0:
            await asyncio.sleep(0.01)
            continue
        idx = events[-1].idx_in_log

        for wrapped_event in events:
            event = wrapped_event.event
            if isinstance(event, TaskStateUpdated):
                if event.task_status == TaskStatus.RUNNING:
                    seen_task_started += 1
                if event.task_status == TaskStatus.COMPLETE:
                    seen_task_finished += 1

            if isinstance(event, ChunkGenerated) and isinstance(
                event.chunk, TokenChunk
            ):
                response_string += event.chunk.text
                token_count += 1
                if event.chunk.finish_reason:
                    finish_reason = event.chunk.finish_reason

    await asyncio.sleep(0.2)

    print(f"event log: {await global_events.get_events_since(0)}")

    return seen_task_started == 1, seen_task_finished == 1, response_string, token_count


T = TypeVar("T")


async def until_event_with_timeout(
    global_events: AsyncSQLiteEventStorage,
    event_type: type[T],
    multiplicity: int = 1,
    condition: Callable[[T], bool] = lambda x: True,
    timeout: float = 30.0,
) -> None:
    idx = await global_events.get_last_idx()
    times_seen = 0
    start_time = asyncio.get_event_loop().time()
    
    while True:
        events = await global_events.get_events_since(idx)
        if events:
            for wrapped_event in events:
                if isinstance(wrapped_event.event, event_type) and condition(
                    wrapped_event.event
                ):
                    times_seen += 1
                    if times_seen >= multiplicity:
                        return
            idx = events[-1].idx_in_log

        current_time = asyncio.get_event_loop().time()
        if current_time - start_time > timeout:
            raise asyncio.TimeoutError(
                f"Timeout waiting for {multiplicity} events of type {event_type.__name__} "
                f"(found {times_seen} in {timeout}s)"
            )
        
        await asyncio.sleep(0.01)
