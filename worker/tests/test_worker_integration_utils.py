

import asyncio
from typing import Tuple

from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.types.events import ChunkGenerated, TaskStateUpdated
from shared.types.events.chunks import TokenChunk
from shared.types.tasks import TaskStatus


async def read_streaming_response(global_events: AsyncSQLiteEventStorage) -> Tuple[bool, bool, str]:
    # Read off all events - these should be our GenerationChunk events
    seen_task_started, seen_task_finished = 0, 0
    response_string = ''
    finish_reason: str | None = None

    idx = 0
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

            if isinstance(event, ChunkGenerated):
                assert isinstance(event.chunk, TokenChunk)
                response_string += event.chunk.text
                if event.chunk.finish_reason:
                    finish_reason = event.chunk.finish_reason

    await asyncio.sleep(0.2)

    print(f'event log: {await global_events.get_events_since(0)}')

    return seen_task_started == 1, seen_task_finished == 1, response_string