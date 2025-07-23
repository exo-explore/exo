import asyncio
import threading
from asyncio.queues import Queue
from logging import Logger

from master.api import start_fastapi_server
from shared.db.sqlite.config import EventLogConfig
from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogManager
from shared.types.common import NodeId
from shared.types.events import ChunkGenerated
from shared.types.events.chunks import TokenChunk
from shared.types.events.commands import Command, CommandId


## TODO: Hook this up properly
async def fake_tokens_task(events_log: AsyncSQLiteEventStorage, command_id: CommandId):
    model_id = "testmodelabc"
    
    for i in range(10):
        await asyncio.sleep(0.1)
        
        # Create the event with proper types and consistent IDs
        chunk_event = ChunkGenerated(
            command_id=command_id,
            chunk=TokenChunk(
                command_id=command_id,  # Use the same task_id
                idx=i,
                model=model_id,   # Use the same model_id
                text=f'text{i}',
                token_id=i
            )
        )
        
        # ChunkGenerated needs to be cast to the expected BaseEvent type
        await events_log.append_events(
            [chunk_event],
            origin=NodeId()
        )

    await asyncio.sleep(0.1)

    # Create the event with proper types and consistent IDs
    chunk_event = ChunkGenerated(
        command_id=command_id,
        chunk=TokenChunk(
            command_id=command_id,  # Use the same task_id
            idx=11,
            model=model_id,   # Use the same model_id
            text=f'text{11}',
            token_id=11,
            finish_reason='stop'
        )
    )    

    # ChunkGenerated needs to be cast to the expected BaseEvent type
    await events_log.append_events(
        [chunk_event],
        origin=NodeId()
    )



async def main():
    logger = Logger(name='master_logger')

    event_log_manager = EventLogManager(EventLogConfig(), logger=logger)
    await event_log_manager.initialize()
    global_events: AsyncSQLiteEventStorage = event_log_manager.global_events

    command_queue: Queue[Command] = asyncio.Queue()

    api_thread = threading.Thread(
        target=start_fastapi_server,
        args=(
            command_queue,
            global_events,
        ),
        daemon=True
    )
    api_thread.start()
    print('Running FastAPI server in a separate thread. Listening on port 8000.')

    while True:
        # master loop
        if not command_queue.empty():
            command = await command_queue.get()

            print(command)

            await fake_tokens_task(global_events, command_id=command.command_id)

        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())