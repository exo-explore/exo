import asyncio
from logging import Logger


from worker.main import get_node_id
from shared.types.common import NodeId
from shared.db.sqlite.event_log_manager import EventLogManager, EventLogConfig

async def main():
    node_id: NodeId = get_node_id()
    logger: Logger = Logger('worker_log')
    
    event_log_manager: EventLogManager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()

    events = await event_log_manager.global_events.get_events_since(0)

    for wrapped_event in events:
        event = wrapped_event.event
        event_type = type(event).__name__.replace('_', ' ').title()
        attributes = ', '.join(f"{key}={value!r}" for key, value in vars(event).items())
        print(f"{event_type}: {attributes}")

if __name__ == "__main__":
    asyncio.run(main())