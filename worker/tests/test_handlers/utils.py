## Tests for worker state handlers



from shared.types.events import (
    Event,
)
from shared.types.worker.ops import (
    RunnerOp,
)
from worker.main import Worker


async def read_events_op(worker: Worker, op: RunnerOp) -> list[Event]:
    events: list[Event] = []
    async for event in worker.execute_op(op):
        events.append(event)
    return events