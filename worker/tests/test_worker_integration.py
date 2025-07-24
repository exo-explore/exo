import asyncio
from logging import Logger
from typing import Callable, Final
from uuid import UUID

from shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from shared.types.common import NodeId
from shared.types.events import InstanceCreated
from shared.types.models import ModelId
from shared.types.state import State
from shared.types.tasks import TaskId
from shared.types.worker.common import InstanceId, RunnerId
from shared.types.worker.instances import Instance
from worker.main import Worker

MASTER_NODE_ID = NodeId(uuid=UUID("ffffffff-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
NODE_A: Final[NodeId] = NodeId(uuid=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
NODE_B: Final[NodeId] = NodeId(uuid=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"))

# Define constant IDs for deterministic test cases
RUNNER_1_ID: Final[RunnerId] = RunnerId()
INSTANCE_1_ID: Final[InstanceId] = InstanceId()
RUNNER_2_ID: Final[RunnerId] = RunnerId()
INSTANCE_2_ID: Final[InstanceId] = InstanceId()
MODEL_A_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
MODEL_B_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
TASK_1_ID: Final[TaskId] = TaskId()

async def test_runner_spin_up(instance: Callable[[NodeId], Instance]):
    # TODO.
    return
    node_id = NodeId()
    logger = Logger('worker_test_logger')
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()

    global_events = event_log_manager.global_events    

    worker = Worker(node_id, State(), logger=logger, worker_events=global_events)
    await worker.start()

    instance_value = instance(node_id)

    await global_events.append_events(
        [
            InstanceCreated(
                instance_id=instance_value.instance_id,
                instance_params=instance_value.instance_params,
                instance_type=instance_value.instance_type
            )
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.1)

    assert worker.assigned_runners