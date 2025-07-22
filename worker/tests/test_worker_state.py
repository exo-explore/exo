## Tests for worker state differentials
## When the worker state changes, this should be reflected by a worker intention.

import asyncio
from typing import Callable
from uuid import uuid4

import pytest

from shared.types.common import NodeId
from shared.types.state import State
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import Instance
from worker.main import Worker


@pytest.mark.asyncio
async def test_worker_runs_and_stops(worker: Worker):
    await worker.start()
    await asyncio.sleep(0.01)

    assert worker._is_running # type: ignore

    await worker.stop()
    await asyncio.sleep(0.01)

    assert not worker._is_running # type: ignore

@pytest.mark.asyncio
async def test_worker_instance_added(worker: Worker, instance: Callable[[NodeId], Instance]):
    await worker.start()
    await asyncio.sleep(0.01)

    worker.state.instances = {InstanceId(uuid4()): instance(worker.node_id)}
    
    print(worker.state.instances)

def test_plan_noop(worker: Worker):
    s = State(
        node_status={
                NodeId(uuid4()): NodeStatus.Idle
            }
        )

    next_op = worker.plan(s)

    assert next_op is None
