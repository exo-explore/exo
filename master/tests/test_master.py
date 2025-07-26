import asyncio
import tempfile
from logging import Logger
from pathlib import Path
from typing import List

import pytest

from master.main import Master
from shared.db.sqlite.config import EventLogConfig
from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogManager
from shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from shared.types.common import NodeId
from shared.types.events import TaskCreated
from shared.types.events._events import TopologyNodeCreated
from shared.types.events.commands import ChatCompletionCommand, Command, CommandId
from shared.types.tasks import ChatCompletionTask, TaskStatus, TaskType


def _create_forwarder_dummy_binary() -> Path:
    path = Path(tempfile.mktemp()) / "forwarder.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\necho dummy forwarder && sleep 1000000\n")
        path.chmod(0o755)
    return path

@pytest.mark.asyncio
async def test_master():
    logger = Logger(name='test_master_logger')
    event_log_manager = EventLogManager(EventLogConfig(), logger=logger)
    await event_log_manager.initialize()
    global_events: AsyncSQLiteEventStorage = event_log_manager.global_events
    await global_events.delete_all_events()

    command_buffer: List[Command] = []

    forwarder_binary_path = _create_forwarder_dummy_binary()

    node_id = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    master = Master(node_id, command_buffer=command_buffer, global_events=global_events, worker_events=event_log_manager.worker_events, forwarder_binary_path=forwarder_binary_path, logger=logger)
    asyncio.create_task(master.run())

    command_buffer.append(
        ChatCompletionCommand(
            command_id=CommandId(),
            request_params=ChatCompletionTaskParams(
                model="llama-3.2-1b",
                messages=[ChatCompletionMessage(role="user", content="Hello, how are you?")]
            )
        )
    )
    while len(await global_events.get_events_since(0)) == 0:
        await asyncio.sleep(0.001)

    events = await global_events.get_events_since(0)
    assert len(events) == 2
    assert events[0].idx_in_log == 1
    assert isinstance(events[0].event, TopologyNodeCreated)
    assert isinstance(events[1].event, TaskCreated)
    assert events[1].event == TaskCreated(
        task_id=events[1].event.task_id,
        task=ChatCompletionTask(
            task_id=events[1].event.task_id,
            task_type=TaskType.CHAT_COMPLETION,
            instance_id=events[1].event.task.instance_id,
            task_status=TaskStatus.PENDING,
            task_params=ChatCompletionTaskParams(
                model="llama-3.2-1b",
                messages=[ChatCompletionMessage(role="user", content="Hello, how are you?")]
            )
        )
    )
    assert len(command_buffer) == 0
