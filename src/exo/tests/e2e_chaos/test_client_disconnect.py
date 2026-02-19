"""E2E Chaos Test: Client disconnect.

Scenarios:
1. Task cancellation after client disconnect -- a TextGeneration command is
   sent, then immediately cancelled (simulating browser tab close).
   Verify the master correctly transitions the task to Cancelled status.
2. Multiple rapid cancellations -- several chat commands are sent and
   cancelled in quick succession; no tasks should remain in a stuck state.
"""

import anyio
import pytest

from exo.master.main import Master
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    TaskCancelled,
    TextGeneration,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
)
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import channel

from .conftest import (
    TEST_MODEL_CARD,
    TEST_MODEL_ID,
    EventCollector,
    make_gathered_info_event,
    make_node_id,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_task_cancelled_after_client_disconnect() -> None:
    """Simulate a browser tab close by sending a TextGeneration command
    followed immediately by a TaskCancelled command.  Verify the task
    transitions to Cancelled status.
    """
    master_nid = make_node_id("master-cancel")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    _collector = EventCollector(ge_receiver.clone())

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register node
        sender_id = NodeId(f"{master_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(master_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        # Place instance
        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        # Send a chat command
        chat_cmd_id = CommandId()
        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=TextGeneration(
                    command_id=chat_cmd_id,
                    task_params=TextGenerationTaskParams(
                        model=TEST_MODEL_ID,
                        input=[InputMessage(role="user", content="Hello world")],
                    ),
                ),
            )
        )

        # Wait for the task to be created
        with anyio.fail_after(3):
            while len(master.state.tasks) == 0:
                await anyio.sleep(0.01)

        # Immediately cancel -- simulating browser tab close
        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=TaskCancelled(
                    command_id=CommandId(),
                    cancelled_command_id=chat_cmd_id,
                ),
            )
        )

        # Wait for the task status to be updated to Cancelled
        with anyio.fail_after(3):
            while True:
                tasks_cancelled = [
                    t
                    for t in master.state.tasks.values()
                    if t.task_status == TaskStatus.Cancelled
                ]
                if tasks_cancelled:
                    break
                await anyio.sleep(0.01)

        assert len(tasks_cancelled) == 1

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_rapid_cancel_does_not_leave_stuck_tasks() -> None:
    """Send multiple chat commands and cancel them all rapidly.
    Verify no tasks remain in Pending or Running state.
    """
    master_nid = make_node_id("master-rapid-cancel")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, _ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register node and place instance
        sender_id = NodeId(f"{master_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(master_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        # Send 5 chat commands and immediately cancel each
        chat_cmd_ids: list[CommandId] = []
        for i in range(5):
            cmd_id = CommandId()
            chat_cmd_ids.append(cmd_id)
            await cmd_sender.send(
                ForwarderCommand(
                    origin=master_nid,
                    command=TextGeneration(
                        command_id=cmd_id,
                        task_params=TextGenerationTaskParams(
                            model=TEST_MODEL_ID,
                            input=[InputMessage(role="user", content=f"Message {i}")],
                        ),
                    ),
                )
            )

        # Wait for all tasks to be created
        with anyio.fail_after(3):
            while len(master.state.tasks) < 5:
                await anyio.sleep(0.01)

        # Cancel all of them
        for cmd_id in chat_cmd_ids:
            await cmd_sender.send(
                ForwarderCommand(
                    origin=master_nid,
                    command=TaskCancelled(
                        command_id=CommandId(),
                        cancelled_command_id=cmd_id,
                    ),
                )
            )

        # Wait for all cancellations to be processed
        with anyio.fail_after(3):
            while True:
                cancelled_count = sum(
                    1
                    for t in master.state.tasks.values()
                    if t.task_status == TaskStatus.Cancelled
                )
                if cancelled_count == 5:
                    break
                await anyio.sleep(0.01)

        # No tasks should be Pending or Running
        stuck = [
            t
            for t in master.state.tasks.values()
            if t.task_status in (TaskStatus.Pending, TaskStatus.Running)
        ]
        assert len(stuck) == 0

        await master.shutdown()
