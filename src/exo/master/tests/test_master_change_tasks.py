"""A request started under one master can be finished or cancelled under the next.

The new master learns about the request's task from the state it takes over, not from having
created it itself.
"""

from collections.abc import Callable

import anyio
import pytest

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    Command,
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    TaskCancelled,
    TaskFinished,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceCreated,
    LocalForwarderEvent,
    StateSnapshot,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import channel
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


async def _until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(10):
        while not condition():
            await anyio.sleep(0.01)


@pytest.mark.parametrize("ending", ["finished", "cancelled"])
async def test_a_request_from_before_a_master_change_can_still_end(ending: str):
    node_id = get_node_zid()
    session_id = SessionId(master_node_id=node_id, election_clock=1)
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    snapshot_sender, _snapshot_receiver = channel[StateSnapshot]()
    master = Master(
        node_id,
        session_id,
        global_event_sender=global_sender,
        local_event_receiver=local_receiver,
        command_receiver=command_receiver,
        download_command_sender=download_sender,
        snapshot_sender=snapshot_sender,
    )
    sent: list[Event] = []

    def broadcast() -> list[Event]:
        sent.extend(message.event for message in global_receiver.collect())
        return sent

    async def send(command: Command) -> None:
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )

    command_id = CommandId()
    runner_id = RunnerId()
    instance = get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=ModelId("test/model"),
        node_to_runner={NodeId("node"): runner_id},
        runner_to_shard={
            runner_id: get_pipeline_shard_metadata(ModelId("test/model"), device_rank=0)
        },
    )
    task = TextGenerationTask(
        instance_id=instance.instance_id,
        command_id=command_id,
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test/model"),
            input=[InputMessage(role="user", content=InputMessageContent("Hi"))],
        ),
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        # The task the previous master created, as the new master's state has it
        for index, event in enumerate(
            [
                InstanceCreated(instance=instance),
                TaskCreated(task_id=task.task_id, task=task),
            ]
        ):
            await local_sender.send(
                LocalForwarderEvent(
                    origin=SystemId("previous master's log"),
                    origin_idx=index,
                    session=session_id,
                    event=event,
                )
            )
        await _until(lambda: task.task_id in master.state.tasks)

        if ending == "finished":
            await send(TaskFinished(finished_command_id=command_id))
            await _until(lambda: task.task_id not in master.state.tasks)
            assert any(
                isinstance(e, TaskDeleted) and e.task_id == task.task_id
                for e in broadcast()
            )
        else:
            await send(TaskCancelled(cancelled_command_id=command_id))
            await _until(
                lambda: master.state.tasks[task.task_id].task_status
                == TaskStatus.Cancelled
            )
            assert any(
                isinstance(e, TaskStatusUpdated)
                and e.task_id == task.task_id
                and e.task_status == TaskStatus.Cancelled
                for e in broadcast()
            )

        await master.shutdown()
