"""A master elected in a running cluster carries on from the state its node already has."""

import anyio

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
)
from exo.shared.types.common import CommandId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TestEvent,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.channels import channel

CARRIED_OVER_IDX = 40


def carried_over_state() -> tuple[State, RunnerId, TaskId]:
    runner_id, task_id = RunnerId(), TaskId()
    task = TextGenerationTask(
        task_id=task_id,
        command_id=CommandId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    state = State(
        last_event_applied_idx=CARRIED_OVER_IDX,
        runners={runner_id: RunnerReady()},
        tasks={task_id: task},
    )
    return state, runner_id, task_id


async def test_a_new_master_carries_on_from_its_nodes_state() -> None:
    state, runner_id, task_id = carried_over_state()
    node_id = get_node_zid()
    session = SessionId(master_node_id=node_id, election_clock=7)
    global_send, global_recv = channel[GlobalForwarderEvent]()
    command_send, command_recv = channel[ForwarderCommand]()
    local_send, local_recv = channel[LocalForwarderEvent]()
    snapshot_send, snapshot_recv = channel[StateSnapshot]()
    event_send, _ = channel[Event]()
    download_send, _ = channel[ForwarderDownloadCommand]()
    master = Master(
        node_id,
        session,
        event_sender=event_send,
        global_event_sender=global_send,
        snapshot_sender=snapshot_send,
        local_event_receiver=local_recv,
        command_receiver=command_recv,
        download_command_sender=download_send,
        initial_state=state,
    )

    # The runners carry on; the requests ended with the old master
    assert runner_id in master.state.runners
    assert task_id not in master.state.tasks
    assert master.command_task_mapping == {}

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        # New events are numbered on from the state it carried over
        await local_send.send(
            LocalForwarderEvent(
                origin_idx=0, origin=SystemId(), session=session, event=TestEvent()
            )
        )
        with anyio.fail_after(5):
            indexed = await global_recv.receive()
        assert indexed.origin_idx == CARRIED_OVER_IDX + 1

        # A node joining its session catches up from a snapshot of that state
        requester = SystemId()
        await command_send.send(
            ForwarderCommand(
                origin=requester, command=RequestEventLog(since_idx=0, snapshot=True)
            )
        )
        with anyio.fail_after(5):
            snapshot = await snapshot_recv.receive()
        assert snapshot.requester == requester
        assert snapshot.state.last_event_applied_idx == CARRIED_OVER_IDX + 1
        assert runner_id in snapshot.state.runners
        assert task_id not in snapshot.state.tasks

        await master.shutdown()


def test_a_master_of_a_new_cluster_starts_empty() -> None:
    node_id = get_node_zid()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        event_sender=channel[Event]()[0],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        snapshot_sender=channel[StateSnapshot]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        command_receiver=channel[ForwarderCommand]()[1],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )

    assert master.state.last_event_applied_idx == State().last_event_applied_idx
    assert not master.state.instances and not master.state.tasks
    assert master.command_task_mapping == {}
