"""A runner that never acknowledges a task holds up neither the other runners on its node nor its
own shutdown. The worker used to wait for the acknowledgement of every task but a generation
one, so a runner stuck before picking up its task stopped all planning on its node."""

from typing import cast

import anyio
import pytest
from anyio.abc import TaskStatus

import exo.worker.plan as plan_mod
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import Event, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ConnectToGroup,
    Shutdown,
    Task,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, RunnerReady
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, channel, mp_channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    INSTANCE_2_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

RUNNER_3_ID = RunnerId("runner-3")

# An instance across this node and another, whose runner here never acknowledges anything
STUCK = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 2),
        RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, 2),
    },
)
# A model on this node alone, ready to serve
SERVING = get_mlx_ring_instance(
    instance_id=INSTANCE_2_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_3_ID},
    runner_to_shard={RUNNER_3_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 1)},
)
DOWNLOADED = [
    DownloadCompleted(
        node_id=NODE_A,
        shard_metadata=STUCK.shard_assignments.runner_to_shard[RUNNER_1_ID],
        total=Memory(),
    )
]
REQUEST = TextGeneration(
    task_id=TASK_1_ID,
    instance_id=INSTANCE_2_ID,
    command_id=COMMAND_1_ID,
    task_params=TextGenerationTaskParams(
        model=MODEL_A_ID,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    ),
)


class _SilentProcess:
    """A runner process that is alive but never answers."""

    exitcode = 0

    def __init__(self):
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()

    def is_alive(self) -> bool:
        return True

    async def run(self, *, task_status: TaskStatus[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        pass


async def _silent_supervisor(
    instance_id: InstanceId, runner_id: RunnerId
) -> tuple[RunnerSupervisor, MpReceiver[Task]]:
    instance = STUCK if instance_id == INSTANCE_1_ID else SERVING
    bound = BoundInstance(
        instance=instance, bound_runner_id=runner_id, bound_node_id=NODE_A
    )
    event_sender, _ = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    proc = cast(AsyncProcess, cast(object, _SilentProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound.bound_shard,
        bound_instance=bound,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    return supervisor, task_receiver


async def _sent(receiver: MpReceiver[Task]) -> list[Task]:
    """What has been sent to a runner (multiprocessing queues deliver with a small delay)"""
    await anyio.sleep(0.3)
    return receiver.collect()


def _worker() -> Worker:
    _, index_recv = channel[IndexedEvent]()
    event_send, _ = channel[Event]()
    command_send, _ = channel[ForwarderCommand]()
    download_send, _ = channel[ForwarderDownloadCommand]()
    return Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )


@pytest.mark.anyio
async def test_a_runner_that_never_acknowledges_does_not_hold_up_its_node() -> None:
    stuck, sent_to_stuck = await _silent_supervisor(INSTANCE_1_ID, RUNNER_1_ID)
    serving, sent_to_serving = await _silent_supervisor(INSTANCE_2_ID, RUNNER_3_ID)
    serving.status = RunnerReady()

    worker = _worker()
    worker.state = State(
        instances={INSTANCE_1_ID: STUCK, INSTANCE_2_ID: SERVING},
        runners={
            RUNNER_1_ID: RunnerIdle(),
            RUNNER_2_ID: RunnerIdle(),
            RUNNER_3_ID: RunnerReady(),
        },
        downloads={NODE_A: DOWNLOADED},
        tasks={TASK_1_ID: REQUEST},
    )
    worker.runners = {RUNNER_1_ID: stuck, RUNNER_3_ID: serving}

    async with anyio.create_task_group() as tg:
        tg.start_soon(stuck.run)
        tg.start_soon(serving.run)
        tg.start_soon(worker.plan_step)

        # The other model on the node still gets its request...
        with anyio.fail_after(2):
            while TASK_1_ID not in serving.in_progress:
                await anyio.sleep(0.05)
        assert [t.task_id for t in await _sent(sent_to_serving)] == [TASK_1_ID]
        # ...and the stuck runner was told to connect once, not again and again
        assert [type(t) for t in await _sent(sent_to_stuck)] == [ConnectToGroup]

        # Deleting the stuck runner's instance shuts it down
        worker.state = worker.state.model_copy(
            update={"instances": {INSTANCE_2_ID: SERVING}}
        )
        with anyio.fail_after(2):
            while RUNNER_1_ID in worker.runners:
                await anyio.sleep(0.05)
        assert [type(t) for t in await _sent(sent_to_stuck)] == [Shutdown]

        tg.cancel_scope.cancel()


def test_a_runner_yet_to_acknowledge_is_not_sent_its_task_again() -> None:
    bound = BoundInstance(
        instance=STUCK, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )

    def plan(awaiting_acknowledgement: bool) -> Task | None:
        runner = FakeRunnerSupervisor(
            bound_instance=bound,
            status=RunnerIdle(),
            awaiting_acknowledgement=awaiting_acknowledgement,
        )
        return plan_mod.plan(
            node_id=NODE_A,
            runners={RUNNER_1_ID: runner},  # type: ignore[arg-type]
            global_download_status={NODE_A: DOWNLOADED},
            instances={INSTANCE_1_ID: STUCK},
            all_runners={RUNNER_1_ID: RunnerIdle(), RUNNER_2_ID: RunnerIdle()},
            tasks={},
            instance_backoff=KeyedBackoff(),
            download_backoff=KeyedBackoff(),
        )

    assert isinstance(plan(awaiting_acknowledgement=False), ConnectToGroup)
    assert plan(awaiting_acknowledgement=True) is None
