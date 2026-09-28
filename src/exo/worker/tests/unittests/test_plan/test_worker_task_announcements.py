"""The worker announces the tasks it creates, not the ones it picks up from the state."""

from dataclasses import dataclass, field
from typing import cast

import anyio

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import Event, IndexedEvent, TaskCreated
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerReady, RunnerStatus
from exo.utils.channels import channel
from exo.worker.main import Worker
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


@dataclass
class RecordingRunner:
    bound_instance: BoundInstance
    status: RunnerStatus
    completed: set[TaskId] = field(default_factory=set)
    in_progress: dict[TaskId, Task] = field(default_factory=dict)
    cancelled: set[TaskId] = field(default_factory=set)
    pending: dict[TaskId, object] = field(default_factory=dict)
    started: list[Task] = field(default_factory=list)

    async def start_task(self, task: Task) -> None:
        self.started.append(task)
        self.in_progress[task.task_id] = task


async def test_worker_does_not_reannounce_a_task_from_the_state() -> None:
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )
    runner = RecordingRunner(
        bound_instance=BoundInstance(
            instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
        ),
        status=RunnerReady(),
    )
    task = TextGeneration(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )

    event_send, event_recv = channel[Event]()
    _index_send, index_recv = channel[IndexedEvent]()
    command_send, _command_recv = channel[ForwarderCommand]()
    download_send, _download_recv = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )
    worker.state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        downloads={NODE_A: []},
        tasks={TASK_1_ID: task},
    )
    worker.runners = {RUNNER_1_ID: cast(RunnerSupervisor, cast(object, runner))}

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker.plan_step)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert [t.task_id for t in runner.started] == [TASK_1_ID]
    announced = [e for e in event_recv.collect() if isinstance(e, TaskCreated)]
    # Re-announcing it could recreate the task after the master cancelled or deleted it
    assert announced == []
