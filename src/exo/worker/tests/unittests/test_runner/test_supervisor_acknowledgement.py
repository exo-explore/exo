"""The worker isn't held up waiting for a busy runner to acknowledge a generation task."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import anyio
import pytest
from anyio.abc import TaskStatus

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import Event
from exo.shared.types.tasks import LoadModel, Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

BOUND: BoundInstance = get_bound_mlx_ring_instance(
    instance_id=InstanceId("instance-a"),
    model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
    runner_id=RunnerId("runner-a"),
    node_id=NodeId("node-a"),
)


class _SilentProcess:
    """A runner that never answers, as one busy with a long prompt."""

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


@asynccontextmanager
async def supervisor() -> AsyncIterator[tuple[RunnerSupervisor, MpReceiver[Task]]]:
    event_sender, _ = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    proc = cast(AsyncProcess, cast(object, _SilentProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    _cancel_receivers[id(cancel_sender)] = cancel_receiver
    yield (
        RunnerSupervisor(
            shard_metadata=BOUND.bound_shard,
            bound_instance=BOUND,
            runner_process=proc,
            _runner_stdio_handler=handler,
            initialize_timeout=400,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _cancel_sender=cancel_sender,
        ),
        task_receiver,
    )


_cancel_receivers: dict[int, MpReceiver[TaskId]] = {}


def cancels_sent(sup: RunnerSupervisor) -> list[TaskId]:
    receiver = _cancel_receivers[id(sup._cancel_sender)]  # pyright: ignore[reportPrivateUsage]
    return receiver.collect()


def generation() -> TextGeneration:
    return TextGeneration(
        instance_id=BOUND.instance.instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=BOUND.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


@pytest.mark.anyio
async def test_generation_task_is_sent_without_waiting_for_the_runner() -> None:
    async with supervisor() as (sup, sent):
        task = generation()
        with anyio.fail_after(2):
            await sup.start_task(task)
        assert task.task_id in sup.in_progress
        assert sent.receive().task_id == task.task_id


@pytest.mark.anyio
async def test_other_tasks_still_wait_for_the_runner() -> None:
    async with supervisor() as (sup, _):
        load = LoadModel(instance_id=BOUND.instance.instance_id)
        returned = anyio.Event()

        async def start() -> None:
            await sup.start_task(load)
            returned.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(start)
            await anyio.sleep(0.3)
            assert not returned.is_set()
            sup.pending[load.task_id].set()  # the runner acknowledges it
            with anyio.fail_after(2):
                await returned.wait()


@pytest.mark.anyio
async def test_cancelling_a_task_the_runner_has_not_picked_up_waits_for_it() -> None:
    async with supervisor() as (sup, _):
        task = generation()
        async with anyio.create_task_group() as tg:
            tg.start_soon(sup.run)
            await anyio.sleep(0.1)
            await sup.start_task(task)

            # The runner ignores cancellations for tasks it doesn't know yet
            with anyio.fail_after(2):
                await sup.cancel_task(task.task_id)
            await anyio.sleep(0.2)
            assert cancels_sent(sup) == []

            # Once it acknowledges the task, the cancellation follows
            sup.pending.pop(task.task_id).set()
            await anyio.sleep(0.2)
            assert cancels_sent(sup) == [task.task_id]
            sup.shutdown()


@pytest.mark.anyio
async def test_cancelling_an_acknowledged_task_is_sent_straight_away() -> None:
    async with supervisor() as (sup, _):
        task = generation()
        async with anyio.create_task_group() as tg:
            tg.start_soon(sup.run)
            await anyio.sleep(0.1)
            await sup.start_task(task)
            sup.pending.pop(task.task_id).set()

            await sup.cancel_task(task.task_id)
            assert cancels_sent(sup) == [task.task_id]
            sup.shutdown()
