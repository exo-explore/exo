"""A request's images travel with it, and nodes keep them only while that request needs them."""

import hashlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import anyio

from exo.shared.types.chunks import ErrorChunk, InputImageChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InputChunkReceived,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    Base64ImageHash,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, Instance
from exo.shared.types.worker.runners import RunnerReady
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    TASK_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

IMAGE = "aGVsbG8gd29ybGQ="
TASK = TextGeneration(
    task_id=TASK_1_ID,
    instance_id=INSTANCE_1_ID,
    task_status=TaskStatus.Pending,
    command_id=COMMAND_1_ID,
    task_params=TextGenerationTaskParams(
        model=MODEL_A_ID,
        input=[InputMessage(role="user", content=InputMessageContent("What's this?"))],
        image_hashes={
            0: Base64ImageHash(hashlib.sha256(IMAGE.encode("ascii")).hexdigest())
        },
    ),
)
CHUNK = InputChunkReceived(
    command_id=COMMAND_1_ID,
    chunk=InputImageChunk(
        model=MODEL_A_ID,
        command_id=COMMAND_1_ID,
        data=IMAGE,
        chunk_index=0,
        total_chunks=1,
        image_index=0,
    ),
)


def instance_on(node: NodeId) -> Instance:
    return get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={node: RUNNER_1_ID},
        runner_to_shard={
            RUNNER_1_ID: get_pipeline_shard_metadata(
                MODEL_A_ID, device_rank=0, world_size=1
            )
        },
    )


async def wait_until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not condition():
            await anyio.sleep(0.01)


@asynccontextmanager
async def worker_on_node_a(
    state: State, *loops: Callable[[Worker], Callable[[], object]]
) -> AsyncIterator[tuple[Worker, Sender[IndexedEvent], Receiver[Event]]]:
    event_send, event_recv = channel[Event]()
    index_send, index_recv = channel[IndexedEvent]()
    command_send, _commands = channel[ForwarderCommand]()
    download_send, _downloads = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )
    worker.state = state
    async with anyio.create_task_group() as tg:
        for loop in loops:
            tg.start_soon(loop(worker))  # pyright: ignore[reportArgumentType]
        yield worker, index_send, event_recv
        tg.cancel_scope.cancel()


def applying(worker: Worker) -> Callable[[], object]:
    return worker._event_applier  # pyright: ignore[reportPrivateUsage]


def has_images(worker: Worker) -> bool:
    return COMMAND_1_ID in worker.input_images


async def test_a_node_that_does_not_run_the_request_drops_its_images() -> None:
    state = State(instances={INSTANCE_1_ID: instance_on(NODE_B)})
    async with worker_on_node_a(state, applying) as (worker, events, _):
        await events.send(IndexedEvent(idx=0, event=CHUNK))
        await wait_until(lambda: has_images(worker))

        await events.send(
            IndexedEvent(idx=1, event=TaskCreated(task_id=TASK_1_ID, task=TASK))
        )
        await wait_until(lambda: not has_images(worker))

    assert worker.input_chunk_buffer == {}


async def test_a_node_that_runs_the_request_keeps_its_images_until_the_task_is_gone() -> (
    None
):
    state = State(instances={INSTANCE_1_ID: instance_on(NODE_A)})
    async with worker_on_node_a(state, applying) as (worker, events, _):
        await events.send(IndexedEvent(idx=0, event=CHUNK))
        await events.send(
            IndexedEvent(idx=1, event=TaskCreated(task_id=TASK_1_ID, task=TASK))
        )
        await wait_until(lambda: worker.state.last_event_applied_idx == 1)
        assert has_images(worker)

        await events.send(IndexedEvent(idx=2, event=TaskDeleted(task_id=TASK_1_ID)))
        await wait_until(lambda: not has_images(worker))


async def test_a_request_whose_images_never_arrived_fails_instead_of_waiting() -> None:
    # e.g. this node caught up from a state snapshot taken after the images were sent
    instance = instance_on(NODE_A)
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerReady()},
        tasks={TASK_1_ID: TASK},
        downloads={NODE_A: []},
    )

    def planning(worker: Worker) -> Callable[[], object]:
        worker.runners[RUNNER_1_ID] = FakeRunnerSupervisor(  # pyright: ignore[reportArgumentType]
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
            ),
            status=RunnerReady(),
        )
        return worker.plan_step

    async with worker_on_node_a(state, planning) as (_, _events, sent):
        # The task stays pending in this test, so the planner keeps offering it
        await anyio.sleep(0.5)

    reported = sent.collect()
    errors = [e for e in reported if isinstance(e, ChunkGenerated)]
    failures = [e for e in reported if isinstance(e, TaskStatusUpdated)]
    assert len(errors) == 1
    assert isinstance(errors[0].chunk, ErrorChunk)
    assert errors[0].command_id == COMMAND_1_ID
    assert [f.task_status for f in failures] == [TaskStatus.Failed]
