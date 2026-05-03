import multiprocessing as mp
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TransientEvent,
)
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.channels import channel, mp_channel
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _DeadProcess:
    exitcode = -6

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, _timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.mark.asyncio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    transient_sender, transient_receiver = channel[TransientEvent]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _transient_event_sender=transient_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await transient_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    transient_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
    with anyio.move_on_after(0.1):
        await transient_receiver.aclose()


@pytest.mark.asyncio
async def test_forward_events_routes_generated_chunks_to_transient() -> None:
    event_sender, event_receiver = channel[Event]()
    transient_sender, transient_receiver = channel[TransientEvent]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _transient_event_sender=transient_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    chunk = ChunkGenerated(
        command_id=command_id,
        chunk=ErrorChunk(
            model=bound_instance.bound_shard.model_card.model_id,
            error_message="test chunk",
        ),
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(chunk)
        assert await transient_receiver.receive() == chunk
        tg.cancel_scope.cancel()

    assert event_receiver.collect() == []

    event_sender.close()
    transient_sender.close()
    ev_send.close()
