import multiprocessing as mp
import time
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import ChunkGenerated, Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
)
from exo.utils.channels import channel, mp_channel
from exo.worker.runner.runner_supervisor import (
    STARTUP_STALL_TIMEOUT_SECONDS,
    RunnerSupervisor,
)
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


class _AliveProcess:
    def __init__(self) -> None:
        self.exitcode: int | None = None
        self._alive = True

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, _timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9


@pytest.mark.asyncio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
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
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_startup_stall_reason_detects_runner_loaded_stall() -> None:
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-b"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-b"),
        node_id=NodeId("node-b"),
    )

    process = _AliveProcess()
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, process)),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerLoaded()
    supervisor._last_status_change_time = (  # pyright: ignore[reportPrivateUsage]
        time.time() - STARTUP_STALL_TIMEOUT_SECONDS - 1
    )

    reason = supervisor._startup_stall_reason()  # pyright: ignore[reportPrivateUsage]

    assert reason is not None
    assert "RunnerLoaded" in reason
    process.terminate()


@pytest.mark.asyncio
async def test_startup_stall_reason_detects_runner_loading_progress_stall() -> None:
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-c"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-c"),
        node_id=NodeId("node-c"),
    )

    process = _AliveProcess()
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, process)),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerLoading(layers_loaded=2, total_layers=16)
    supervisor._last_loading_progress_time = (  # pyright: ignore[reportPrivateUsage]
        time.time() - STARTUP_STALL_TIMEOUT_SECONDS - 1
    )

    reason = supervisor._startup_stall_reason()  # pyright: ignore[reportPrivateUsage]

    assert reason is not None
    assert "RunnerLoading" in reason
    assert "without layer progress" in reason
    process.terminate()


@pytest.mark.asyncio
async def test_check_runner_turns_startup_stall_into_runner_failed() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-d"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-d"),
        node_id=NodeId("node-d"),
    )
    process = _AliveProcess()

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, process)),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.shutdown = lambda: None

    failure_message = "Runner startup stalled in RunnerLoaded for 121.0s"
    await supervisor._check_runner(  # pyright: ignore[reportPrivateUsage]
        TimeoutError(failure_message),
        failure_message=failure_message,
    )

    got_status = await event_receiver.receive()

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)
    assert failure_message in (got_status.runner_status.error_message or "")
    assert process.exitcode == -15

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
