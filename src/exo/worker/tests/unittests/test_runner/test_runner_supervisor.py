from typing import cast

import anyio
import pytest

import exo.worker.runner.supervisor as supervisor_module
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import ChunkGenerated, Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerRunning
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import Receiver, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _DeadProcess:
    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = -6

    def is_alive(self) -> bool:
        return False


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
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
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
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


class _StuckProcess:
    """A runner process that is alive but will never make progress again."""

    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def stop(self) -> None:
        self.exitcode = -15


@pytest.mark.anyio
async def test_a_runner_whose_ring_aborted_is_stopped_and_reported_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-b"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-b"),
        node_id=NodeId("node-b"),
    )
    stuck = _StuckProcess()
    proc = cast(AsyncProcess, cast(object, stuck))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.shutdown = lambda: None
    for _ in range(10):
        handler.diagnostics.record_line(
            "[ring] Receiving from socket 51 failed with errno 54"
        )
    handler.diagnostics.record_line("[ring] Too many send/recv errors. Aborting...")

    with anyio.fail_after(2):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        status = await event_receiver.receive()

    assert not stuck.is_alive()
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)


@pytest.mark.anyio
async def test_a_healthy_runner_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-c"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-c"),
        node_id=NodeId("node-c"),
    )
    healthy = _StuckProcess()
    proc = cast(AsyncProcess, cast(object, healthy))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    handler.diagnostics.record_line("some unrelated warning")

    with anyio.move_on_after(0.1):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]

    assert healthy.is_alive()


class _AliveProcess:
    """A runner process that stays alive until it is stopped."""

    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def stop(self) -> None:
        self.exitcode = -15


async def generating_supervisor(
    rank: int,
) -> tuple[RunnerSupervisor, _AliveProcess, Receiver[Event]]:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-s"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-s"),
        node_id=NodeId("node-s"),
    )
    if rank != 0:
        bound_instance = BoundInstance(
            instance=bound_instance.instance,
            bound_runner_id=RunnerId("other_runner"),
            bound_node_id=NodeId("other_node"),
        )
    process = _AliveProcess()
    proc = cast(AsyncProcess, cast(object, process))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.shutdown = lambda: None
    supervisor.status = RunnerRunning()
    task = TextGeneration(
        task_id=TaskId("task-s"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-s"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    supervisor.in_progress[task.task_id] = task
    return supervisor, process, event_receiver


@pytest.mark.anyio
async def test_a_first_rank_runner_silent_mid_generation_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, events = await generating_supervisor(rank=0)

    with anyio.fail_after(2):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        chunk = await events.receive()
        status = await events.receive()

    assert not process.is_alive()
    assert isinstance(chunk, ChunkGenerated)
    assert isinstance(chunk.chunk, ErrorChunk)
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)


@pytest.mark.anyio
async def test_a_runner_that_keeps_reporting_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, _ = await generating_supervisor(rank=0)

    async def keep_reporting() -> None:
        while True:
            supervisor._last_heard = anyio.current_time()  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0.01)

    async with anyio.create_task_group() as tg:
        tg.start_soon(keep_reporting)
        with anyio.move_on_after(0.3):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        tg.cancel_scope.cancel()

    assert process.is_alive()


@pytest.mark.anyio
async def test_other_ranks_are_not_judged_by_their_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, _ = await generating_supervisor(rank=1)

    with anyio.move_on_after(0.3):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]

    assert process.is_alive()
