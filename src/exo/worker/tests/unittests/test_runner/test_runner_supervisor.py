from typing import cast, final

import anyio
import pytest

import exo.worker.runner.supervisor as supervisor_module
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import generation_error_code
from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    Event,
    RunnerEvent,
    RunnerProgressed,
    RunnerStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerRunning
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import Receiver, Sender, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.diagnostics import RunnerMetalOutOfMemory
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


async def _supervisor_of_dead_runner(event_sender: Sender[Event]) -> RunnerSupervisor:
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[RunnerEvent | RunnerTerminationError]()

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
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.shutdown = lambda: None
    return supervisor


def _streamed(supervisor: RunnerSupervisor, task: TextGeneration) -> Receiver[Chunk]:
    """Give the supervisor a request in progress, and return the stream its output goes to"""
    send, receive = channel[Chunk]()
    supervisor.in_progress[task.task_id] = task
    supervisor._outputs[task.task_id] = send  # pyright: ignore[reportPrivateUsage]
    supervisor._output_tasks[task.command_id] = task.task_id  # pyright: ignore[reportPrivateUsage]
    return receive


def _text_generation(supervisor: RunnerSupervisor, name: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"task-{name}"),
        instance_id=supervisor.bound_instance.instance.instance_id,
        command_id=CommandId(f"cmd-{name}"),
        task_params=TextGenerationTaskParams(
            model=supervisor.bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    supervisor = await _supervisor_of_dead_runner(event_sender)
    task = _text_generation(supervisor, "a")
    output = _streamed(supervisor, task)

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await output.receive()
    # The request's error goes to its stream; the cluster hears only of the runner
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ErrorChunk)
    assert got_chunk.error_message == (
        "The model stopped before the reply finished: boom."
    )
    assert generation_error_code(got_chunk.error_message) == "model_failed"

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@final
class _FinishesTheOtherWhileSent:
    """A request's stream, during whose send the runner's last results arrive.

    While the supervisor reports one request as failed, the event loop runs the coroutine that
    forwards the runner's output, which marks another request complete.
    """

    def __init__(self, supervisor: RunnerSupervisor, other: TaskId) -> None:
        self.sent: list[Chunk] = []
        self.supervisor = supervisor
        self.other = other

    async def send(self, chunk: Chunk) -> None:
        self.sent.append(chunk)
        self.supervisor.in_progress.pop(self.other, None)
        await anyio.sleep(0)

    def send_nowait(self, chunk: Chunk) -> None:
        self.sent.append(chunk)

    def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_check_runner_survives_a_request_finishing_while_it_reports() -> None:
    event_sender, event_receiver = channel[Event]()
    supervisor = await _supervisor_of_dead_runner(event_sender)
    first, second = _text_generation(supervisor, "a"), _text_generation(supervisor, "b")
    output = _FinishesTheOtherWhileSent(supervisor, second.task_id)
    _streamed(supervisor, first)
    second_output = _streamed(supervisor, second)
    supervisor._outputs[first.task_id] = cast(Sender[Chunk], cast(object, output))  # pyright: ignore[reportPrivateUsage]

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    assert [type(chunk) for chunk in output.sent] == [ErrorChunk]
    assert second_output.collect() == []
    assert isinstance(event_receiver.collect()[-1], RunnerStatusUpdated)


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
    _, ev_recv = mp_channel[RunnerEvent | RunnerTerminationError]()
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
    _, ev_recv = mp_channel[RunnerEvent | RunnerTerminationError]()
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
) -> tuple[RunnerSupervisor, _AliveProcess, Receiver[Event], Receiver[Chunk]]:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[RunnerEvent | RunnerTerminationError]()
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
    output = _streamed(supervisor, task)
    return supervisor, process, event_receiver, output


@pytest.mark.anyio
async def test_a_first_rank_runner_silent_mid_generation_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, events, output = await generating_supervisor(rank=0)

    with anyio.fail_after(2):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        chunk = await output.receive()
        status = await events.receive()

    assert not process.is_alive()
    assert isinstance(chunk, ErrorChunk)
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)


@pytest.mark.anyio
async def test_a_runner_that_keeps_reporting_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runner reports every 10 ms; the stall timeout leaves room for a busy CI machine to pause the event loop
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.5)
    supervisor, process, _, _ = await generating_supervisor(rank=0)

    async def keep_reporting() -> None:
        while True:
            supervisor._last_heard = anyio.current_time()  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0.01)

    async with anyio.create_task_group() as tg:
        tg.start_soon(keep_reporting)
        with anyio.move_on_after(1.0):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        tg.cancel_scope.cancel()

    assert process.is_alive()


@pytest.mark.anyio
async def test_a_runner_writing_a_tool_call_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool call streams only once complete; meanwhile the runner says it is still
    generating (RunnerProgressed), which keeps its supervisor from stopping it and never
    reaches the cluster."""
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.5)
    supervisor, process, events, output = await generating_supervisor(rank=0)
    progress, ev_recv = mp_channel[RunnerEvent | RunnerTerminationError]()
    supervisor._ev_recv = ev_recv  # pyright: ignore[reportPrivateUsage]

    async def keep_writing() -> None:
        while True:
            progress.send(RunnerProgressed(task_id=TaskId("task-s")))
            await anyio.sleep(0.05)

    async with anyio.create_task_group() as tg:
        tg.start_soon(keep_writing)
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        with anyio.move_on_after(1.5):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        tg.cancel_scope.cancel()

    assert process.is_alive()
    with pytest.raises(anyio.WouldBlock):
        events.receive_nowait()  # nothing went to the cluster
    with pytest.raises(anyio.WouldBlock):
        output.receive_nowait()  # nor to the request's stream


@pytest.mark.anyio
async def test_other_ranks_are_not_judged_by_their_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, _, _ = await generating_supervisor(rank=1)

    with anyio.move_on_after(0.3):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]

    assert process.is_alive()


_OUT_OF_MEMORY = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


@pytest.mark.anyio
async def test_a_runner_out_of_gpu_memory_fails_with_the_cause() -> None:
    event_sender, event_receiver = channel[Event]()
    supervisor = await _supervisor_of_dead_runner(event_sender)

    await supervisor._check_runner(  # pyright: ignore[reportPrivateUsage]
        RunnerTerminationError(
            exception_type="RuntimeError",
            exception_message=_OUT_OF_MEMORY + ".",
            exception_repr=f"RuntimeError('{_OUT_OF_MEMORY}.')",
            traceback="",
        )
    )

    with anyio.fail_after(1):
        status = await event_receiver.receive()
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)
    assert [type(d) for d in status.runner_status.diagnostics] == [
        RunnerMetalOutOfMemory
    ]


@pytest.mark.anyio
async def test_a_request_never_ends_with_the_runners_traceback() -> None:
    """A node of the instance quit mid-reply: the runner here raised from MLX's ring. Its
    requests end with one sentence and a code, and the traceback stays in the log."""
    event_sender, _ = channel[Event]()
    supervisor = await _supervisor_of_dead_runner(event_sender)
    task = _text_generation(supervisor, "a")
    output = _streamed(supervisor, task)

    await supervisor._check_runner(  # pyright: ignore[reportPrivateUsage]
        RunnerTerminationError(
            exception_type="RuntimeError",
            exception_message="[ring] connection to a peer was lost",
            exception_repr="RuntimeError('[ring] connection to a peer was lost')",
            traceback=(
                "Traceback (most recent call last):\n"
                '  File "runner.py", line 1, in main\n'
                "RuntimeError: [ring] connection to a peer was lost\n"
            ),
        )
    )

    with anyio.fail_after(1):
        got_chunk = await output.receive()
    assert isinstance(got_chunk, ErrorChunk)
    assert got_chunk.error_message == (
        "A device serving this model disconnected before the reply finished."
    )
    assert generation_error_code(got_chunk.error_message) == "device_left"
