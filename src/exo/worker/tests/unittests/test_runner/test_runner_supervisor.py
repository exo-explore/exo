from typing import cast, final

import anyio
import pytest

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
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import Sender, channel, mp_channel
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


async def _supervisor_of_dead_runner(event_sender: Sender[Event]) -> RunnerSupervisor:
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
    supervisor.shutdown = lambda: None
    return supervisor


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
    supervisor.in_progress[task.task_id] = task

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == task.command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@final
class _FinishesATaskWhileSending:
    """An event sender during whose sends the runner's last results arrive.

    While the supervisor reports one request as failed, the event loop runs the coroutine that
    forwards the runner's output, which marks another request complete.
    """

    def __init__(self) -> None:
        self.sent: list[Event] = []
        self.supervisor: RunnerSupervisor | None = None

    async def send(self, event: Event) -> None:
        self.sent.append(event)
        assert self.supervisor is not None
        if len(self.supervisor.in_progress) > 1:
            self.supervisor.in_progress.pop(next(reversed(self.supervisor.in_progress)))
        await anyio.sleep(0)


@pytest.mark.anyio
async def test_check_runner_survives_a_request_finishing_while_it_reports() -> None:
    sender = _FinishesATaskWhileSending()
    supervisor = await _supervisor_of_dead_runner(
        cast(Sender[Event], cast(object, sender))
    )
    sender.supervisor = supervisor
    first, second = _text_generation(supervisor, "a"), _text_generation(supervisor, "b")
    supervisor.in_progress[first.task_id] = first
    supervisor.in_progress[second.task_id] = second

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    failed = [e.command_id for e in sender.sent if isinstance(e, ChunkGenerated)]
    assert failed == [first.command_id]
    assert isinstance(sender.sent[-1], RunnerStatusUpdated)
