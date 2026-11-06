"""A request the ranks cancelled is reported once, and its supervisor forgets it.

The runner used to drop a cancelled request without a word, so its supervisor kept the whole
task (prompt, images) in `in_progress` for as long as the runner lived, and reported it as
failed when the runner later failed."""

from collections.abc import Generator, Iterable
from typing import BinaryIO, cast

import anyio
from anyio.abc import TaskStatus as Started

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import Event, RunnerEvent, TaskStatusUpdated
from exo.shared.types.tasks import (
    GenerationTask,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpSender, channel, mp_channel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.runner import Runner
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")
NODE = NodeId("node")
RUNNER = RunnerId("runner")
INSTANCE = InstanceId("instance")


def bound() -> BoundInstance:
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE,
        model_id=MODEL,
        node_to_runner={NODE: RUNNER},
        runner_to_shard={RUNNER: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )
    return BoundInstance(instance=instance, bound_runner_id=RUNNER, bound_node_id=NODE)


def request() -> TextGeneration:
    return TextGeneration(
        task_id=TaskId("request"),
        instance_id=INSTANCE,
        command_id=CommandId("command"),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


class CancellingEngine(Engine):
    """Every request it is given, the ranks agree to cancel"""

    def __init__(self) -> None:
        self._cancelled_tasks: set[TaskId] = set()
        self._submitted: list[TaskId] = []

    def warmup(self) -> None:
        pass

    def submit(self, task: GenerationTask) -> None:
        self._submitted.append(task.task_id)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        cancelled = [(task_id, CancelledResponse()) for task_id in self._submitted]
        self._submitted = []
        return cancelled

    def close(self) -> None:
        pass

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError


class CancellingBuilder(Builder):
    def connect(self, bound_instance: BoundInstance) -> None:
        pass

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        yield from ()

    def build(self) -> Engine:
        return CancellingEngine()

    def close(self) -> None:
        pass


class Collector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


def test_a_runner_reports_a_request_the_ranks_cancelled() -> None:
    events = Collector()
    task_sender, task_receiver = mp_channel[Task]()
    the_request = request()
    with task_sender:
        for task in (
            LoadModel(instance_id=INSTANCE),
            StartWarmup(instance_id=INSTANCE),
            the_request,
            Shutdown(instance_id=INSTANCE, runner_id=RUNNER),
        ):
            task_sender.send(task)
        # The runner closes its end as it stops; the test's end stays open meanwhile
        task_receiver.close = lambda: None
        task_receiver.join = lambda: None
        Runner(
            bound(),
            CancellingBuilder(),
            cast(MpSender[RunnerEvent], cast(object, events)),
            task_receiver,
        ).main()

    cancelled = [
        event
        for event in events.events
        if isinstance(event, TaskStatusUpdated)
        and event.task_id == the_request.task_id
        and event.task_status == TaskStatus.Cancelled
    ]
    assert len(cancelled) == 1


class _SilentProcess:
    exitcode = 0

    def __init__(self) -> None:
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()

    def is_alive(self) -> bool:
        return True

    async def run(self, *, task_status: Started[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        pass


async def test_a_supervisor_forgets_a_request_its_runner_cancelled() -> None:
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    runner_events, event_receiver = mp_channel[RunnerEvent | RunnerTerminationError]()
    process = cast(AsyncProcess, cast(object, _SilentProcess()))
    supervisor = RunnerSupervisor(
        shard_metadata=bound().bound_shard,
        bound_instance=bound(),
        runner_process=process,
        _runner_stdio_handler=await RunnerStdioHandler.create(
            stdout_rx=process.stdout, stderr_rx=process.stderr
        ),
        _ev_recv=event_receiver,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerReady()
    the_request = request()
    supervisor.in_progress[the_request.task_id] = the_request

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)
        runner_events.send(
            TaskStatusUpdated(
                task_id=the_request.task_id, task_status=TaskStatus.Cancelled
            )
        )
        with anyio.fail_after(5):
            while the_request.task_id in supervisor.in_progress:
                await anyio.sleep(0.01)
        tg.cancel_scope.cancel()
    assert the_request.task_id in supervisor.completed
