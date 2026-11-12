"""The runner tells its engine when it has no request in flight (Engine.on_idle), once the
replies are out. Work deferred so that it doesn't delay a reply runs then: the prefix cache
writes its entries to disk (EXO_PREFIX_CACHE_DISK)."""

import threading
from collections.abc import Generator, Iterable
from typing import BinaryIO, cast

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
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import MpSender, mp_channel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.runner import Runner
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


class Collector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class FinishingEngine(Engine):
    """Finishes every request it is given at its first step, and notes when it is told it's
    idle: whether a request was in flight then, and whether its reply was out."""

    def __init__(self, events: Collector, on_reply_out: threading.Event) -> None:
        self._cancelled_tasks: set[TaskId] = set()
        self._submitted: list[TaskId] = []
        self._in_flight: set[TaskId] = set()
        self._events = events
        self._on_reply_out = on_reply_out
        self.idle_calls: list[tuple[bool, bool]] = []  # (in flight, reply out)

    def warmup(self) -> None:
        pass

    def submit(self, task: GenerationTask) -> None:
        self._submitted.append(task.task_id)
        self._in_flight.add(task.task_id)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        finished = [(task_id, FinishedResponse()) for task_id in self._submitted]
        self._submitted = []
        return finished

    def on_idle(self) -> None:
        reply_out = any(
            isinstance(event, TaskStatusUpdated)
            and event.task_status == TaskStatus.Complete
            for event in self._events.events
        )
        if reply_out:
            self._in_flight.clear()
        self.idle_calls.append((bool(self._in_flight), reply_out))
        if reply_out:
            self._on_reply_out.set()

    def close(self) -> None:
        pass

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError


class FinishingBuilder(Builder):
    def __init__(self, engine: FinishingEngine) -> None:
        self.engine = engine

    def connect(self, bound_instance: BoundInstance) -> None:
        pass

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        yield from ()

    def build(self) -> Engine:
        return self.engine

    def close(self) -> None:
        pass


def test_the_engine_is_told_it_is_idle_once_the_reply_is_out() -> None:
    events = Collector()
    reply_out = threading.Event()
    engine = FinishingEngine(events, reply_out)
    task_sender, task_receiver = mp_channel[Task]()
    the_request = request()

    def shut_down_when_idle() -> None:
        reply_out.wait(timeout=30)
        task_sender.send(Shutdown(instance_id=INSTANCE, runner_id=RUNNER))

    with task_sender:
        for task in (
            LoadModel(instance_id=INSTANCE),
            StartWarmup(instance_id=INSTANCE),
            the_request,
        ):
            task_sender.send(task)
        stopper = threading.Thread(target=shut_down_when_idle, daemon=True)
        stopper.start()
        # The runner closes its end as it stops; the test's end stays open meanwhile
        task_receiver.close = lambda: None
        task_receiver.join = lambda: None
        Runner(
            bound(),
            FinishingBuilder(engine),
            cast(MpSender[RunnerEvent], cast(object, events)),
            task_receiver,
        ).main()
        stopper.join(timeout=5)

    assert reply_out.is_set(), engine.idle_calls
    # Never told while the request was being served
    assert not any(in_flight for in_flight, _ in engine.idle_calls), engine.idle_calls
