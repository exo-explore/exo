"""The real Runner must notice engine death even while its task stream is idle."""

import queue
import threading
from collections.abc import Generator, Iterable, Iterator
from types import TracebackType
from typing import BinaryIO, Self, cast

from exo.shared.types.chunks import Chunk
from exo.shared.types.common import NodeId
from exo.shared.types.events import RunnerEvent, RunnerStatusUpdated
from exo.shared.types.tasks import GenerationTask, LoadModel, StartWarmup, Task, TaskId
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    TensorFoldInstance,
)
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import whole_model_shard
from exo.testkit.cluster import StubModel
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.runner import Runner


class IdleEngine(Engine):
    supports_prefill = False

    def __init__(self) -> None:
        self._cancelled_tasks: set[TaskId] = set()
        self.dead = threading.Event()
        self.checked = threading.Event()
        self.closed = threading.Event()

    def warmup(self) -> None:
        pass

    def check_health(self) -> None:
        self.checked.set()
        if self.dead.is_set():
            raise RuntimeError("Native process died while the runner was idle")

    def submit(self, task: GenerationTask) -> None:
        raise AssertionError("This test must never submit a generation request")

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        raise AssertionError("An idle runner must not enter generation")

    def close(self) -> None:
        self.closed.set()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise AssertionError("The native engine must not start MLX prefill serving")


class IdleBuilder(Builder):
    def __init__(self, engine: IdleEngine):
        self.engine = engine

    def connect(self, bound_instance: BoundInstance) -> None:
        raise AssertionError("A native instance must not join an MLX group")

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        layers = bound_instance.bound_shard.n_layers
        yield ModelLoadingResponse(layers_loaded=layers, total=layers)

    def build(self) -> Engine:
        return self.engine

    def close(self) -> None:
        self.engine.close()


class Events:
    def __init__(self) -> None:
        self.ready = threading.Event()

    def send(self, event: RunnerEvent) -> None:
        if isinstance(event, RunnerStatusUpdated) and isinstance(
            event.runner_status, RunnerReady
        ):
            self.ready.set()


class TaskInput:
    """A controllable task channel; leave it open and empty during the assertion."""

    def __init__(self) -> None:
        self.items: queue.Queue[Task | None] = queue.Queue()
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[Task]:
        while (task := self.items.get()) is not None:
            yield task

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.items.put(None)


def test_idle_runner_fails_and_releases_engine_without_a_new_request() -> None:
    model = StubModel().card()
    node, runner_id = NodeId("idle-health-node"), RunnerId("idle-health-runner")
    instance = TensorFoldInstance(
        instance_id=InstanceId("idle-health-instance"),
        shard_assignments=ShardAssignments(
            model_id=model.model_id,
            node_to_runner={node: runner_id},
            runner_to_shard={runner_id: whole_model_shard(model)},
        ),
    )
    bound = BoundInstance(
        instance=instance, bound_node_id=node, bound_runner_id=runner_id
    )
    engine, events, tasks = IdleEngine(), Events(), TaskInput()
    runner = Runner(
        bound,
        IdleBuilder(engine),
        cast(MpSender[RunnerEvent], cast(object, events)),
        cast(MpReceiver[Task], cast(object, tasks)),
    )
    for task in (
        LoadModel(instance_id=instance.instance_id),
        StartWarmup(instance_id=instance.instance_id),
    ):
        tasks.items.put(task)
    errors: list[BaseException] = []
    finished = threading.Event()

    def run() -> None:
        try:
            runner.main()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert events.ready.wait(timeout=2), "The real runner never became ready"
        assert engine.checked.wait(timeout=2), "An idle runner never checked its engine"
        assert not tasks.closed and not finished.is_set()
        engine.dead.set()
        assert finished.wait(timeout=2), (
            "Engine death left an idle runner blocked forever"
        )
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert "Native process died" in str(errors[0])
        assert engine.closed.is_set(), "The failed engine was not released"
        assert tasks.closed
    finally:
        tasks.close()
        thread.join(timeout=2)
    assert not thread.is_alive()
