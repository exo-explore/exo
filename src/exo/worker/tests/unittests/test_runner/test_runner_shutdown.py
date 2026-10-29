"""A runner shuts down cleanly when the supervisor stops listening right after it acknowledges
the shutdown, which it does as soon as the acknowledgement arrives."""

from collections.abc import Generator
from typing import cast, final

from anyio import ClosedResourceError

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, TaskAcknowledged
from exo.shared.types.tasks import Shutdown, Task
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.runners import RunnerId, RunnerShutdown
from exo.utils.channels import MpSender, mp_channel
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.runner import Runner
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

BOUND: BoundInstance = get_bound_mlx_ring_instance(
    instance_id=InstanceId("instance-shutdown"),
    model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
    runner_id=RunnerId("runner-shutdown"),
    node_id=NodeId("node-shutdown"),
)


class _IdleBuilder(Builder):
    def connect(self, bound_instance: BoundInstance) -> None:
        raise NotImplementedError

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        raise NotImplementedError

    def build(self) -> Engine:
        raise NotImplementedError

    def close(self) -> None:
        pass


@final
class _SupervisorThatStopsListening:
    """Receives the runner's events until it has the acknowledgement, then closes"""

    def __init__(self) -> None:
        self.received: list[Event] = []
        self.closed = False

    def send(self, event: Event) -> None:
        if self.closed:
            raise ClosedResourceError
        self.received.append(event)
        if isinstance(event, TaskAcknowledged):
            self.closed = True


def test_a_runner_shuts_down_cleanly_when_the_supervisor_stops_listening() -> None:
    supervisor = _SupervisorThatStopsListening()
    _, tasks = mp_channel[Task]()
    runner = Runner(
        BOUND,
        _IdleBuilder(),
        cast(MpSender[Event], cast(object, supervisor)),
        tasks,
    )
    shutdown = Shutdown(
        instance_id=BOUND.instance.instance_id, runner_id=BOUND.bound_runner_id
    )

    runner.shutdown(shutdown)

    assert isinstance(runner.current_status, RunnerShutdown)
    assert any(
        isinstance(e, TaskAcknowledged) and e.task_id == shutdown.task_id
        for e in supervisor.received
    )
    tasks.close()
