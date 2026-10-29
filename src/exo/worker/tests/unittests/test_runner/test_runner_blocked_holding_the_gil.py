"""What a runner sends reaches the worker even when the runner then blocks in native code that
holds the GIL, as MLX's distributed init does while it waits for the other ranks to connect.

The other ranks of an instance are only told to connect once this runner's RunnerConnecting
status reaches the cluster, so a status held back until the call returns is a deadlock."""

import ctypes
import os
import signal
import sys
from collections.abc import Generator, Iterator
from typing import cast

import anyio
import loguru
import pytest
from loguru import logger

import exo.worker.runner.supervisor as supervisor_module
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import ConnectToGroup, Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.runners import RunnerConnecting, RunnerId, RunnerIdle
from exo.utils.channels import MpReceiver, MpSender, channel
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.runner import Runner
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

BOUND: BoundInstance = get_bound_mlx_ring_instance(
    instance_id=InstanceId("instance-gil"),
    model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
    runner_id=RunnerId("runner-gil"),
    node_id=NodeId("node-gil"),
)


class _ConnectHoldsTheGil(Builder):
    def __init__(self) -> None:
        # Looked up now, because loading a library releases the GIL. A function called through
        # PyDLL keeps the GIL for as long as it runs.
        self._hold_the_gil = ctypes.PyDLL(None).sleep

    def connect(self, bound_instance: BoundInstance) -> None:
        self._hold_the_gil(600)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        raise NotImplementedError

    def build(self) -> Engine:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _runner_whose_connect_holds_the_gil(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    _cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    Runner(
        bound_instance,
        _ConnectHoldsTheGil(),
        cast(MpSender[Event], event_sender),
        task_receiver,
    ).main()


@pytest.fixture
def queued_logging() -> Iterator[None]:
    """The supervisor hands its logger to the runner's process, which only works when its
    sinks are queued, as exo sets them up: pytest's captured stderr can't be sent over."""
    logger.remove()
    handler = logger.add(sys.stderr, enqueue=True)
    yield
    logger.remove(handler)
    logger.add(sys.stderr)


@pytest.mark.anyio
@pytest.mark.usefixtures("queued_logging")
async def test_a_runner_says_it_is_connecting_before_it_blocks_holding_the_gil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor_module, "entrypoint", _runner_whose_connect_holds_the_gil
    )
    event_sender, events = channel[Event]()
    sup = await RunnerSupervisor.create(bound_instance=BOUND, event_sender=event_sender)
    connect = ConnectToGroup(instance_id=BOUND.instance.instance_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sup.run)
        try:
            with anyio.fail_after(60):
                async for event in events:
                    if isinstance(event, RunnerStatusUpdated) and isinstance(
                        event.runner_status, RunnerIdle
                    ):
                        break
            tg.start_soon(sup.start_task, connect)

            with anyio.fail_after(10):
                while not (
                    isinstance(sup.status, RunnerConnecting)
                    and connect.task_id not in sup.pending
                ):
                    await anyio.sleep(0.05)
        finally:
            os.kill(sup.runner_process.pid, signal.SIGKILL)
            sup.shutdown()
