"""A runner that doesn't get through a step on its way to serving (starting, connecting to the
instance's other runners, loading, warming up) within a bound is stopped and failed with the
reason, so that the worker starts it again instead of the instance waiting forever."""

import ctypes
import os
import signal
import sys
from collections.abc import Generator, Iterator
from typing import cast

import anyio
import loguru
import pytest
from anyio.abc import TaskStatus
from loguru import logger

import exo.worker.runner.supervisor as supervisor_module
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ConnectToGroup, StartWarmup, Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Receiver, channel, mp_channel
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.runner import Runner
from exo.worker.runner.supervisor import (
    RunnerStdioHandler,
    RunnerSupervisor,
    runner_load_timeout,
    runner_warmup_timeout,
)
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

BOUND: BoundInstance = get_bound_mlx_ring_instance(
    instance_id=InstanceId("instance-steps"),
    model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
    runner_id=RunnerId("runner-steps"),
    node_id=NodeId("node-steps"),
)


class _AliveProcess:
    """A runner process that stays alive, silent, until it is stopped."""

    def __init__(self):
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def run(self, *, task_status: TaskStatus[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        self.exitcode = -15


async def _supervisor() -> tuple[
    RunnerSupervisor,
    _AliveProcess,
    Receiver[Event],
    MpSender[Event | RunnerTerminationError],
]:
    event_sender, events = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event | RunnerTerminationError]()
    process = _AliveProcess()
    proc = cast(AsyncProcess, cast(object, process))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=BOUND.bound_shard,
        bound_instance=BOUND,
        runner_process=proc,
        _runner_stdio_handler=handler,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    return supervisor, process, events, ev_send


async def _failure_reason(events: Receiver[Event]) -> str:
    async for event in events:
        if isinstance(event, RunnerStatusUpdated) and isinstance(
            event.runner_status, RunnerFailed
        ):
            return event.runner_status.error_message or ""
    raise AssertionError("the runner was never reported failed")


@pytest.mark.anyio
async def test_a_runner_that_never_starts_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_START_TIMEOUT", 0.05)
    supervisor, process, events, _ = await _supervisor()

    with anyio.fail_after(2):
        await supervisor.run()
        reason = await _failure_reason(events)

    assert not process.is_alive()
    assert "took more than 0s to start" in reason


@pytest.mark.anyio
async def test_a_runner_that_never_connects_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_CONNECT_TIMEOUT", 0.05)
    supervisor, process, events, _ = await _supervisor()
    supervisor.shutdown = lambda: None

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            supervisor.start_task,
            ConnectToGroup(instance_id=BOUND.instance.instance_id),
        )
        with anyio.fail_after(2):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
            reason = await _failure_reason(events)
            assert not process.is_alive()
            assert "to connect to the instance's other runners" in reason
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_a_runner_that_connects_in_time_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_CONNECT_TIMEOUT", 0.2)
    supervisor, process, _, runner_events = await _supervisor()
    connect = ConnectToGroup(instance_id=BOUND.instance.instance_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(supervisor.start_task, connect)
        await anyio.sleep(0.05)
        runner_events.send(
            RunnerStatusUpdated(
                runner_id=BOUND.bound_runner_id, runner_status=RunnerConnected()
            )
        )
        with anyio.move_on_after(0.6):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        tg.cancel_scope.cancel()

    assert process.is_alive()


def test_loading_allows_for_a_slow_disk() -> None:
    card = BOUND.bound_shard.model_card
    small = card.model_copy(update={"storage_size": Memory.from_mb(700)})
    large = card.model_copy(update={"storage_size": Memory.from_gb(400)})

    # Even a small model gets minutes, and a large one as long as a slow disk takes to read it
    assert runner_load_timeout(small) >= 300
    assert runner_load_timeout(large) >= 400e9 / 20e6


def test_warming_up_allows_for_the_cpu_but_stays_bounded_on_a_mac() -> None:
    card = BOUND.bound_shard.model_card
    small = card.model_copy(update={"storage_size": Memory.from_mb(700)})
    large = card.model_copy(update={"storage_size": Memory.from_gb(100)})

    # MLX's CPU backend took about 160s to warm up the 0.7 GB model
    assert runner_warmup_timeout(small, on_metal=False) >= 4 * 160
    assert runner_warmup_timeout(large, on_metal=False) >= 100 * 4 * 160 / 0.7
    # A Mac's GPU warms it up in under a second: minutes for any model, not hours
    assert runner_warmup_timeout(small, on_metal=True) >= 600
    assert 15 * 60 <= runner_warmup_timeout(large, on_metal=True) <= 20 * 60


@pytest.mark.anyio
@pytest.mark.parametrize("on_metal", [True, False])
async def test_warming_up_is_bounded_for_where_mlx_computes(
    monkeypatch: pytest.MonkeyPatch, on_metal: bool
) -> None:
    monkeypatch.setattr(supervisor_module, "MLX_USES_METAL", on_metal)
    supervisor, _, _, _ = await _supervisor()

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            supervisor.start_task,
            StartWarmup(instance_id=BOUND.instance.instance_id),
        )
        await anyio.sleep(0.1)
        step = supervisor._step  # pyright: ignore[reportPrivateUsage]
        assert step is not None
        assert step.timeout == runner_warmup_timeout(
            BOUND.bound_shard.model_card, on_metal=on_metal
        )
        tg.cancel_scope.cancel()


class _ConnectHoldsTheGil(Builder):
    def __init__(self) -> None:
        # Looked up now, because loading a library releases the GIL. A function called through
        # PyDLL keeps the GIL for as long as it runs.
        self._hold_the_gil = ctypes.PyDLL(None).sleep

    def connect(self, bound_instance: BoundInstance) -> None:
        # As MLX's distributed init does while it waits for ranks that never connect
        self._hold_the_gil(600)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        raise NotImplementedError

    def build(self) -> Engine:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _runner_that_never_connects(
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
async def test_a_real_runner_stuck_connecting_is_stopped_and_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "entrypoint", _runner_that_never_connects)
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.1)
    monkeypatch.setattr(supervisor_module, "RUNNER_CONNECT_TIMEOUT", 1.0)
    event_sender, events = channel[Event]()
    supervisor = await RunnerSupervisor.create(
        bound_instance=BOUND, event_sender=event_sender
    )
    connect = ConnectToGroup(instance_id=BOUND.instance.instance_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)
        try:
            with anyio.fail_after(60):
                async for event in events:
                    if isinstance(event, RunnerStatusUpdated) and isinstance(
                        event.runner_status, RunnerIdle
                    ):
                        break
            tg.start_soon(supervisor.start_task, connect)

            with anyio.fail_after(20):
                reason = await _failure_reason(events)
            assert "to connect to the instance's other runners" in reason
            # the worker waiting on the runner is released, and the runner is gone
            with anyio.fail_after(5):
                while connect.task_id in supervisor.pending and not (
                    supervisor.pending[connect.task_id].is_set()
                ):
                    await anyio.sleep(0.05)
                while supervisor.runner_process.is_alive():
                    await anyio.sleep(0.05)
        finally:
            if supervisor.runner_process.is_alive():
                os.kill(supervisor.runner_process.pid, signal.SIGKILL)
            tg.cancel_scope.cancel()
