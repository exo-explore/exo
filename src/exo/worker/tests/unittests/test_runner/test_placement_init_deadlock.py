from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import ConnectToGroup, Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerConnecting, RunnerFailed, RunnerId
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import ClosedResourceError, WouldBlock, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

BOUND: BoundInstance = get_bound_mlx_ring_instance(
    instance_id=InstanceId("instance-test-deadlock"),
    model_id=ModelId("mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
    runner_id=RunnerId("runner-test-deadlock"),
    node_id=NodeId("node-test-deadlock"),
)


class _DummySilentProcess:
    exitcode = 0

    def __init__(self):
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    async def run(self, *, task_status: anyio.abc.TaskStatus[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        self._alive = False
        self.exitcode = -15


def test_mp_sender_flush():
    """Verify MpSender.flush ensures all items in buffer are drained to the OS pipe."""
    sender, receiver = mp_channel[str]()
    try:
        for i in range(5):
            sender.send_nowait(f"msg-{i}")

        # Call flush()
        sender.flush(timeout=2.0)

        # mp.Queue internal _buffer should be drained
        buf = getattr(sender._state.buffer, "_buffer", None)
        assert buf is not None
        assert len(buf) == 0

        # Receiver should be able to collect all items
        for i in range(5):
            assert receiver.receive_nowait() == f"msg-{i}"
    finally:
        sender.close()
        receiver.close()


@pytest.mark.anyio
async def test_start_task_bounded_timeout_when_runner_wedged(
    monkeypatch: pytest.MonkeyPatch,
):
    """When a runner never acknowledges ConnectToGroup, start_task must timeout rather than hang forever."""
    import exo.worker.runner.supervisor as sup_mod

    # Set timeout to a short duration for test
    monkeypatch.setattr(sup_mod, "RUNNER_TASK_ACK_TIMEOUT", 0.3)

    event_sender, _ = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    proc = cast(AsyncProcess, cast(object, _DummySilentProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )

    supervisor = RunnerSupervisor(
        shard_metadata=BOUND.bound_shard,
        bound_instance=BOUND,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=5.0,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    connect_task = ConnectToGroup(instance_id=BOUND.instance.instance_id)

    # Should raise TimeoutError within ~0.3s instead of hanging indefinitely
    with pytest.raises(TimeoutError):
        with anyio.fail_after(2.0):
            await supervisor.start_task(connect_task)

    # Verify task was cleaned up from pending
    assert connect_task.task_id not in supervisor.pending
    assert connect_task.task_id not in supervisor.in_progress

    # Cleanup
    supervisor.shutdown()
    task_receiver.close()
    cancel_receiver.close()


@pytest.mark.anyio
async def test_watchdog_detects_initialization_timeout():
    """Verify supervisor watchdog trips when a runner is stuck in connecting state beyond initialize_timeout."""
    event_sender, event_receiver = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    proc = cast(AsyncProcess, cast(object, _DummySilentProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )

    supervisor = RunnerSupervisor(
        shard_metadata=BOUND.bound_shard,
        bound_instance=BOUND,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=0.2,  # Short timeout for testing
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    # Simulate runner entering RunnerConnecting with work in progress
    supervisor.status = RunnerConnecting()
    task = ConnectToGroup(instance_id=BOUND.instance.instance_id)
    supervisor.in_progress[task.task_id] = task
    # Simulate time passing with no word from runner
    supervisor._last_heard = anyio.current_time() - 0.5

    assert supervisor._initialization_timed_out() is True

    # Run _check_runner to verify it marks RunnerFailed
    await supervisor._check_runner(TimeoutError("Initialization timed out"))
    assert isinstance(supervisor.status, RunnerFailed)
    assert "Initialization timed out" in supervisor.status.error_message

    # Verify RunnerStatusUpdated was sent
    received_status = False
    while True:
        try:
            ev = event_receiver.receive_nowait()
            if isinstance(ev, RunnerStatusUpdated) and isinstance(
                ev.runner_status, RunnerFailed
            ):
                received_status = True
                break
        except (WouldBlock, ClosedResourceError):
            break
    assert received_status is True

    # Cleanup
    supervisor.shutdown()
    task_receiver.close()
    cancel_receiver.close()
