from __future__ import annotations

import multiprocessing
import os
import signal as signal_module
from collections.abc import Callable
from multiprocessing.sharedctypes import Synchronized
from typing import Any

import anyio

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.runners import RunnerFailed, RunnerIdle, RunnerShutdown
from exo.utils.channels import Receiver, Sender, channel, mp_channel
from exo.worker.runner.runner_supervisor import (
    HEALTH_CHECK_INTERVAL_SECONDS,
    HEARTBEAT_STALE_CHECKS,
    RunnerSupervisor,
)

from ...constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
)
from ..conftest import get_bound_mlx_ring_instance


def _die_immediately() -> None:
    """Subprocess target that exits with a non-zero code."""
    os._exit(1)


def _die_with_signal() -> None:
    """Subprocess target that kills itself with SIGKILL (simulates OOM)."""
    os.kill(os.getpid(), signal_module.SIGKILL)


def _exit_cleanly() -> None:
    """Subprocess target that exits with code 0."""
    os._exit(0)


def _hang_forever() -> None:
    """Subprocess target that hangs without updating heartbeat (simulates freeze)."""
    import time

    # Write one heartbeat so the supervisor starts tracking, then stop.
    time.sleep(100000)


def _build_supervisor(
    event_sender: Sender[Event],
    target: Callable[..., Any],
) -> RunnerSupervisor:
    """Build a RunnerSupervisor with a custom subprocess target.

    Uses a clone of event_sender (matching real Worker behavior) so that
    closing the supervisor's copy doesn't close the test's receiver.
    """
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NODE_A,
    )

    _ev_send, ev_recv = mp_channel[Event]()
    task_sender, _task_recv = mp_channel[Task]()
    runner_process = multiprocessing.Process(target=target, daemon=True)
    heartbeat: Synchronized[int] = multiprocessing.Value("Q", 0)

    return RunnerSupervisor(
        bound_instance=bound_instance,
        shard_metadata=bound_instance.bound_shard,
        runner_process=runner_process,
        initialize_timeout=10,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender.clone(),
        _heartbeat=heartbeat,
    )


def _collect_failed_events(
    event_receiver: Receiver[Event],
) -> list[RunnerFailed]:
    """Drain the receiver and return all RunnerFailed statuses."""
    out: list[RunnerFailed] = []
    while True:
        try:
            event = event_receiver.receive_nowait()
        except Exception:
            break
        if isinstance(event, RunnerStatusUpdated) and isinstance(
            event.runner_status, RunnerFailed
        ):
            out.append(event.runner_status)
    return out


async def test_health_check_detects_dead_process():
    """When the runner process dies with a non-zero exit code, the health check
    should emit a RunnerFailed event and run() should return."""
    event_sender, event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _die_immediately)

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    failures = _collect_failed_events(event_receiver)
    assert len(failures) == 1
    assert failures[0].error_message is not None
    assert "exitcode=1" in failures[0].error_message


async def test_health_check_detects_signal_death():
    """When the runner process is killed by a signal (e.g. OOM -> SIGKILL),
    the health check should report the signal in the failure message."""
    event_sender, event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _die_with_signal)

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    failures = _collect_failed_events(event_receiver)
    assert len(failures) == 1
    assert failures[0].error_message is not None
    assert "signal=9" in failures[0].error_message


async def test_health_check_releases_pending_tasks():
    """When the runner dies, any pending start_task() waiters should be unblocked."""
    event_sender, _event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _die_immediately)

    # Register a pending waiter as if start_task() was waiting for acknowledgement
    task_event = anyio.Event()
    tid = TaskId("pending-task")
    supervisor.pending[tid] = task_event

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    assert task_event.is_set()


async def test_clean_exit_no_failure_when_shutdown_status():
    """When the runner was in RunnerShutdown status and exits with code 0,
    no RunnerFailed event should be emitted."""
    event_sender, event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _exit_cleanly)

    # Simulate that the runner had already reported shutdown via events
    supervisor.status = RunnerShutdown()

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    failures = _collect_failed_events(event_receiver)
    assert len(failures) == 0


async def test_unexpected_exit_code_zero_emits_failure():
    """When the runner exits with code 0 but was NOT in a shutdown state,
    this is unexpected and should still emit RunnerFailed."""
    event_sender, event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _exit_cleanly)

    assert isinstance(supervisor.status, RunnerIdle)

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    failures = _collect_failed_events(event_receiver)
    assert len(failures) == 1
    assert failures[0].error_message is not None
    assert "exitcode=0" in failures[0].error_message


async def test_heartbeat_timeout_detects_unresponsive_process():
    """When the runner process is alive but its heartbeat goes stale,
    the health check should kill it and emit RunnerFailed."""
    event_sender, event_receiver = channel[Event]()
    supervisor = _build_supervisor(event_sender, _hang_forever)

    # Pre-seed the heartbeat counter with a non-zero value and set the
    # supervisor's last-seen value to match so it appears stale immediately.
    # Set stale count to HEARTBEAT_STALE_CHECKS - 1 so a single check triggers.
    supervisor._heartbeat.value = 42  # pyright: ignore[reportPrivateUsage]
    supervisor._last_heartbeat_value = 42  # pyright: ignore[reportPrivateUsage]
    supervisor._heartbeat_stale_count = HEARTBEAT_STALE_CHECKS - 1  # pyright: ignore[reportPrivateUsage]

    with anyio.fail_after(HEALTH_CHECK_INTERVAL_SECONDS + 5):
        await supervisor.run()

    failures = _collect_failed_events(event_receiver)
    assert len(failures) == 1
    assert failures[0].error_message is not None
    assert "unresponsive" in failures[0].error_message.lower()
