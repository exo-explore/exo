"""The worker gives up on an instance (asks for it to be deleted) after EXO_MAX_INSTANCE_RETRIES
attempts to start its runner. Those are meant to be attempts that failed: a runner that started
and later died, its node restarted or its process killed, is not a sign the instance can't start."""

from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    RunnerStatusUpdated,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
)
from exo.utils.channels import channel
from exo.worker.main import Worker
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test/model")


def _worker() -> Worker:
    _, event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NodeId(),
        event_receiver=event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )
    return worker


async def _apply(worker: Worker, events: list[Event]) -> None:
    """Run the worker's event loop over `events`, as they arrive from the master."""
    sender, receiver = channel[IndexedEvent]()
    worker.event_receiver = receiver
    start = worker.state.last_event_applied_idx + 1
    for idx, event in enumerate(events, start=start):
        await sender.send(IndexedEvent(idx=idx, event=event))
    sender.close()
    await worker._event_applier()  # pyright: ignore[reportPrivateUsage]


def _instance(worker: Worker) -> tuple[InstanceCreated, InstanceId, RunnerId, RunnerId]:
    """A two-node instance with a runner on this worker's node and one on another node."""
    instance_id, ours, theirs = InstanceId(), RunnerId(), RunnerId()
    created = InstanceCreated(
        instance=get_mlx_ring_instance(
            instance_id,
            MODEL,
            {worker.node_id: ours, NodeId(): theirs},
            {
                ours: get_pipeline_shard_metadata(MODEL, 0, 2),
                theirs: get_pipeline_shard_metadata(MODEL, 1, 2),
            },
        )
    )
    return created, instance_id, ours, theirs


def _start_attempt(worker: Worker, instance_id: InstanceId) -> None:
    """What the worker records each time it creates a runner for an instance."""
    worker._instance_backoff.record_attempt(instance_id)  # pyright: ignore[reportPrivateUsage]


def _attempts(worker: Worker, instance_id: InstanceId) -> int:
    return worker._instance_backoff.attempts(instance_id)  # pyright: ignore[reportPrivateUsage]


async def test_a_runner_that_keeps_starting_does_not_use_up_the_retries():
    worker = _worker()
    created, instance_id, ours, _ = _instance(worker)
    await _apply(worker, [created])

    for _ in range(EXO_MAX_INSTANCE_RETRIES * 2):
        _start_attempt(worker, instance_id)
        await _apply(
            worker,
            [
                RunnerStatusUpdated(runner_id=ours, runner_status=RunnerLoading()),
                RunnerStatusUpdated(runner_id=ours, runner_status=RunnerReady()),
                # ... it serves, then its process is killed
                RunnerStatusUpdated(
                    runner_id=ours,
                    runner_status=RunnerFailed(error_message="killed", diagnostics=[]),
                ),
            ],
        )

    assert _attempts(worker, instance_id) == 0


async def test_failed_starts_in_a_row_still_add_up():
    worker = _worker()
    created, instance_id, ours, _ = _instance(worker)
    await _apply(worker, [created])

    for _ in range(EXO_MAX_INSTANCE_RETRIES):
        _start_attempt(worker, instance_id)
        await _apply(
            worker,
            [
                RunnerStatusUpdated(runner_id=ours, runner_status=RunnerLoading()),
                RunnerStatusUpdated(
                    runner_id=ours,
                    runner_status=RunnerFailed(error_message="oom", diagnostics=[]),
                ),
            ],
        )

    assert _attempts(worker, instance_id) == EXO_MAX_INSTANCE_RETRIES


async def test_only_this_nodes_runner_starting_counts():
    """Another node's runner being ready says nothing about whether ours can start."""
    worker = _worker()
    created, instance_id, _, theirs = _instance(worker)
    await _apply(worker, [created])
    _start_attempt(worker, instance_id)
    _start_attempt(worker, instance_id)

    await _apply(
        worker,
        [
            RunnerStatusUpdated(runner_id=theirs, runner_status=RunnerReady()),
            RunnerStatusUpdated(runner_id=RunnerId(), runner_status=RunnerReady()),
        ],
    )

    assert _attempts(worker, instance_id) == 2
