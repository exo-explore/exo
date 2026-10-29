from exo.shared.types.events import InstanceDeleted, RunnerStatusUpdated
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from exo.worker.plan import instance_to_reset_backoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def _make_runners() -> dict[RunnerId, FakeRunnerSupervisor]:
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    return {
        RUNNER_1_ID: FakeRunnerSupervisor(
            bound_instance=bound_instance, status=RunnerConnected()
        )
    }


def test_resets_backoff_when_local_runner_becomes_ready():
    event = RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady())

    result = instance_to_reset_backoff(event, _make_runners())  # type: ignore[arg-type]

    assert result == INSTANCE_1_ID


def test_resets_backoff_when_local_runner_becomes_running():
    event = RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerRunning())

    result = instance_to_reset_backoff(event, _make_runners())  # type: ignore[arg-type]

    assert result == INSTANCE_1_ID


def test_does_not_reset_on_runner_connected():
    """RunnerConnected is reached before LoadModel -- a rank that connects but
    keeps crashing during load (bad weights, OOM) must still trip the
    retry-exhaustion circuit breaker, so resetting here would be wrong."""
    event = RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected())

    result = instance_to_reset_backoff(event, _make_runners())  # type: ignore[arg-type]

    assert result is None


def test_does_not_reset_for_unknown_runner():
    event = RunnerStatusUpdated(runner_id=RUNNER_2_ID, runner_status=RunnerReady())

    result = instance_to_reset_backoff(event, _make_runners())  # type: ignore[arg-type]

    assert result is None


def test_ignores_non_runner_status_events():
    event = InstanceDeleted(instance_id=INSTANCE_1_ID)

    result = instance_to_reset_backoff(event, _make_runners())  # type: ignore[arg-type]

    assert result is None
