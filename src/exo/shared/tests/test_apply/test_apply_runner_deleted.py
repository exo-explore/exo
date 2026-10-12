from exo.shared.apply import apply_instance_deleted, apply_runner_status_updated
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import InstanceDeleted, RunnerStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")


def instance_with(*runner_ids: RunnerId) -> Instance:
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={NodeId(f"node-{i}"): rid for i, rid in enumerate(runner_ids)},
        runner_to_shard={
            rid: get_pipeline_shard_metadata(
                MODEL, device_rank=i, world_size=len(runner_ids)
            )
            for i, rid in enumerate(runner_ids)
        },
    )


def test_apply_runner_shutdown_removes_runner():
    runner_id = RunnerId()
    instance = instance_with(runner_id)
    state = State(
        instances={instance.instance_id: instance}, runners={runner_id: RunnerIdle()}
    )

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()), state
    )

    assert runner_id not in new_state.runners


def test_apply_runner_status_updated_adds_runner():
    runner_id = RunnerId()
    instance = instance_with(runner_id)
    state = State(instances={instance.instance_id: instance})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id in new_state.runners


def test_deleting_an_instance_forgets_its_runners():
    kept, first, second = RunnerId(), RunnerId(), RunnerId()
    other = instance_with(kept)
    deleted = instance_with(first, second)
    state = State(
        instances={other.instance_id: other, deleted.instance_id: deleted},
        runners={kept: RunnerReady(), first: RunnerReady(), second: RunnerIdle()},
        prefill_server_ports={kept: 1, first: 2},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=deleted.instance_id), state
    )

    assert dict(new_state.runners) == {kept: RunnerReady()}
    assert dict(new_state.prefill_server_ports) == {kept: 1}


def test_a_runner_reporting_after_its_instance_was_deleted_is_not_kept():
    # Its shutdown is reported after the instance is gone, and its last report (that it
    # has shut down) usually never arrives: keeping it would leave it in the state forever
    runner_id = RunnerId()

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShuttingDown()),
        State(),
    )

    assert runner_id not in new_state.runners
