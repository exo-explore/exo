from exo.shared.apply import apply_instance_deleted, apply_runner_status_updated
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, RunnerStatusUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def test_apply_runner_shutdown_removes_runner():
    runner_id = RunnerId()
    state = State(runners={runner_id: RunnerIdle()})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()), state
    )

    assert runner_id not in new_state.runners


def test_apply_runner_status_updated_adds_runner():
    runner_id = RunnerId()
    state = State()

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id in new_state.runners


def test_apply_instance_deleted_removes_owned_runners():
    instance_id = InstanceId()
    runner_id = RunnerId()
    unrelated_runner_id = RunnerId()
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={NodeId(): runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    state = State(
        instances={instance_id: instance},
        runners={runner_id: RunnerIdle(), unrelated_runner_id: RunnerIdle()},
    )

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=instance_id), state)

    assert instance_id not in new_state.instances
    assert runner_id not in new_state.runners
    assert unrelated_runner_id in new_state.runners
