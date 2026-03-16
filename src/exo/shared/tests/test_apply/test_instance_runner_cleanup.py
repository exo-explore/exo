from exo.shared.apply import event_apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, RunnerDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_instance() -> Instance:
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    runner_id = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=10,
                    n_layers=10,
                )
            },
            node_to_runner={NodeId(): runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_instance_delete_sequence_removes_bound_runners_from_state() -> None:
    instance = _make_instance()
    runner_id = next(iter(instance.shard_assignments.runner_to_shard))
    state = State(
        instances={instance.instance_id: instance},
        runners={runner_id: RunnerReady()},
    )

    state = event_apply(RunnerDeleted(runner_id=runner_id), state)
    state = event_apply(InstanceDeleted(instance_id=instance.instance_id), state)

    assert instance.instance_id not in state.instances
    assert runner_id not in state.runners


def test_runner_deleted_is_idempotent_when_runner_is_already_gone() -> None:
    runner_id = RunnerId("runner-gone")
    state = State()

    updated = event_apply(RunnerDeleted(runner_id=runner_id), state)

    assert updated.runners == {}
