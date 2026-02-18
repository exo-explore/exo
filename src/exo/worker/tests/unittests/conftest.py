from dataclasses import dataclass, field

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import BaseTask, TaskId
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, RunnerStatus, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


# Runner supervisor without multiprocessing logic.
@dataclass(frozen=True)
class FakeRunnerSupervisor:
    bound_instance: BoundInstance
    status: RunnerStatus
    completed: set[TaskId] = field(default_factory=set)


class OtherTask(BaseTask):
    pass


# TODO: Is this actually better than using Mock/Fake dataclasses?
#  e.g. commit d01cd292344df15759070966826a6c027945792b
def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100000),
            n_layers=32,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


def get_shard_assignments(
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
    runner_to_shard: dict[RunnerId, ShardMetadata],
) -> ShardAssignments:
    return ShardAssignments(
        model_id=model_id,
        node_to_runner=node_to_runner,
        runner_to_shard=runner_to_shard,
    )


def get_mlx_ring_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
    runner_to_shard: dict[RunnerId, ShardMetadata],
) -> Instance:
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=get_shard_assignments(
            model_id, node_to_runner, runner_to_shard
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def get_bound_mlx_ring_instance(
    instance_id: InstanceId, model_id: ModelId, runner_id: RunnerId, node_id: NodeId
) -> BoundInstance:
    shard = get_pipeline_shard_metadata(model_id=model_id, device_rank=0, world_size=2)
    other_shard = get_pipeline_shard_metadata(
        model_id=model_id, device_rank=1, world_size=2
    )
    instance = get_mlx_ring_instance(
        instance_id=instance_id,
        model_id=model_id,
        node_to_runner={
            node_id: runner_id,
            NodeId("other_node"): RunnerId("other_runner"),
        },
        runner_to_shard={runner_id: shard, RunnerId("other_runner"): other_shard},
    )
    return BoundInstance(
        instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
    )
