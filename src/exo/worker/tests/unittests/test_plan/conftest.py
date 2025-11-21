from dataclasses import dataclass

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.tasks import BaseTask
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, RunnerStatus, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


@dataclass(frozen=True)
class FakeRunnerSupervisor:
    bound_instance: BoundInstance
    status: RunnerStatus


class OtherTask(BaseTask):
    pass


# TODO: Is this actually better than using Mock/Fake dataclasses?
#  e.g. commit d01cd292344df15759070966826a6c027945792b
def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_meta=ModelMetadata(
            model_id=model_id,
            pretty_name=str(model_id),
            storage_size=Memory.from_mb(100000),
            n_layers=32,
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
        hosts=[],
    )
