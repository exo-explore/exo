from enum import Enum
from typing import Generic, TypeVar, Annotated, Literal

from pydantic import BaseModel, DirectoryPath, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.models.common import ModelId

class PartitionStrategy(str, Enum):
    pipeline = 'pipeline'

PartitionStrategyT = TypeVar(name='PartitionStrategyT', bound=PartitionStrategy)

class BaseModelShardMeta(BaseModel, Generic[PartitionStrategyT]):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """
    device_rank: int
    world_size: int
    model_id: ModelId
    model_path: DirectoryPath # pydantic DirectoryPath ensures that the directory exists.

class PipelineShardMeta(BaseModelShardMeta[PartitionStrategy.pipeline]):
    """
    Pipeline parallelism shard meta.
    """
    partition_strategy: Literal[PartitionStrategy.pipeline] = Field(
        default=PartitionStrategy.pipeline, frozen=True
    )
    start_layer: Annotated[int, Field(ge=0)]
    end_layer: Annotated[int, Field(ge=0)]

ShardMeta = Annotated[
    PipelineShardMeta,
    Field(discriminator="partition_strategy")
]
ShardMetaAdapter: TypeAdapter[ShardMeta] = TypeAdapter(ShardMeta)


class ShardPlacement(BaseModel, Generic[PartitionStrategyT]):
    """
    A shard placement is the description of a model distributed across a set of nodes.
    The Generic[PartitionStrategyT] enforces that the shard assignments all use the same partition strategy.
    """
    model_id: ModelId
    shard_assignments: dict[NodeId, BaseModelShardMeta[PartitionStrategyT]]
