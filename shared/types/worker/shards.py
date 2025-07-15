from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, DirectoryPath, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.models.common import ModelId


class PartitionStrategy(str, Enum):
    pipeline = "pipeline"


class ShardMetadata[PartitionStrategyT: PartitionStrategy](BaseModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    partition_strategy: PartitionStrategyT
    device_rank: int
    world_size: int
    model_id: ModelId
    model_path: DirectoryPath


class PipelineShardMetadata(ShardMetadata[PartitionStrategy.pipeline]):
    """
    Pipeline parallelism shard meta.
    """

    partition_strategy: Literal[PartitionStrategy.pipeline] = Field(
        default=PartitionStrategy.pipeline, frozen=True
    )
    start_layer: Annotated[int, Field(ge=0)]
    end_layer: Annotated[int, Field(ge=0)]


_ShardMetadata = Annotated[
    PipelineShardMetadata, Field(discriminator="partition_strategy")
]
ShardMetaParser: TypeAdapter[ShardMetadata[PartitionStrategy]] = TypeAdapter(
    _ShardMetadata
)


class ShardPlacement[PartitionStrategyT: PartitionStrategy](BaseModel):
    """
    A shard placement is the description of a model distributed across a set of nodes.
    The Generic[PartitionStrategyT] enforces that the shard assignments all use the same partition strategy.
    """

    model_id: ModelId
    shard_assignments: dict[NodeId, ShardMetadata[PartitionStrategyT]]
