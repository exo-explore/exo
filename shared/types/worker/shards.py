from enum import Enum
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, DirectoryPath, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.models import ModelId


class PartitionStrategy(str, Enum):
    pipeline = "pipeline"


PartitionStrategyT = TypeVar("PartitionStrategyT", bound=PartitionStrategy, covariant=True)


class BaseShardMetadata(BaseModel, Generic[PartitionStrategyT]):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    partition_strategy: PartitionStrategyT
    device_rank: int
    world_size: int
    model_id: ModelId
    model_path: DirectoryPath


class PipelineShardMetadata(BaseShardMetadata[Literal[PartitionStrategy.pipeline]]):
    """
    Pipeline parallelism shard meta.
    """

    partition_strategy: Literal[PartitionStrategy.pipeline] = Field(
        default=PartitionStrategy.pipeline, frozen=True
    )
    start_layer: Annotated[int, Field(ge=0)]
    end_layer: Annotated[int, Field(ge=0)]


ShardMetadata = Annotated[
    PipelineShardMetadata, Field(discriminator="partition_strategy")
]
ShardMetadataParser: TypeAdapter[ShardMetadata] = TypeAdapter(
    ShardMetadata
)

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

# "ShardMeta" is a widely-used alias for the concrete, fully-parameterised
# `ShardMetadata` type.  Defining it here avoids repetitive generic
# parameters at call-sites and resolves unknown-import diagnostics in
# downstream modules.

ShardMeta: TypeAlias = ShardMetadata


class ShardPlacement(BaseModel, Generic[PartitionStrategyT]):
    """
    A shard placement is the description of a model distributed across a set of nodes.
    The Generic[PartitionStrategyT] enforces that the shard assignments all use the same partition strategy.
    """

    model_id: ModelId
    shard_assignments: dict[NodeId, BaseShardMetadata[PartitionStrategyT]]
