from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.models.common import ModelId


class PartitionStrategy(str, Enum):
    pipeline = "pipeline"


PartitionStrategyT = TypeVar(name="PartitionStrategyT", bound=PartitionStrategy)


class BaseShardMeta(BaseModel, Generic[PartitionStrategyT]):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    device_rank: int
    world_size: int
    model_id: ModelId
    model_path: DirectoryPath


class PipelineShardMeta(BaseShardMeta[Literal[PartitionStrategy.pipeline]]):
    """
    Pipeline parallelism shard meta.
    """
    model_config = ConfigDict(use_enum_values=False)

    partition_strategy: Literal[PartitionStrategy.pipeline] = PartitionStrategy.pipeline
    start_layer: Annotated[int, Field(ge=0)]
    end_layer: Annotated[int, Field(ge=0)]


_ShardMeta = Annotated[PipelineShardMeta, Field(discriminator="partition_strategy")]
ShardMeta = _ShardMeta  # Public alias for the discriminated union
ShardMetaAdapter: TypeAdapter[BaseShardMeta[PartitionStrategy]] = TypeAdapter(
    _ShardMeta
)


class ShardPlacement(BaseModel, Generic[PartitionStrategyT]):
    """
    A shard placement is the description of a model distributed across a set of nodes.
    The Generic[PartitionStrategyT] enforces that the shard assignments all use the same partition strategy.
    """

    model_id: ModelId
    shard_assignments: dict[NodeId, BaseShardMeta[PartitionStrategyT]]
