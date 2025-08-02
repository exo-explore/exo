from enum import Enum
from typing import Annotated, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.common import NodeId
from shared.types.models import ModelId, ModelMetadata


class PartitionStrategy(str, Enum):
    pipeline = "pipeline"


PartitionStrategyT = TypeVar("PartitionStrategyT", bound=PartitionStrategy, covariant=True)


class BaseShardMetadata(BaseModel, Generic[PartitionStrategyT]):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_meta: ModelMetadata
    partition_strategy: PartitionStrategyT
    device_rank: int
    world_size: int
    
    # Error handling; equivalent to monkey-patch, but we can't monkey-patch runner.py
    # This is kinda annoying because it allocates memory in the ShardMetadata object. Can be rethought after Shanghai.
    immediate_exception: bool = False
    should_timeout: Optional[float] = None


class PipelineShardMetadata(BaseShardMetadata[Literal[PartitionStrategy.pipeline]]):
    """
    Pipeline parallelism shard meta.
    
    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """

    partition_strategy: Literal[PartitionStrategy.pipeline] = Field(
        default=PartitionStrategy.pipeline, frozen=True
    )
    start_layer: Annotated[int, Field(ge=0)]
    end_layer: Annotated[int, Field(ge=0)]
    n_layers: Annotated[int, Field(ge=0)]

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0
    
    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash((self.model_meta.model_id, self.start_layer, self.end_layer, self.n_layers))


ShardMetadata = Annotated[
    PipelineShardMetadata, Field(discriminator="partition_strategy")
]
ShardMetadataParser: TypeAdapter[ShardMetadata] = TypeAdapter(
    ShardMetadata
)


class ShardPlacement(BaseModel, Generic[PartitionStrategyT]):
    """
    A shard placement is the description of a model distributed across a set of nodes.
    The Generic[PartitionStrategyT] enforces that the shard assignments all use the same partition strategy.
    """

    model_id: ModelId
    shard_assignments: dict[NodeId, BaseShardMetadata[PartitionStrategyT]]
