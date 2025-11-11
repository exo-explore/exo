from enum import Enum

from pydantic import Field

from exo.shared.types.models import ModelMetadata
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_meta: ModelMetadata
    device_rank: int
    world_size: int

    # Error handling; equivalent to monkey-patch, but we can't monkey-patch runner.py
    # This is kinda annoying because it allocates memory in the ShardMetadata object. Can be rethought after Shanghai.
    immediate_exception: bool = False
    should_timeout: float | None = None

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_meta.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """


class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata = PipelineShardMetadata | TensorShardMetadata
