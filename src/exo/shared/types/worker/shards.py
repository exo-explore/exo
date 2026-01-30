from enum import Enum

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_card: ModelCard
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
                self.model_card.model_id,
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

    CFG parallelism fields:
    - cfg_rank: 0 = positive branch, 1 = negative branch (or 0 if no CFG parallel)
    - cfg_world_size: 1 = sequential CFG, 2 = parallel CFG
    """

    cfg_rank: int = 0
    cfg_world_size: int = 1

    @property
    def pipeline_world_size(self) -> int:
        return self.world_size // self.cfg_world_size

    @property
    def pipeline_rank(self) -> int:
        return self.device_rank % self.pipeline_world_size

    @property
    def is_pipeline_first(self) -> bool:
        return self.pipeline_rank == 0

    @property
    def is_pipeline_last(self) -> bool:
        return self.pipeline_rank == self.pipeline_world_size - 1

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
                self.cfg_rank,
                self.cfg_world_size,
            )
        )


class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata = PipelineShardMetadata | TensorShardMetadata
