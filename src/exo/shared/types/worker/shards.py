from enum import Enum
from typing import TypeAlias, final

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


@final
class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """


@final
class CfgShardMetadata(BaseShardMetadata):
    """Shard metadata for CFG-parallel image generation models."""

    cfg_rank: int  # 0 = positive branch, 1 = negative branch
    cfg_world_size: int = 2

    # Pipeline-relative coordinates (computed at placement time)
    pipeline_rank: int  # rank within the pipeline group (0, 1, 2, ...)
    pipeline_world_size: int  # number of nodes per pipeline group


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata
)


def is_primary_output_shard(shard: ShardMetadata) -> bool:
    """Whether an image model's runner with this shard emits the images: the last pipeline
    stage (of CFG group 0, the positive prompt, for CFG models)."""
    if isinstance(shard, CfgShardMetadata):
        return (
            shard.pipeline_rank == shard.pipeline_world_size - 1 and shard.cfg_rank == 0
        )
    if isinstance(shard, PipelineShardMetadata):
        return shard.device_rank == shard.world_size - 1
    return False


def whole_model_shard(card: ModelCard) -> PipelineShardMetadata:
    """All of a model, as one device downloads it."""
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
