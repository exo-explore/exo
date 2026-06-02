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
    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

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

    def is_primary_output(self) -> bool:
        return self.device_rank == self.world_size - 1

    def is_primary_output_node(self) -> bool:
        return self.is_primary_output()


@final
class PipelineShardMetadata(BaseShardMetadata):
    pass


@final
class CfgShardMetadata(BaseShardMetadata):
    # example
    # world_size 6
    # rank prank crank
    #  0     0     0
    #  1     1     0
    #  2     2     0
    #  3     2     1
    #  4     1     1
    #  5     0     1

    @property
    def cfg_rank(self) -> int:
        # 0 = positive branch, 1 = negative branch
        return 0 if self.device_rank < self.world_size // 2 else 1

    @property
    def cfg_world_size(self) -> int:
        return 2

    @property
    def pipeline_rank(self) -> int:
        return (
            self.device_rank
            if self.cfg_rank == 0
            else (self.world_size - self.device_rank - 1)
        )

    @property
    def pipeline_world_size(self) -> int:
        return self.world_size // 2

    def is_primary_output(self) -> bool:
        """
        For CFG models: the last pipeline stage in CFG group 0 (positive prompt).
        For non-CFG models: the last pipeline stage.
        """
        assert self.pipeline_world_size == self.world_size // 2
        assert self.world_size % 2 == 0
        return self.device_rank == (self.world_size // 2) - 1


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata
)
