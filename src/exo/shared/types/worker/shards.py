from enum import Enum
from typing import TypeAlias, final

from pydantic import Field, field_validator

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class TensorShardMode(str, Enum):
    Greedy = "Greedy"
    Constant = "Constant"


class TensorShardStrategy(str, Enum):
    Naive = "Naive"
    Memory = "Memory"
    Compute = "Compute"
    Bandwidth = "Bandwidth"


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
    shard_weights: list[float] | None = None
    shard_mode: TensorShardMode = TensorShardMode.Constant

    @field_validator("shard_mode", mode="before")
    @classmethod
    def _coerce_shard_mode(cls, v: object) -> TensorShardMode:
        if isinstance(v, str):
            return TensorShardMode(v)
        if isinstance(v, TensorShardMode):
            return v
        raise ValueError(f"expected TensorShardMode or str, got {type(v).__name__}")


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata
)
