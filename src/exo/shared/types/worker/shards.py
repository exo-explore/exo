from enum import Enum
from typing import TypeAlias, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    AsymmetricTensor = "AsymmetricTensor"
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


@final
class AsymmetricTensorShardMetadata(BaseShardMetadata):
    """
    Asymmetric tensor parallelism shard metadata.

    Unlike standard tensor parallelism which splits weights 50/50 (or equally
    across N nodes), asymmetric TP splits weights proportionally to each node's
    available memory. This enables heterogeneous clusters (e.g. 128GB + 48GB)
    to run models using tensor parallelism where equal splits wouldn't fit.

    Each node holds a different fraction of each weight tensor, but ALL nodes
    compute every layer simultaneously. The all_sum reduction still works
    correctly because (x_a @ W_a^T) + (x_b @ W_b^T) = x @ W^T regardless
    of how W is partitioned.
    """

    ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of each weight tensor this node holds. "
        "e.g. 0.75 means this node gets 75% of each weight's split dimension.",
    )


ShardMetadata: TypeAlias = (
    PipelineShardMetadata
    | CfgShardMetadata
    | TensorShardMetadata
    | AsymmetricTensorShardMetadata
)
