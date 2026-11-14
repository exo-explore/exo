from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import FrozenModel


class PlacementRequest(FrozenModel):
    """What to place: a model, how to shard it, the kind of instance, and the fewest nodes."""

    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    tensorfold_context_length: int = Field(default=8192, gt=0)
    tensorfold_parallel: int | None = Field(default=None, gt=0)
