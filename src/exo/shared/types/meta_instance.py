from typing import final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import Id, NodeId
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import FrozenModel


class MetaInstanceId(Id):
    """Identifier for a MetaInstance."""


@final
class MetaInstance(FrozenModel):
    """Declarative constraint: ensure an instance matching these parameters always exists."""

    meta_instance_id: MetaInstanceId = Field(default_factory=MetaInstanceId)
    model_card: ModelCard
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1
    node_ids: frozenset[NodeId] | None = None
