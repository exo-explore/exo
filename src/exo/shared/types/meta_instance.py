from typing import final

from pydantic import Field

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import MetaInstanceId, NodeId
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import FrozenModel


@final
class MetaInstance(FrozenModel):
    """Declarative constraint: ensure an instance matching these parameters always exists."""

    meta_instance_id: MetaInstanceId = Field(default_factory=MetaInstanceId)
    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1
    node_ids: list[NodeId] | None = None
    # Failure tracking
    placement_error: str | None = None
    consecutive_failures: int = 0
    last_failure_error: str | None = None
