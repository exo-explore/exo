from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.models.common import ModelId
from shared.types.worker.common import RunnerId


class ShardType(str, Enum):
    PipelineParallel = "PipelineParallel"


ShardTypeT = TypeVar("ShardTypeT", bound=ShardType)


class ShardData(BaseModel, Generic[ShardTypeT]):
    shard_type: ShardTypeT


class Shard(BaseModel, Generic[ShardTypeT]):
    shard_data: ShardData[ShardTypeT]
    runner_id: RunnerId


class ShardPlacement(BaseModel):
    model_id: ModelId
    shard_assignments: dict[NodeId, Shard[ShardType]]
