from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel


class ShardType(str, Enum):
    PipelineParallel = "PipelineParallel"


ShardTypeT = TypeVar("ShardTypeT", bound=ShardType)


class ShardData(BaseModel, Generic[ShardTypeT]):
    shard_type: ShardTypeT
