from typing import Literal, final

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import FrozenModel

StoragePolicy = Literal["manual", "auto-evict"]


@final
class StorageConfig(FrozenModel):
    max_storage: Memory | None = None
    storage_policy: StoragePolicy = "manual"
