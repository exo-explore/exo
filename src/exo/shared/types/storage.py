from typing import Any, Literal, Self, final

from exo.shared.models.model_cards import ModelId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import FrozenModel

StoragePolicy = Literal["manual", "auto-evict"]


@final
class StorageConfig(FrozenModel):
    max_storage: Memory | None = None
    storage_policy: StoragePolicy = "manual"

    @classmethod
    def from_disk(cls, data: dict[str, Any]) -> Self:
        """Parse from a TOML config dict (e.g. from tomllib)."""
        max_storage: Memory | None = None
        if "max_storage_gb" in data:
            max_storage = Memory.from_gb(float(data["max_storage_gb"]))  # pyright: ignore[reportAny]
        policy: StoragePolicy = data.get("storage_policy", "manual")  # pyright: ignore[reportAny]
        return cls(max_storage=max_storage, storage_policy=policy)

    def to_disk(self) -> dict[str, Any]:
        """Serialize to a dict suitable for writing to TOML."""
        result: dict[str, Any] = {}
        if self.max_storage is not None:
            result["max_storage_gb"] = round(self.max_storage.in_gb, 2)
        result["storage_policy"] = self.storage_policy
        return result


@final
class StorageAllow(FrozenModel):
    pass


@final
class StorageEvict(FrozenModel):
    model_ids: list[ModelId]


@final
class StorageReject(FrozenModel):
    reason: str
    available: Memory


StorageDecision = StorageAllow | StorageEvict | StorageReject
