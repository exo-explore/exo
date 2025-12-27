from typing import Self

from pydantic import PositiveInt

from exo.shared.types.common import Id
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class ModelId(Id):
    pass


class ModelMetadata(CamelCaseModel):
    model_id: ModelId
    pretty_name: str
    storage_size: Memory
    n_layers: PositiveInt

    @classmethod
    def fixture(cls) -> Self:
        return cls(
            model_id=ModelId("llama-3.2-1b"),
            pretty_name="Llama 3.2 1B",
            n_layers=16,
            storage_size=Memory.from_bytes(678948),
        )
