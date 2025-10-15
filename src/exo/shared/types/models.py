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
