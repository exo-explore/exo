from typing import Annotated, TypeAlias

from pydantic import BaseModel, PositiveInt

ModelId: TypeAlias = str


class ModelMetadata(BaseModel):
    model_id: ModelId
    pretty_name: str
    storage_size_kilobytes: Annotated[int, PositiveInt]
    n_layers: Annotated[int, PositiveInt]
