from typing import Annotated

from pydantic import BaseModel, PositiveInt


class ModelMetadata(BaseModel):
    pretty_name: str
    storage_size_kilobytes: Annotated[int, PositiveInt]
    n_layers: Annotated[int, PositiveInt]
