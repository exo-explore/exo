from typing import Annotated

from pydantic import BaseModel, PositiveInt


class ModelMetadata(BaseModel):
    storage_size_kilobytes: Annotated[int, PositiveInt]
    n_layers: Annotated[int, PositiveInt]
