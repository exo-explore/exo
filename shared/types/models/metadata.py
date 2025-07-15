from typing import Annotated, final

from pydantic import BaseModel, PositiveInt


@final
class ModelMetadata(BaseModel):
    pretty_name: str
    storage_size_kilobytes: Annotated[int, PositiveInt]
    n_layers: Annotated[int, PositiveInt]
