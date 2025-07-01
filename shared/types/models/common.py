from typing import Annotated, Sequence, final
from uuid import UUID

from pydantic import BaseModel, TypeAdapter
from pydantic.types import UuidVersion

from shared.types.models.metadata import ModelMetadata
from shared.types.models.sources import ModelSource

_ModelId = Annotated[UUID, UuidVersion(4)]
ModelId = type("ModelId", (UUID,), {})
ModelIdParser: TypeAdapter[ModelId] = TypeAdapter(_ModelId)


@final
class Model(BaseModel):
    model_id: ModelId
    model_sources: Sequence[ModelSource]
    model_metadata: ModelMetadata


ModelIdAdapter: TypeAdapter[ModelId] = TypeAdapter(_ModelId)
