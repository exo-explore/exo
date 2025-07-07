from typing import final, Sequence

from pydantic import BaseModel, TypeAdapter

from shared.types.models.common import ModelId
from shared.types.models.metadata import ModelMetadata
from shared.types.models.sources import ModelSource


@final
# Concerned by the naming here; model could also be an instance of a model.
class ModelInfo(BaseModel):
    model_id: ModelId
    model_sources: Sequence[ModelSource]
    model_metadata: ModelMetadata


ModelIdAdapter: TypeAdapter[ModelId] = TypeAdapter(ModelId)