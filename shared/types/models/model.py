from typing import Sequence, final

from pydantic import BaseModel, TypeAdapter

from shared.types.models.common import ModelId
from shared.types.models.metadata import ModelMetadata
from shared.types.models.sources import ModelSource


@final
class ModelInfo(BaseModel):
    model_id: ModelId
    model_sources: Sequence[ModelSource]
    model_metadata: ModelMetadata


ModelIdAdapter: TypeAdapter[ModelId] = TypeAdapter(ModelId)
