from typing import Sequence, final

from pydantic import BaseModel

from shared.types.common import NewUUID
from shared.types.models.metadata import ModelMetadata
from shared.types.models.sources import ModelSource


class ModelId(NewUUID):
    pass


@final
class Model(BaseModel):
    model_id: ModelId
    model_sources: Sequence[ModelSource]
    model_metadata: ModelMetadata
