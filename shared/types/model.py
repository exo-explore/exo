from typing import Annotated, Any, Generic, Literal, Sequence, TypeVar, Union, final
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field, PositiveInt, TypeAdapter
from pydantic.types import UuidVersion

SourceType = Literal["HuggingFace", "GitHub"]

T = TypeVar("T", bound=SourceType)

_ModelId = Annotated[UUID, UuidVersion(4)]
ModelId = type("ModelId", (UUID,), {})
ModelIdParser: TypeAdapter[ModelId] = TypeAdapter(_ModelId)

RepoPath = Annotated[str, Field(pattern=r"^[^/]+/[^/]+$")]


class BaseModelSource(BaseModel, Generic[T]):
    model_uuid: ModelId
    source_type: T
    source_data: Any


@final
class HuggingFaceModelSourceData(BaseModel):
    path: RepoPath


@final
class GitHubModelSourceData(BaseModel):
    url: AnyHttpUrl


@final
class HuggingFaceModelSource(BaseModelSource[Literal["HuggingFace"]]):
    source_type: Literal["HuggingFace"] = "HuggingFace"
    source_data: HuggingFaceModelSourceData


@final
class GitHubModelSource(BaseModelSource[Literal["GitHub"]]):
    source_type: Literal["GitHub"] = "GitHub"
    source_data: GitHubModelSourceData


class ModelMetadata(BaseModel):
    storage_size_kilobytes: Annotated[int, PositiveInt]
    n_layers: Annotated[int, PositiveInt]


_ModelSource = Annotated[
    Union[
        HuggingFaceModelSource,
        GitHubModelSource,
    ],
    Field(discriminator="source_type"),
]
ModelSource = BaseModelSource[SourceType]


@final
class Model(BaseModel):
    model_id: ModelId
    model_sources: Sequence[ModelSource]
    model_metadata: ModelMetadata


ModelIdAdapter: TypeAdapter[ModelId] = TypeAdapter(_ModelId)
ModelSourceAdapter: TypeAdapter[ModelSource] = TypeAdapter(_ModelSource)
