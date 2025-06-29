from typing import Annotated, Any, Generic, Literal, TypeVar, final
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter
from pydantic.types import UuidVersion

SourceType = Literal["HuggingFace", "GitHub"]

T = TypeVar("T", bound=SourceType)

_ModelId = Annotated[UUID, UuidVersion(4)]
ModelId = type("ModelId", (UUID,), {})
ModelIdParser: TypeAdapter[ModelId] = TypeAdapter(_ModelId)

RepoPath = Annotated[str, Field(pattern=r'^[^/]+/[^/]+$')]
RepoURL = Annotated[str, AnyHttpUrl]

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

RepoType = BaseModelSource[SourceType]

RepoValidatorThing = Annotated[
    RepoType,
    Field(discriminator="source_type")
]

RepoValidator: TypeAdapter[RepoValidatorThing] = TypeAdapter(RepoValidatorThing)
