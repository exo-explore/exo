from typing import Annotated, Any, Generic, Literal, TypeVar, Union, final

from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter

from shared.types.models.common import ModelId

SourceType = Literal["HuggingFace", "GitHub"]

T = TypeVar("T", bound=SourceType)

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


_ModelSource = Annotated[
    Union[
        HuggingFaceModelSource,
        GitHubModelSource,
    ],
    Field(discriminator="source_type"),
]
ModelSource = BaseModelSource[SourceType]
ModelSourceAdapter: TypeAdapter[ModelSource] = TypeAdapter(_ModelSource)
