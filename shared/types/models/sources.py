from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, Union, final

from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter

from shared.types.models.common import ModelId


class SourceType(str, Enum):
    HuggingFace = "HuggingFace"
    GitHub = "GitHub"


class SourceFormatType(str, Enum):
    HuggingFaceTransformers = "HuggingFaceTransformers"


T = TypeVar("T", bound=SourceType)
S = TypeVar("S", bound=SourceFormatType)

RepoPath = Annotated[str, Field(pattern=r"^[^/]+/[^/]+$")]


class BaseModelSource(BaseModel, Generic[T, S]):
    model_uuid: ModelId
    source_type: T
    source_format: S
    source_data: Any


@final
class HuggingFaceModelSourceData(BaseModel):
    path: RepoPath


@final
class GitHubModelSourceData(BaseModel):
    url: AnyHttpUrl


@final
class HuggingFaceModelSource(
    BaseModelSource[SourceType.HuggingFace, SourceFormatType.HuggingFaceTransformers]
):
    source_type: Literal[SourceType.HuggingFace] = SourceType.HuggingFace
    source_format: Literal[SourceFormatType.HuggingFaceTransformers] = (
        SourceFormatType.HuggingFaceTransformers
    )
    source_data: HuggingFaceModelSourceData


@final
class GitHubModelSource(BaseModelSource[SourceType.GitHub, S]):
    source_type: Literal[SourceType.GitHub] = SourceType.GitHub
    source_data: GitHubModelSourceData


_ModelSource = Annotated[
    Union[
        HuggingFaceModelSource,
        GitHubModelSource[SourceFormatType.HuggingFaceTransformers],
    ],
    Field(discriminator="source_type"),
]
ModelSource = BaseModelSource[SourceType, SourceFormatType]
ModelSourceAdapter: TypeAdapter[ModelSource] = TypeAdapter(_ModelSource)
