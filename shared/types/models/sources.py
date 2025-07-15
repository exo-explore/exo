from enum import Enum
from typing import Annotated, Any, Literal, Union, final

from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter

from shared.types.models.common import ModelId


@final
class SourceType(str, Enum):
    HuggingFace = "HuggingFace"
    GitHub = "GitHub"


@final
class SourceFormatType(str, Enum):
    HuggingFaceTransformers = "HuggingFaceTransformers"


RepoPath = Annotated[str, Field(pattern=r"^[^/]+/[^/]+$")]


class BaseModelSource[T: SourceType, S: SourceFormatType](BaseModel):
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
class GitHubModelSource(BaseModelSource[SourceType.GitHub, SourceFormatType]):
    source_type: Literal[SourceType.GitHub] = SourceType.GitHub
    source_format: SourceFormatType
    source_data: GitHubModelSourceData


_ModelSource = Annotated[
    Union[
        HuggingFaceModelSource,
        GitHubModelSource,
    ],
    Field(discriminator="source_type"),
]
ModelSource = BaseModelSource[SourceType, SourceFormatType]
ModelSourceAdapter: TypeAdapter[ModelSource] = TypeAdapter(_ModelSource)
