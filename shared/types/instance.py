from typing import Annotated, Literal, Generic, TypeVar, Union
from enum import Enum
from pydantic import BaseModel, UUID4, PositiveInt, Field

from shared.types.model import ModelId

InstanceId = Annotated[str, UUID4]
NodeId = Annotated[str, UUID4]
RunnerId = Annotated[str, UUID4]


class ShardType(str, Enum):
    PipelineParallel = "PipelineParallel"


ShardTypeT = TypeVar("ShardTypeT", bound=ShardType)


class ShardData(BaseModel, Generic[ShardTypeT]):
    shard_type: ShardTypeT


class Shard(BaseModel, Generic[ShardTypeT]):
    shard_data: ShardData[ShardTypeT]
    runner_id: RunnerId


class ShardPlacement(BaseModel):
    model_id: ModelId
    shard_assignments: dict[NodeId, Shard[ShardType]]


class DownloadProgressData(BaseModel):
    total_bytes: Annotated[int, PositiveInt]
    downloaded_bytes: Annotated[int, PositiveInt]


class DownloadStatus(str, Enum):
    Pending = "Pending"
    Downloading = "Downloading"
    Completed = "Completed"
    Failed = "Failed"


DownloadStatusT = TypeVar("DownloadStatusT", bound=DownloadStatus)


class BaseDownloadProgress(BaseModel, Generic[DownloadStatusT]):
    node_id: NodeId
    download_status: DownloadStatusT


class DownloadPending(BaseDownloadProgress[DownloadStatus.Pending]):
    download_status: Literal[DownloadStatus.Pending] = Field(DownloadStatus.Pending)


class DownloadCompleted(BaseDownloadProgress[DownloadStatus.Completed]):
    download_status: Literal[DownloadStatus.Completed] = Field(DownloadStatus.Completed)


class DownloadFailed(BaseDownloadProgress[DownloadStatus.Failed]):
    download_status: Literal[DownloadStatus.Failed] = Field(DownloadStatus.Failed)
    error_message: str


class DownloadOngoing(BaseDownloadProgress[DownloadStatus.Downloading]):
    download_status: Literal[DownloadStatus.Downloading] = Field(
        DownloadStatus.Downloading
    )
    download_progress: DownloadProgressData


DownloadProgress = Annotated[
    Union[
        DownloadPending,
        DownloadCompleted,
        DownloadFailed,
        DownloadOngoing,
    ],
    Field(discriminator="download_status"),
]


class Instance(ShardPlacement):
    instance_id: InstanceId


class InstanceDownloadProgress(BaseModel, Generic[DownloadStatusT]):
    instance_id: InstanceId
    download_progress: BaseDownloadProgress[DownloadStatusT]
