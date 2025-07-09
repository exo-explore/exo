from enum import Enum
from typing import (
    Annotated,
    Callable,
    Generic,
    Literal,
    NewType,
    Sequence,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, PositiveInt

from shared.types.common import NodeId
from shared.types.models.common import ModelId
from shared.types.models.sources import ModelSource
from shared.types.worker.shards import PartitionStrategy, ShardMetadata


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


BytesToDownload = NewType("BytesToDownload", int)
BytesDownloaded = NewType("BytesDownloaded", int)

DownloadEffectHandler = Callable[
    [ModelId, DownloadStatus, BytesToDownload, BytesDownloaded], None
]


def download_shard(
    model_id: ModelId,
    model_source: ModelSource,
    shard_meta: ShardMetadata[PartitionStrategy],
    effect_handlers: Sequence[DownloadEffectHandler],
) -> None: ...
