from enum import Enum
from typing import (
    Annotated,
    Callable,
    Literal,
    NewType,
    Sequence,
    Union,
)

from pydantic import BaseModel, Field, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.shards import ShardMetadata


class DownloadProgressData(BaseModel):
    total_bytes: Annotated[int, PositiveInt]
    downloaded_bytes: Annotated[int, PositiveInt]


class DownloadStatus(str, Enum):
    Pending = "Pending"
    Downloading = "Downloading"
    Completed = "Completed"
    Failed = "Failed"


class BaseDownloadProgress[DownloadStatusT: DownloadStatus](BaseModel):
    node_id: NodeId
    download_status: DownloadStatusT


class DownloadPending(BaseDownloadProgress[DownloadStatus.Pending]):
    download_status: Literal[DownloadStatus.Pending] = Field(default=DownloadStatus.Pending)


class DownloadCompleted(BaseDownloadProgress[DownloadStatus.Completed]):
    download_status: Literal[DownloadStatus.Completed] = Field(default=DownloadStatus.Completed)


class DownloadFailed(BaseDownloadProgress[DownloadStatus.Failed]):
    download_status: Literal[DownloadStatus.Failed] = Field(default=DownloadStatus.Failed)
    error_message: str


class DownloadOngoing(BaseDownloadProgress[DownloadStatus.Downloading]):
    download_status: Literal[DownloadStatus.Downloading] = Field(
        default=DownloadStatus.Downloading
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
    shard_metadata: ShardMetadata,
    effect_handlers: Sequence[DownloadEffectHandler],
) -> None: ...
