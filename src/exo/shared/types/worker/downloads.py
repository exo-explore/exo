from enum import Enum
from typing import (
    Annotated,
    Literal,
    Union,
)
from pydantic import Field

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class DownloadProgressData(CamelCaseModel):
    total_bytes: Memory
    downloaded_bytes: Memory
    downloaded_bytes_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]

class DownloadStatus(str, Enum):
    Pending = "Pending"
    Downloading = "Downloading"
    Completed = "Completed"
    Failed = "Failed"


class BaseDownloadProgress[DownloadStatusT: DownloadStatus](CamelCaseModel):
    node_id: NodeId
    download_status: DownloadStatusT


class DownloadPending(BaseDownloadProgress[DownloadStatus.Pending]):
    download_status: Literal[DownloadStatus.Pending] = Field(
        default=DownloadStatus.Pending
    )


class DownloadCompleted(BaseDownloadProgress[DownloadStatus.Completed]):
    download_status: Literal[DownloadStatus.Completed] = Field(
        default=DownloadStatus.Completed
    )


class DownloadFailed(BaseDownloadProgress[DownloadStatus.Failed]):
    download_status: Literal[DownloadStatus.Failed] = Field(
        default=DownloadStatus.Failed
    )
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
