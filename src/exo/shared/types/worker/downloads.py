from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class DownloadProgressData(CamelCaseModel):
    total_bytes: Memory
    downloaded_bytes: Memory
    downloaded_bytes_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata


class DownloadPending(BaseDownloadProgress):
    pass


class DownloadCompleted(BaseDownloadProgress):
    total_bytes: Memory


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    speed: float
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    model_config = ConfigDict(frozen=True)


class RepoDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    shard: ShardMetadata
    completed_files: int
    total_files: int
    downloaded_bytes: Memory
    downloaded_bytes_this_session: Memory
    total_bytes: Memory
    overall_speed: float
    overall_eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    file_progress: dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
