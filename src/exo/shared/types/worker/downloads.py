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
    pass


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)
