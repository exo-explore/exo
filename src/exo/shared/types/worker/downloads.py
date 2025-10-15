from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class DownloadProgressData(CamelCaseModel):
    total_bytes: Memory
    downloaded_bytes: Memory


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId


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
