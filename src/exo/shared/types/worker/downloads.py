from pydantic import BaseModel, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class DownloadProgressData(FrozenModel):
    total: Memory
    downloaded: Memory
    downloaded_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata
    model_directory: str = ""


class DownloadPending(BaseDownloadProgress):
    downloaded: Memory = Memory()
    total: Memory = Memory()


class DownloadCompleted(BaseDownloadProgress):
    total: Memory
    read_only: bool = False


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)


def is_not_on_disk(progress: DownloadProgress) -> bool:
    """Whether this says only that the model isn't on its node: Pending with no bytes.

    The state leaves these out, so a node lists only the models it has something of. Most
    of the catalogue is on no node, and these entries were 96% of a cluster's state.
    """
    return isinstance(progress, DownloadPending) and progress.downloaded.in_bytes == 0


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt | None = None


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]
