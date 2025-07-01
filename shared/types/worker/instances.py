from typing import Generic, TypeVar

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.downloads import BaseDownloadProgress, DownloadStatus
from shared.types.worker.shards import ShardPlacement

DownloadStatusT = TypeVar("DownloadStatusT", bound=DownloadStatus)


class Instance(ShardPlacement):
    instance_id: InstanceId


class InstanceDownloadProgress(BaseModel, Generic[DownloadStatusT]):
    instance_id: InstanceId
    download_progress: BaseDownloadProgress[DownloadStatusT]
