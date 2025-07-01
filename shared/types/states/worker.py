from collections.abc import Mapping
from typing import Tuple

from shared.types.models.common import ModelId
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeState
from shared.types.worker.downloads import BaseDownloadProgress, DownloadStatus
from shared.types.worker.shards import ShardData, ShardType


class WorkerState(SharedState):
    node_state: NodeState
    download_state: Mapping[
        Tuple[ModelId, ShardData[ShardType]], BaseDownloadProgress[DownloadStatus]
    ]
