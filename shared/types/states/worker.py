from typing import Tuple

from shared.types.models.common import ModelId
from shared.types.states.shared import SharedState
from shared.types.tasks.common import Task, TaskId, TaskType
from shared.types.worker.downloads import BaseDownloadProgress, DownloadStatus
from shared.types.worker.shards import ShardData, ShardType


class WorkerState(SharedState):
    download_state: dict[
        Tuple[ModelId, ShardData[ShardType]], BaseDownloadProgress[DownloadStatus]
    ]
    compute_tasks: dict[TaskId, Task[TaskType]]
