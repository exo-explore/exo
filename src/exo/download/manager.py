from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
from anyio import current_time
from anyio.abc import TaskGroup
from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeDownloadProgress, TaskStatusUpdated
from exo.shared.types.models import ModelId
from exo.shared.types.tasks import DownloadModel, TaskStatus
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Sender
from exo.worker.download.download_utils import (
    map_repo_download_progress_to_download_progress_data,
)
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader

if TYPE_CHECKING:
    pass


@dataclass
class DownloadManager:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    _tg: TaskGroup | None = field(default=None, init=False)
    _event_sender: Sender[Event] | None = field(default=None, init=False)

    async def run(self, event_sender: Sender[Event]) -> None:
        """Run download manager background tasks."""
        self._event_sender = event_sender
        async with anyio.create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._emit_existing_download_progress)

    def shutdown(self) -> None:
        if self._tg:
            self._tg.cancel_scope.cancel()

    async def start_download(
        self,
        task: DownloadModel,
        event_sender: Sender[Event],
    ) -> None:
        """Initiate a download for the given task.

        This is called by Worker when it encounters a DownloadModel task.
        """
        shard = task.shard_metadata
        model_id = shard.model_meta.model_id

        if model_id not in self.download_status:
            progress = DownloadPending(shard_metadata=shard, node_id=self.node_id)
            self.download_status[model_id] = progress
            event_sender.send_nowait(NodeDownloadProgress(download_progress=progress))

        initial_progress = await self.shard_downloader.get_shard_download_status_for_shard(
            shard
        )

        if initial_progress.status == "complete":
            progress = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total_bytes=initial_progress.total_bytes,
            )
            self.download_status[model_id] = progress
            event_sender.send_nowait(NodeDownloadProgress(download_progress=progress))
            event_sender.send_nowait(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
        else:
            event_sender.send_nowait(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            self._handle_shard_download_process(task, initial_progress, event_sender)

    def _handle_shard_download_process(
        self,
        task: DownloadModel,
        initial_progress: RepoDownloadProgress,
        event_sender: Sender[Event],
    ) -> None:
        """Manages the shard download process with progress tracking."""
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=task.shard_metadata,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
        )
        self.download_status[task.shard_metadata.model_meta.model_id] = status
        event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        last_progress_time = 0.0
        throttle_interval_secs = 1.0

        def download_progress_callback(
            shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            nonlocal last_progress_time
            if progress.status == "complete":
                status = DownloadCompleted(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    total_bytes=progress.total_bytes,
                )
                self.download_status[shard.model_meta.model_id] = status
                event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                event_sender.send_nowait(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            elif (
                progress.status == "in_progress"
                and current_time() - last_progress_time > throttle_interval_secs
            ):
                status = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                )
                self.download_status[shard.model_meta.model_id] = status
                event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                last_progress_time = current_time()

        self.shard_downloader.on_progress(download_progress_callback)
        assert self._tg
        self._tg.start_soon(self.shard_downloader.ensure_shard, task.shard_metadata)

    async def _emit_existing_download_progress(self) -> None:
        """Periodically emit existing download progress."""
        try:
            while True:
                logger.info("Fetching and emitting existing download progress...")
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    if progress.status == "complete":
                        status: DownloadProgress = DownloadCompleted(
                            node_id=self.node_id,
                            shard_metadata=progress.shard,
                            total_bytes=progress.total_bytes,
                        )
                    elif progress.status in ["in_progress", "not_started"]:
                        if progress.downloaded_bytes_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id, shard_metadata=progress.shard
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_meta.model_id] = status
                    if self._event_sender:
                        await self._event_sender.send(
                            NodeDownloadProgress(download_progress=status)
                        )
                logger.info("Done emitting existing download progress.")
                await anyio.sleep(5 * 60)  # 5 minutes
        except Exception as e:
            logger.error(f"Error emitting existing download progress: {e}")
