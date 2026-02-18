import asyncio
import socket
from dataclasses import dataclass, field

import anyio
from anyio import current_time
from anyio.abc import TaskGroup
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    map_repo_download_progress_to_download_progress_data,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    LocalForwarderEvent,
    NodeDownloadProgress,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Receiver, Sender, channel


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    session_id: SessionId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    local_event_sender: Sender[LocalForwarderEvent]
    offline: bool = False
    _system_id: SystemId = field(default_factory=SystemId)

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, asyncio.Task[None]] = field(default_factory=dict)

    # Internal event channel for forwarding (initialized in __post_init__)
    event_sender: Sender[Event] = field(init=False)
    event_receiver: Receiver[Event] = field(init=False)
    _tg: TaskGroup = field(init=False, default_factory=anyio.create_task_group)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_sender, self.event_receiver = channel[Event]()
        if self.offline:
            self.shard_downloader.set_internet_connection(False)
        self.shard_downloader.on_progress(self._download_progress_callback)

    def _model_dir(self, model_id: ModelId) -> str:
        return str(EXO_MODELS_DIR / model_id.normalize())

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        if progress.status == "complete":
            completed = DownloadCompleted(
                shard_metadata=callback_shard,
                node_id=self.node_id,
                total=progress.total,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            if model_id in self.active_downloads:
                del self.active_downloads[model_id]
            self._last_progress_time.pop(model_id, None)
        elif (
            progress.status == "in_progress"
            and current_time() - self._last_progress_time.get(model_id, 0.0)
            > throttle_interval_secs
        ):
            ongoing = DownloadOngoing(
                node_id=self.node_id,
                shard_metadata=callback_shard,
                download_progress=map_repo_download_progress_to_download_progress_data(
                    progress
                ),
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = ongoing
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=ongoing)
            )
            self._last_progress_time[model_id] = current_time()

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        if not self.offline:
            self._test_internet_connection()
        async with self._tg as tg:
            tg.start_soon(self._command_processor)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._emit_existing_download_progress)
            if not self.offline:
                tg.start_soon(self._check_internet_connection)

    def _test_internet_connection(self) -> None:
        # Try multiple endpoints since some ISPs/networks block specific IPs
        for host in ("1.1.1.1", "8.8.8.8", "1.0.0.1"):
            try:
                socket.create_connection((host, 443), timeout=3).close()
                self.shard_downloader.set_internet_connection(True)
                logger.debug(f"Internet connectivity: True (via {host})")
                return
            except OSError:
                continue
        self.shard_downloader.set_internet_connection(False)
        logger.debug("Internet connectivity: False")

    async def _check_internet_connection(self) -> None:
        first_connection = True
        while True:
            await asyncio.sleep(10)

            # Assume that internet connection is set to False on 443 errors.
            if self.shard_downloader.internet_connection:
                continue

            self._test_internet_connection()

            if first_connection and self.shard_downloader.internet_connection:
                first_connection = False
                self._tg.start_soon(self._emit_existing_download_progress)

    def shutdown(self) -> None:
        self._tg.cancel_scope.cancel()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Only process commands targeting this node
                if cmd.command.target_node_id != self.node_id:
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard):
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id):
                        await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads.pop(model_id).cancel()

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id

        # Check if already downloading, complete, or recently failed
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadCompleted, DownloadFailed)):
                logger.debug(
                    f"Download for {model_id} already in progress, complete, or failed, skipping"
                )
                return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._model_dir(model_id),
        )
        self.download_status[model_id] = progress
        await self.event_sender.send(NodeDownloadProgress(download_progress=progress))

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=initial_progress.total,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return

        # Start actual download
        self._start_download_task(shard, initial_progress)

    def _start_download_task(
        self, shard: ShardMetadata, initial_progress: RepoDownloadProgress
    ) -> None:
        model_id = shard.model_card.model_id

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._model_dir(model_id),
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        async def download_wrapper() -> None:
            try:
                await self.shard_downloader.ensure_shard(shard)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            finally:
                if model_id in self.active_downloads:
                    del self.active_downloads[model_id]

        task = asyncio.create_task(download_wrapper())
        self.active_downloads[model_id] = task

    async def _delete_download(self, model_id: ModelId) -> None:
        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()
            del self.active_downloads[model_id]

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _forward_events(self) -> None:
        idx = 0
        with self.event_receiver as events:
            async for event in events:
                fe = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=self._system_id,
                    session=self.session_id,
                    event=event,
                )
                idx += 1
                logger.debug(
                    f"DownloadCoordinator published event {idx}: {str(event)[:100]}"
                )
                await self.local_event_sender.send(fe)

    async def _emit_existing_download_progress(self) -> None:
        try:
            while True:
                logger.debug(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    model_id = progress.shard.model_card.model_id

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if progress.status == "complete":
                        status: DownloadProgress = DownloadCompleted(
                            node_id=self.node_id,
                            shard_metadata=progress.shard,
                            total=progress.total,
                            model_directory=self._model_dir(
                                progress.shard.model_card.model_id
                            ),
                        )
                    elif progress.status in ["in_progress", "not_started"]:
                        if progress.downloaded_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._model_dir(
                                    progress.shard.model_card.model_id
                                ),
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                                model_directory=self._model_dir(
                                    progress.shard.model_card.model_id
                                ),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
                await anyio.sleep(60)
        except Exception as e:
            logger.error(
                f"DownloadCoordinator: Error emitting existing download progress: {e}"
            )
