import asyncio
from dataclasses import dataclass, field
from typing import Iterator

import anyio
from anyio import current_time
from anyio.abc import TaskGroup
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    map_repo_download_progress_to_download_progress_data,
)
from exo.download.model_store_server import ModelStoreServer
from exo.download.node_address_book import NodeAddressBook
from exo.download.shard_downloader import ShardDownloader
from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    Event,
    ForwarderEvent,
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
    connection_message_receiver: Receiver[ConnectionMessage]
    node_address_book: NodeAddressBook
    model_store_port: int
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    local_event_sender: Sender[ForwarderEvent]
    event_index_counter: Iterator[int]

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, asyncio.Task[None]] = field(default_factory=dict)

    # Internal event channel for forwarding (initialized in __post_init__)
    event_sender: Sender[Event] = field(init=False)
    event_receiver: Receiver[Event] = field(init=False)
    _tg: TaskGroup = field(init=False)

    def __post_init__(self) -> None:
        self.event_sender, self.event_receiver = channel[Event]()
        self._tg = anyio.create_task_group()

    async def run(self) -> None:
        logger.info("Starting DownloadCoordinator")
        async with self._tg as tg:
            tg.start_soon(self._command_processor)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._emit_existing_download_progress)
            tg.start_soon(self._track_node_addresses)

            if self.node_id == self.session_id.master_node_id:
                tg.start_soon(ModelStoreServer(port=self.model_store_port).run)

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

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id

        # Check if already downloading or complete
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadCompleted)):
                logger.debug(
                    f"Download for {model_id} already in progress or complete, skipping"
                )
                return

        # Emit pending status
        progress = DownloadPending(shard_metadata=shard, node_id=self.node_id)
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
                total_bytes=initial_progress.total_bytes,
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
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
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        last_progress_time = 0.0
        throttle_interval_secs = 1.0

        async def download_progress_callback(
            callback_shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            nonlocal last_progress_time

            if progress.status == "complete":
                completed = DownloadCompleted(
                    shard_metadata=callback_shard,
                    node_id=self.node_id,
                    total_bytes=progress.total_bytes,
                )
                self.download_status[callback_shard.model_card.model_id] = completed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=completed)
                )
                # Clean up active download tracking
                if callback_shard.model_card.model_id in self.active_downloads:
                    del self.active_downloads[callback_shard.model_card.model_id]
            elif (
                progress.status == "in_progress"
                and current_time() - last_progress_time > throttle_interval_secs
            ):
                ongoing = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=callback_shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                )
                self.download_status[callback_shard.model_card.model_id] = ongoing
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=ongoing)
                )
                last_progress_time = current_time()

        self.shard_downloader.on_progress(download_progress_callback)

        async def download_wrapper() -> None:
            try:
                await self.shard_downloader.ensure_shard(shard)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
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
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _track_node_addresses(self) -> None:
        with self.connection_message_receiver as messages:
            async for msg in messages:
                match msg.connection_type:
                    case ConnectionMessageType.Connected:
                        self.node_address_book.set_ipv4(msg.node_id, msg.remote_ipv4)
                    case ConnectionMessageType.Disconnected:
                        self.node_address_book.remove(msg.node_id)

    async def _forward_events(self) -> None:
        with self.event_receiver as events:
            async for event in events:
                idx = next(self.event_index_counter)
                fe = ForwarderEvent(
                    origin_idx=idx,
                    origin=self.node_id,
                    session=self.session_id,
                    event=event,
                )
                logger.debug(
                    f"DownloadCoordinator published event {idx}: {str(event)[:100]}"
                )
                await self.local_event_sender.send(fe)

    async def _emit_existing_download_progress(self) -> None:
        try:
            while True:
                logger.info(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
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

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                logger.info(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
                await anyio.sleep(5 * 60)  # 5 minutes
        except Exception as e:
            logger.error(
                f"DownloadCoordinator: Error emitting existing download progress: {e}"
            )
