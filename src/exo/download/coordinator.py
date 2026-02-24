import asyncio
from dataclasses import dataclass, field
from random import random

import anyio
from anyio import current_time
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    map_repo_download_progress_to_download_progress_data,
    resolve_model_in_path,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.constants import EXO_MODELS_DIR, EXO_MODELS_PATH
from exo.shared.models.model_cards import ModelId, get_model_cards
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    EventId,
    # TODO(evan): just for acks, should delete this ASAP
    GlobalForwarderEvent,
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
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    session_id: SessionId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    local_event_sender: Sender[LocalForwarderEvent]

    # ack stuff
    _global_event_receiver: Receiver[GlobalForwarderEvent]
    _out_for_delivery: dict[EventId, LocalForwarderEvent] = field(default_factory=dict)

    offline: bool = False

    _system_id: SystemId = field(default_factory=SystemId)

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, asyncio.Task[None]] = field(default_factory=dict)

    # Internal event channel for forwarding (initialized in __post_init__)
    event_sender: Sender[Event] = field(init=False)
    event_receiver: Receiver[Event] = field(init=False)
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_sender, self.event_receiver = channel[Event]()
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
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._forward_events)
                tg.start_soon(self._emit_existing_download_progress)
                tg.start_soon(self._resend_out_for_delivery)
                tg.start_soon(self._clear_ofd)
        finally:
            for task in self.active_downloads.values():
                task.cancel()

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    # directly copied from worker
    async def _resend_out_for_delivery(self) -> None:
        # This can also be massively tightened, we should check events are at least a certain age before resending.
        # Exponential backoff would also certainly help here.
        while True:
            await anyio.sleep(1 + random())
            for event in self._out_for_delivery.copy().values():
                await self.local_event_sender.send(event)

    async def _clear_ofd(self) -> None:
        with self._global_event_receiver as events:
            async for event in events:
                self._out_for_delivery.pop(event.event.event_id, None)

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

        # Check EXO_MODELS_PATH for pre-downloaded models
        found_path = resolve_model_in_path(model_id)
        if found_path is not None:
            logger.info(
                f"DownloadCoordinator: Model {model_id} found in EXO_MODELS_PATH at {found_path}"
            )
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=shard.model_card.storage_size,
                model_directory=str(found_path),
                read_only=True,
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
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
        # Protect read-only models (from EXO_MODELS_PATH) from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(
                    f"Refusing to delete read-only model {model_id} (from EXO_MODELS_PATH)"
                )
                return

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
                self._out_for_delivery[event.event_id] = fe

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
                                downloaded=progress.downloaded,
                                total=progress.total,
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
                # Scan EXO_MODELS_PATH for pre-downloaded models
                if EXO_MODELS_PATH is not None:
                    for card in await get_model_cards():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = resolve_model_in_path(mid)
                        if found is not None:
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=path_shard,
                                total=card.storage_size,
                                model_directory=str(found),
                                read_only=True,
                            )
                            self.download_status[mid] = path_completed
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=path_completed)
                            )

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
                await anyio.sleep(60)
        except Exception as e:
            logger.error(
                f"DownloadCoordinator: Error emitting existing download progress: {e}"
            )
