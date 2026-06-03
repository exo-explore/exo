from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, current_time, to_thread
from exo_rs import LVPublisher, Mailbox, SessionHandle
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    is_read_only_model_dir,
    map_repo_download_progress_to_download_progress_data,
    resolve_existing_model,
)
from exo.download.shard_downloader import ShardDownloader
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.constants import EXO_DEFAULT_MODELS_DIR, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    DownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Sender
from exo.utils.task_group import TaskGroup

_DOWNLOAD_COMMAND_ADAPTER = TypeAdapter[DownloadCommand](DownloadCommand)


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    event_sender: Sender[Event]
    session_handle: SessionHandle
    offline: bool = False
    mailbox: Mailbox | None = None

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)
    _mailbox: Mailbox = field(init=False)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)
    _download_publishers: dict[ModelId, LVPublisher] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)
        self._mailbox = self.mailbox or self.session_handle.mailbox(self.node_id)

    @staticmethod
    def _default_model_dir(model_id: ModelId) -> str:
        return str(EXO_DEFAULT_MODELS_DIR / model_id.normalize())

    def _completed_from_path(
        self,
        shard: ShardMetadata,
        found: Path,
        total: Memory,
    ) -> DownloadCompleted:
        return DownloadCompleted(
            shard_metadata=shard,
            node_id=self.node_id,
            total=total,
            model_directory=str(found),
            read_only=is_read_only_model_dir(found),
        )

    def _publisher_for_model(self, model_id: ModelId) -> LVPublisher:
        if (publisher := self._download_publishers.get(model_id)) is None:
            publisher = self.session_handle.last_value_publisher(
                f"node_metrics/{self.node_id}/downloads/{model_id}"
            )
            self._download_publishers[model_id] = publisher

        return publisher

    async def _set_download_status(self, status: DownloadProgress) -> None:
        model_id = status.shard_metadata.model_card.model_id
        self.download_status[model_id] = status
        publisher = self._publisher_for_model(model_id)
        await publisher.put(status.model_dump_json())

    async def _delete_download_status(self, model_id: ModelId) -> None:
        self.download_status.pop(model_id, None)
        publisher = self._publisher_for_model(model_id)
        await publisher.delete()

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        try:
            if progress.status == "complete":
                found = await to_thread.run_sync(
                    resolve_existing_model, model_id, callback_shard.model_card
                )
                if found is not None:
                    completed = self._completed_from_path(
                        callback_shard, found, progress.total
                    )
                else:
                    completed = DownloadCompleted(
                        shard_metadata=callback_shard,
                        node_id=self.node_id,
                        total=progress.total,
                        model_directory=self._default_model_dir(model_id),
                    )
                await self._set_download_status(completed)
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
                    model_directory=self._default_model_dir(model_id),
                )
                await self._set_download_status(ongoing)
                self._last_progress_time[model_id] = current_time()
        except (BrokenResourceError, ClosedResourceError):
            logger.debug(
                f"Event stream closed while sending download progress for {model_id}, skipping update"
            )

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # don't forget to clean up resources
            self.event_sender.close()

            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        while (payload := await self._mailbox.recv()) is not None:
            try:
                command = _DOWNLOAD_COMMAND_ADAPTER.validate_json(payload)
            except ValidationError:
                continue

            if command.target_node_id != self.node_id:
                continue

            match command:
                case StartDownload(shard_metadata=shard):
                    await self._start_download(shard)
                case DeleteDownload(model_id=model_id):
                    await self._delete_download(model_id)
                case CancelDownload(model_id=model_id):
                    await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            current_status = self.download_status[model_id]
            downloaded = Memory()
            total = Memory()
            if isinstance(current_status, DownloadOngoing):
                downloaded = current_status.download_progress.downloaded
                total = current_status.download_progress.total
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
                downloaded=downloaded,
                total=total,
            )
            await self._set_download_status(pending)

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

        # Check all model directories for pre-existing complete models
        found_path = await to_thread.run_sync(
            resolve_existing_model, model_id, shard.model_card
        )
        if found_path is not None:
            logger.info(f"DownloadCoordinator: Model {model_id} found at {found_path}")
            completed = self._completed_from_path(
                shard, found_path, shard.model_card.storage_size
            )
            await self._set_download_status(completed)
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._default_model_dir(model_id),
        )
        await self._set_download_status(progress)

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            found = await to_thread.run_sync(
                resolve_existing_model, model_id, shard.model_card
            )
            if found is not None:
                completed = self._completed_from_path(
                    shard, found, initial_progress.total
                )
            else:
                completed = DownloadCompleted(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    total=initial_progress.total,
                    model_directory=self._default_model_dir(model_id),
                )
            await self._set_download_status(completed)
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._default_model_dir(model_id),
            )
            await self._set_download_status(failed)
            return

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._default_model_dir(model_id),
        )
        await self._set_download_status(status)

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                with cancel_scope:
                    await self.shard_downloader.ensure_shard(shard)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._default_model_dir(model_id),
                )
                await self._set_download_status(failed)
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _delete_download(self, model_id: ModelId) -> None:
        # Protect read-only models from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(f"Refusing to delete read-only model {model_id}")
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Remove local and aggregated download status.
        if model_id in self.download_status:
            await self._delete_download_status(model_id)

    async def _emit_existing_download_progress(self) -> None:
        while True:
            try:
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
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status: DownloadProgress = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        else:
                            status = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                total=progress.total,
                                model_directory=self._default_model_dir(model_id),
                            )
                    elif progress.status in ["in_progress", "not_started"]:
                        # TODO(ciaran): temporary solution
                        # Don't downgrade a model that is already confirmed complete.
                        if isinstance(
                            self.download_status.get(model_id), DownloadCompleted
                        ):
                            continue
                        # The per-file size check compares local files against
                        # the latest HF "main" revision, which is a moving
                        # target.  When HF updates text files (README, YAML,
                        # jinja) in a new commit, the cached file list has new
                        # sizes while local files still match the old revision.
                        # Fall back to the authoritative completeness check
                        # (is_model_directory_complete) which validates that all
                        # safetensors weight files are present.
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        elif progress.downloaded_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._default_model_dir(model_id),
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
                                model_directory=self._default_model_dir(model_id),
                            )
                    else:
                        continue

                    await self._set_download_status(status)
                # Scan read-only directories for pre-downloaded models
                if EXO_MODELS_READ_ONLY_DIRS:
                    for card in await model_cards.card_cache.list_all():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = await to_thread.run_sync(
                            resolve_existing_model, mid, card
                        )
                        if found is not None and is_read_only_model_dir(found):
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = (
                                self._completed_from_path(
                                    path_shard, found, card.storage_size
                                )
                            )
                            await self._set_download_status(path_completed)

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
