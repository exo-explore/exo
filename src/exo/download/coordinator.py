import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import aiofiles
import aiofiles.os as aios
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
from exo.shared.constants import (
    EXO_MODEL_USAGE_FILE,
    EXO_MODELS_DIR,
    EXO_MODELS_PATH,
)
from exo.shared.models.model_cards import ModelId, get_model_cards
from exo.shared.storage import (
    calculate_used_storage,
    decide_storage_action,
    persist_storage_config,
)
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeDownloadProgress,
    StorageConfigUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.storage import (
    StorageAllow,
    StorageConfig,
    StorageEvict,
    StorageReject,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadEvicted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadRejected,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_receiver: Receiver[IndexedEvent]
    event_sender: Sender[Event]
    offline: bool = False
    storage_config: StorageConfig = field(default_factory=StorageConfig)

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, asyncio.Task[None]] = field(default_factory=dict)

    _model_last_used: dict[ModelId, datetime] = field(default_factory=dict)
    _active_model_ids: set[ModelId] = field(default_factory=set)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
        await self._load_model_usage()
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
                tg.start_soon(self._event_watcher)
        finally:
            for task in self.active_downloads.values():
                task.cancel()

    async def _event_watcher(self) -> None:
        active_instances: dict[str, ModelId] = {}
        with self.event_receiver as events:
            async for indexed_event in events:
                match indexed_event.event:
                    case StorageConfigUpdated(node_id=node_id) if (
                        node_id == self.node_id
                    ):
                        self.storage_config = indexed_event.event.storage_config
                        await self.clear_rejections()
                        await persist_storage_config(indexed_event.event.storage_config)
                    case InstanceCreated(instance=instance):
                        active_instances[instance.instance_id] = (
                            instance.shard_assignments.model_id
                        )
                        await self.update_active_models(set(active_instances.values()))
                    case InstanceDeleted(instance_id=instance_id):
                        if instance_id in active_instances:
                            del active_instances[instance_id]
                            await self.update_active_models(
                                set(active_instances.values())
                            )
                    case _:
                        pass

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

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
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )

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
            if isinstance(status, DownloadRejected):
                del self.download_status[model_id]

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

        action = decide_storage_action(
            shard.model_card.storage_size,
            self.storage_config,
            list(self.download_status.values()),
            self._model_last_used,
            frozenset(self._active_model_ids),
        )
        match action:
            case StorageReject(reason=reason, available=available):
                await self._reject_download(shard, reason, available)
                return
            case StorageEvict(model_ids=model_ids):
                if not await self._execute_evictions(model_ids, shard):
                    return
            case StorageAllow():
                pass

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

    async def _remove_model_from_disk(self, model_id: ModelId) -> bool:
        # Protect read-only models (from EXO_MODELS_PATH) from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(
                    f"Refusing to delete read-only model {model_id} (from EXO_MODELS_PATH)"
                )
                return False

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()
            del self.active_downloads[model_id]

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)

        if not deleted:
            logger.warning(f"Failed to delete model {model_id} from disk")
            return False

        logger.info(f"Successfully deleted model {model_id}")
        return True

    async def _delete_download(self, model_id: ModelId) -> bool:
        success = await self._remove_model_from_disk(model_id)
        if not success:
            return False

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

        return True

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

                    # Don't overwrite DownloadEvicted — the model may still be on disk
                    # while deletion is finishing
                    if isinstance(self.download_status.get(model_id), DownloadEvicted):
                        continue

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if isinstance(self.download_status.get(model_id), DownloadRejected):
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
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)

    async def _reject_download(
        self, shard: ShardMetadata, reason: str, available: Memory
    ) -> None:
        assert self.storage_config.max_storage is not None
        model_id = shard.model_card.model_id
        rejected = DownloadRejected(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._model_dir(model_id),
            reason=reason,
            required=shard.model_card.storage_size,
            available=available if available.in_bytes > 0 else Memory(),
            limit=self.storage_config.max_storage,
        )
        self.download_status[model_id] = rejected
        await self.event_sender.send(NodeDownloadProgress(download_progress=rejected))

    async def _execute_evictions(
        self, model_ids: list[ModelId], shard: ShardMetadata
    ) -> bool:
        """Execute disk deletions for the given model IDs. Returns False on failure."""
        target_model_id = shard.model_card.model_id

        for evict_model_id in model_ids:
            logger.info(
                f"Auto-evicting model {evict_model_id} to free space for {target_model_id}"
            )
            evicted_status = self.download_status.get(evict_model_id)
            success = await self._remove_model_from_disk(evict_model_id)
            if not success:
                current_used = calculate_used_storage(
                    list(self.download_status.values())
                )
                assert self.storage_config.max_storage is not None
                current_available = self.storage_config.max_storage - current_used
                await self._reject_download(
                    shard,
                    f"Failed to delete model {evict_model_id} from disk",
                    current_available,
                )
                return False

            if evicted_status is not None:
                evicted = DownloadEvicted(
                    shard_metadata=evicted_status.shard_metadata,
                    node_id=self.node_id,
                    model_directory=self._model_dir(evict_model_id),
                    evicted_for=target_model_id,
                )
                self.download_status[evict_model_id] = evicted
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=evicted)
                )

        return True

    async def clear_rejections(self) -> None:
        rejected = [
            (model_id, status)
            for model_id, status in self.download_status.items()
            if isinstance(status, DownloadRejected)
        ]
        for model_id, status in rejected:
            logger.info(
                f"Clearing DownloadRejected for {model_id} after storage config change"
            )
            pending = DownloadPending(
                shard_metadata=status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )

    async def update_active_models(self, active_model_ids: set[ModelId]) -> None:
        new_models = active_model_ids - self._active_model_ids
        for mid in new_models:
            self._model_last_used[mid] = datetime.now(UTC)
        self._active_model_ids = active_model_ids.copy()
        if new_models:
            await self._persist_model_usage()

    async def _persist_model_usage(self) -> None:
        try:
            await aios.makedirs(EXO_MODEL_USAGE_FILE.parent, exist_ok=True)
            data = {mid: ts.isoformat() for mid, ts in self._model_last_used.items()}
            async with aiofiles.open(EXO_MODEL_USAGE_FILE, "w") as f:
                await f.write(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to persist model usage: {e}")

    async def _load_model_usage(self) -> None:
        try:
            if await aios.path.exists(EXO_MODEL_USAGE_FILE):
                async with aiofiles.open(EXO_MODEL_USAGE_FILE, "r") as f:
                    raw: dict[str, str] = json.loads(await f.read())  # pyright: ignore[reportAny]
                self._model_last_used = {
                    ModelId(k): datetime.fromisoformat(v) for k, v in raw.items()
                }
                logger.debug(
                    f"Loaded model usage for {len(self._model_last_used)} models"
                )
        except Exception as e:
            logger.warning(f"Failed to load model usage: {e}")
