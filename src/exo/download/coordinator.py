"""Model downloads on one node: one state machine per model.

    idle ──Start──> running ──done──> completed
      ^                │
      │ Cancel/Delete  │ an error retrying can't fix
      │                v
      └─────────── failed ──Start──> running

Idle models are reported from disk: Completed, or Pending with the bytes already
there. Cancel and Delete stop a running download and wait for it to end before
anything else happens. Commands never wait on the network. A failed download is
retried by the next StartDownload; the worker sends those, with a backoff, for
as long as an instance on this node needs the model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, to_thread
from loguru import logger

from exo.download import store, transfer
from exo.download.hub import Hub, PermanentError, RevisionGoneError
from exo.download.store import NotEnoughSpaceError
from exo.download.transfer import (
    DownloadSettings,
    Locks,
    Progress,
    disk_error_message,
)
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup

# Every rescan re-announces models with files on this node, so a status the master
# dropped heals within a minute; models never downloaded are announced when they
# change, and on every this-many-th scan.
_FULL_RESCAN_EVERY = 10


@dataclass
class _Running:
    progress: Progress
    scope: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    done: anyio.Event = field(default_factory=anyio.Event)
    # held while a progress report is sent, so none can follow the final status
    reporting: anyio.Lock = field(default_factory=anyio.Lock)
    stopped: bool = False
    # progress is reported once the files are listed and the dir chosen
    planned: bool = False
    # what was on disk when it started, shown until the files are listed
    initial: DownloadPending | None = None


@dataclass
class _Model:
    shard: ShardMetadata
    status: DownloadProgress | None = None
    running: _Running | None = None
    # bumped on every change of phase, so a rescan that raced a command backs off
    generation: int = 0


def _full_shard(card: ModelCard) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    settings: DownloadSettings
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    offline: bool = False

    _models: dict[ModelId, _Model] = field(init=False, default_factory=dict)
    _locks: Locks = field(init=False, default_factory=Locks)
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)
    # Cleared once the node announces it is leaving: a status applied after its departure
    # would list it in the cluster's downloads again
    _reporting: bool = field(init=False, default=True)

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._rescan_loop)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            pass
        finally:
            self.download_command_receiver.close()
            self.event_sender.close()
            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    def stop_reporting(self) -> None:
        """Stop sending download statuses: this node is leaving the cluster."""
        self._reporting = False

    async def reannounce(self) -> None:
        """Send every status again, e.g. to a new master whose state may lack them."""
        for model in list(self._models.values()):
            if model.status is not None:
                await self._send(model.status)

    # --- commands --------------------------------------------------------------------

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for forwarded in commands:
                command = forwarded.command
                if command.target_node_id != self.node_id:
                    continue
                try:
                    match command:
                        case StartDownload(shard_metadata=shard):
                            await self._start(shard)
                        case CancelDownload(model_id=model_id):
                            await self._cancel(model_id)
                        case DeleteDownload(model_id=model_id):
                            await self._delete(model_id)
                except ValueError as error:
                    # e.g. a model id that isn't a valid folder name
                    logger.warning(f"Ignoring download command {command}: {error}")
                except (BrokenResourceError, ClosedResourceError):
                    raise
                except Exception as error:
                    # one model's trouble must never stop the others
                    logger.opt(exception=error).error(
                        f"Download command {type(command).__name__} failed"
                    )

    async def _start(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id
        model = self._models.setdefault(model_id, _Model(shard))
        if model.running is not None or isinstance(model.status, DownloadCompleted):
            # Workers only ask when the cluster's state lacks this status: repeat it
            if model.status is not None:
                await self._send(model.status)
            return
        model.shard = shard
        model.generation += 1
        try:
            status = await to_thread.run_sync(self._idle_status, model)
        except OSError as error:
            logger.opt(exception=error).warning(f"Can't read {model_id} on disk")
            await self._fail(model, disk_error_message(error, f"reading {model_id}"))
            return
        if isinstance(status, DownloadCompleted):
            await self._settle(model, status)
        elif self.offline:
            await self._fail(model, f"Offline, and {model_id} isn't fully downloaded.")
        else:
            await self._launch(model, status)

    async def _cancel(self, model_id: ModelId) -> None:
        model = self._models.get(model_id)
        if model is None:
            return
        if model.running is None and not isinstance(model.status, DownloadFailed):
            return
        logger.info(f"Cancelling download of {model_id}")
        await self._stop(model)
        await self._settle_idle(model)

    async def _delete(self, model_id: ModelId) -> None:
        model = self._models.get(model_id)
        if (
            model is not None
            and isinstance(model.status, DownloadCompleted)
            and model.status.read_only
        ):
            logger.warning(f"Refusing to delete read-only model {model_id}")
            return
        if model is not None:
            await self._stop(model)
        logger.info(f"Deleting model files for {model_id}")
        try:
            if not await to_thread.run_sync(store.delete, model_id):
                logger.warning(f"Model {model_id} was not found on disk")
        except OSError as error:
            logger.opt(exception=error).warning(f"Couldn't delete {model_id}")
            if model is not None:
                reason = os.strerror(error.errno) if error.errno else str(error)
                await self._fail(model, f"Couldn't delete all of {model_id}: {reason}.")
            return
        if model is not None:
            await self._settle_idle(model)

    async def _settle_idle(self, model: _Model) -> None:
        """Report what is on disk now (Pending or Completed)."""
        try:
            status = await to_thread.run_sync(self._idle_status, model)
        except OSError as error:
            card = model.shard.model_card
            logger.opt(exception=error).warning(f"Can't read {card.model_id} on disk")
            await self._fail(
                model, disk_error_message(error, f"reading {card.model_id}")
            )
            return
        await self._settle(model, status)

    # --- running a download --------------------------------------------------------

    async def _launch(self, model: _Model, initial: DownloadProgress) -> None:
        running = _Running(Progress())
        if isinstance(initial, DownloadPending):
            running.initial = initial
        model.running = running
        model.generation += 1
        # Ongoing at once, so the cluster sees it running (and the master can
        # cancel it) while the files are still being listed
        model.status = self._ongoing(model, running)
        await self._send(model.status)
        self._tg.start_soon(self._download, model, running)
        self._tg.start_soon(self._report, model, running)

    async def _stop(self, model: _Model) -> None:
        """Stop the running download, if any, and wait until it has ended."""
        model.generation += 1
        running = model.running
        if running is not None:
            running.scope.cancel()
            await running.done.wait()

    async def _download(self, model: _Model, running: _Running) -> None:
        card = model.shard.model_card
        outcome: str | Path | None = None
        try:
            with running.scope:
                outcome = await self._fetch(model, running)
        except NotEnoughSpaceError as error:
            outcome = error.message
        except PermanentError as error:
            outcome = error.message
        except Exception as error:
            logger.opt(exception=error).error(f"Download of {card.model_id} failed")
            outcome = f"Download failed unexpectedly: {type(error).__name__}."
        finally:
            running.stopped = True
            with anyio.CancelScope(shield=True):
                async with running.reporting:
                    pass
            if model.running is running:
                model.running = None
            running.done.set()
        match outcome:
            case None:
                pass  # cancelled; whoever cancelled reports the new status
            case Path() as directory:
                await self._settle(model, self._completed(model, directory))
            case str() as message:
                logger.warning(f"Download of {card.model_id} failed: {message}")
                await self._fail(model, message)

    async def _fetch(self, model: _Model, running: _Running) -> Path:
        settings = self.settings
        card, progress = model.shard.model_card, running.progress
        for attempt in (1, 2):
            async with Hub.connect(
                connect_timeout=settings.connect_timeout,
                idle_timeout=settings.idle_timeout,
                metadata_timeout=settings.metadata_timeout,
            ) as hub:
                try:
                    parts = await transfer.retrying(
                        lambda: transfer.plan(
                            hub,
                            card,
                            progress,
                            self._locks,
                            lambda repo: self._reserved(model, repo),
                        ),
                        progress,
                        settings,
                    )
                    running.planned = True
                    await self._announce_progress(model, running)
                    await transfer.download(hub, parts, progress, settings, self._locks)
                    return parts[0].model_dir
                except RevisionGoneError as error:
                    if attempt == 2:
                        raise
                    logger.warning(f"{error.repo}: pinned commit gone, pinning again")
                    running.planned = False
                    for directory in store.writable_copies(error.repo):
                        async with self._locks.directory(directory):
                            await to_thread.run_sync(store.forget_manifest, directory)
                    progress.reset()
        raise AssertionError("unreachable")

    def _reserved(self, excluding: _Model, repo: ModelId) -> dict[Path, int]:
        """Bytes other running downloads still need, by models dir; a repo dir two
        downloads share (a vision weights repo) counts once, and ``repo`` not at all."""
        by_dir: dict[Path, int] = {}
        for model in self._models.values():
            if model.running is not None and model is not excluding:
                for directory, left in model.running.progress.remaining().items():
                    if directory.name != repo.normalize():
                        by_dir[directory] = max(by_dir.get(directory, 0), left)
        reserved: dict[Path, int] = {}
        for directory, left in by_dir.items():
            reserved[directory.parent] = reserved.get(directory.parent, 0) + left
        return reserved

    async def _report(self, model: _Model, running: _Running) -> None:
        """Send progress at most every ``progress_interval``, and only when it changed."""
        last_fetched = 0
        last_speed = 0.0
        while True:
            await anyio.sleep(self.settings.progress_interval)
            if running.stopped:
                return
            speed = running.progress.speed()
            if not running.planned or (
                running.progress.fetched == last_fetched
                and not (last_speed > 0 and speed == 0)
            ):
                continue
            last_fetched, last_speed = running.progress.fetched, speed
            try:
                await self._announce_progress(model, running, speed)
            except (BrokenResourceError, ClosedResourceError):
                raise
            except Exception as error:
                logger.opt(exception=error).error("Reporting download progress failed")

    async def _announce_progress(
        self, model: _Model, running: _Running, speed: float = 0.0
    ) -> None:
        async with running.reporting:
            if running.stopped:
                return
            model.status = self._ongoing(model, running, speed)
            await self._send(model.status)

    async def _fail(self, model: _Model, message: str) -> None:
        """Report a failure. It is retried by the next StartDownload: the worker
        sends one, with a backoff, while an instance on this node needs the model."""
        model.generation += 1
        await self._settle(
            model,
            DownloadFailed(
                node_id=self.node_id,
                shard_metadata=model.shard,
                error_message=message,
                model_directory=self._directory(model),
            ),
        )

    # --- statuses ----------------------------------------------------------------------

    async def _settle(self, model: _Model, status: DownloadProgress) -> None:
        model.status = status
        await self._send(status)

    async def _send(self, status: DownloadProgress) -> None:
        if not self._reporting:
            return
        try:
            await self.event_sender.send(NodeDownloadProgress(download_progress=status))
        except (BrokenResourceError, ClosedResourceError):
            logger.debug("Event stream closed; dropping a download status")

    def _directory(self, model: _Model) -> str:
        running = model.running
        if running is not None and running.progress.model_dir is not None:
            return str(running.progress.model_dir)
        if model.status is not None and model.status.model_directory:
            return model.status.model_directory
        return str(store.default_dir() / model.shard.model_card.model_id.normalize())

    def _completed(self, model: _Model, directory: Path) -> DownloadCompleted:
        manifest = store.read_manifest(directory)
        total = (
            Memory.from_bytes(manifest.total)
            if manifest is not None
            else model.shard.model_card.storage_size
        )
        return DownloadCompleted(
            node_id=self.node_id,
            shard_metadata=model.shard,
            total=total,
            model_directory=str(directory),
            read_only=store.is_read_only(directory),
        )

    def _idle_status(self, model: _Model) -> DownloadProgress:
        """Completed or Pending, from what is on disk (blocking)."""
        card = model.shard.model_card
        found = store.find_complete(card.model_id, card)
        if found is not None:
            return self._completed(model, found)
        directory = next(
            (d for d in store.candidates(card.model_id) if not store.is_read_only(d)),
            None,
        )
        downloaded, total = 0, card.storage_size.in_bytes
        if directory is not None:
            manifest = store.read_manifest(directory)
            if manifest is not None:
                total = manifest.total
                downloaded = sum(
                    store.present_bytes(directory, f) for f in manifest.files
                )
            else:
                downloaded = min(store.legacy_present_bytes(directory), total)
        return DownloadPending(
            node_id=self.node_id,
            shard_metadata=model.shard,
            model_directory=str(
                directory or store.default_dir() / card.model_id.normalize()
            ),
            downloaded=Memory.from_bytes(downloaded),
            total=Memory.from_bytes(total),
        )

    def _ongoing(
        self, model: _Model, running: _Running, speed: float = 0.0
    ) -> DownloadOngoing:
        progress = running.progress
        now = time.monotonic()
        if progress.files:
            total, downloaded = progress.total, progress.downloaded
        elif running.initial is not None and running.initial.downloaded.in_bytes > 0:
            total = running.initial.total.in_bytes
            downloaded = running.initial.downloaded.in_bytes
        else:
            total, downloaded = 0, 0  # not listed yet
        files = {
            path: DownloadProgressData(
                total=Memory.from_bytes(file.size),
                downloaded=Memory.from_bytes(file.downloaded),
                downloaded_this_session=Memory.from_bytes(file.fetched),
                completed_files=1 if file.done else 0,
                total_files=1,
                speed=file.speed(now),
                eta_ms=_eta_ms(file.size - file.downloaded, file.speed(now)),
                files={},
            )
            for path, file in progress.files.items()
        }
        return DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=model.shard,
            model_directory=self._directory(model),
            download_progress=DownloadProgressData(
                total=Memory.from_bytes(total),
                downloaded=Memory.from_bytes(min(downloaded, total)),
                downloaded_this_session=Memory.from_bytes(progress.fetched),
                completed_files=progress.completed_files,
                total_files=len(progress.files),
                speed=speed,
                eta_ms=_eta_ms(total - downloaded, speed),
                files=files,
            ),
        )

    # --- the rescan ----------------------------------------------------------------

    async def _rescan_loop(self) -> None:
        for scan in count():
            try:
                cards = await model_cards.card_cache.list_all()
            except Exception as error:
                logger.opt(exception=error).error("Download rescan can't list models")
                cards = []
            for card in cards:
                try:
                    await self._rescan(card, resend=scan % _FULL_RESCAN_EVERY == 0)
                except (BrokenResourceError, ClosedResourceError):
                    raise
                except Exception as error:
                    # one unreadable model must not hide the others
                    logger.opt(exception=error).warning(
                        f"Download rescan of {card.model_id} failed"
                    )
            await anyio.sleep(self.settings.rescan_interval)

    async def _rescan(self, card: ModelCard, *, resend: bool) -> None:
        model = self._models.setdefault(card.model_id, _Model(_full_shard(card)))
        if model.running is not None or isinstance(model.status, DownloadFailed):
            return
        generation = model.generation
        status = await to_thread.run_sync(self._idle_status, model)
        if model.generation != generation or model.running is not None:
            return  # a command got there first
        changed = status != model.status
        model.status = status
        has_files = not (
            isinstance(status, DownloadPending) and status.downloaded.in_bytes == 0
        )
        if changed or resend or has_files:
            await self._send(status)


def _eta_ms(remaining: int, speed: float) -> int:
    return int(remaining / speed * 1000) if speed > 0 and remaining > 0 else 0
