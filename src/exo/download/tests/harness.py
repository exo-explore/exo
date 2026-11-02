"""Runs exo's real download stack against a :class:`FakeHub`.

A :class:`DownloadNode` is one node's download side: the real
``DownloadCoordinator`` and shard downloader, fed with download commands, its
events applied to a ``State`` with the real ``apply`` function, so tests see what
the cluster would see. With ``worker=True`` it also asks for the models it
:meth:`~DownloadNode.want`, the way the worker does for a placed instance: it sends
``StartDownload`` while the model has no status or a ``DownloadPending`` one, with
the worker's backoff, and while it is ``DownloadFailed``, with the worker's slower
backoff for failed downloads.

Everything that depends on how the downloader is built lives in the
"implementation adapter" section, so the scenarios don't change when the
implementation does.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, final
from unittest.mock import patch

from exo.download.tests.fake_hub import FakeHub, FakeRepo
from exo.shared.apply import apply_node_download_progress
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    DownloadCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.keyed_backoff import KeyedBackoff

NODE_ID = NodeId("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
IGNORED_PREFIXES = ("original/", "metal/")


class DiskUsage(NamedTuple):
    """What ``shutil.disk_usage`` returns."""

    total: int
    used: int
    free: int


@dataclass(frozen=True)
class Timing:
    """Real-world timing knobs, shrunk so tests run in seconds.

    The defaults are the production values divided by about 100.
    """

    # No bytes for this long means the connection is dead (60 s in production).
    stall_timeout: float = 0.6
    connect_timeout: float = 0.6
    # Cap on one whole file request (main: 30 min). None: no cap.
    file_request_timeout: float | None = 18.0
    # Cap on one metadata request: file list, HEAD (30 s in production).
    metadata_timeout: float = 0.6
    # Multiplies every retry and backoff sleep, including HF's rate-limit waits.
    backoff_scale: float = 0.01
    # How often the node rescans its model directories (60 s in production).
    rescan_interval: float = 0.6
    # The worker's backoff between StartDownload requests (0.5 s doubling to 10 s).
    worker_backoff_base: float = 0.005
    worker_backoff_cap: float = 0.1
    # ... and between StartDownload requests for a failed download (30 s to 5 min)
    worker_failed_backoff_base: float = 0.3
    worker_failed_backoff_cap: float = 3.0


def card_for(repo: FakeRepo, n_layers: int = 2) -> ModelCard:
    return ModelCard(
        model_id=ModelId(repo.repo_id),
        storage_size=Memory.from_bytes(
            sum(
                b.size for p, b in repo.main.files.items() if p.endswith(".safetensors")
            )
        ),
        n_layers=n_layers,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def shard_for(
    card: ModelCard, start_layer: int = 0, end_layer: int | None = None
) -> ShardMetadata:
    end = card.n_layers if end_layer is None else end_layer
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=start_layer,
        end_layer=end,
        n_layers=card.n_layers,
    )


def expected_files(repo: FakeRepo, revision: str | None = None) -> dict[str, bytes]:
    commit = repo.main if revision is None else repo.at(revision)
    assert commit is not None
    return {
        path: blob.content
        for path, blob in commit.files.items()
        if not path.startswith(IGNORED_PREFIXES)
    }


@final
class DownloadNode:
    def __init__(
        self,
        hub: FakeHub,
        root: Path,
        cards: list[ModelCard],
        *,
        timing: Timing | None = None,
        offline: bool = False,
        worker: bool = True,
        extra_dirs: tuple[Path, ...] = (),
        read_only_dirs: tuple[Path, ...] = (),
    ) -> None:
        self.hub = hub
        self.root = root
        self.models_dir = root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dirs: tuple[Path, ...] = (self.models_dir, *extra_dirs)
        self.read_only_dirs = read_only_dirs
        self.cards = {card.model_id: card for card in cards}
        self.timing = timing or Timing()
        self.offline = offline
        self.worker = worker
        self.state = State()
        self.history: dict[ModelId, list[DownloadProgress]] = {}
        self.events: list[NodeDownloadProgress] = []
        self.wanted: dict[ModelId, ShardMetadata] = {}
        self.commands_sent: list[DownloadCommand] = []
        # Disk faults: (path glob, bytes allowed before ENOSPC)
        self.write_budget: int | None = None
        # Free bytes to report for a models directory (None: the real value)
        self.free_space: dict[Path, int] = {}
        self._backoff: KeyedBackoff[ModelId] = KeyedBackoff(
            base=self.timing.worker_backoff_base, cap=self.timing.worker_backoff_cap
        )
        self._failed_backoff: KeyedBackoff[ModelId] = KeyedBackoff(
            base=self.timing.worker_failed_backoff_base,
            cap=self.timing.worker_failed_backoff_cap,
        )
        self._command_sender: Sender[ForwarderDownloadCommand] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._patches = contextlib.ExitStack()
        self._implementation: _Implementation | None = None

    # --- lifecycle --------------------------------------------------------------

    async def __aenter__(self) -> DownloadNode:
        self._patches.enter_context(_environment(self))
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
        self._patches.close()

    async def start(self) -> None:
        command_sender, command_receiver = channel[ForwarderDownloadCommand]()
        event_sender, event_receiver = channel[Event]()
        self._command_sender = command_sender
        self._implementation = _Implementation(self, command_receiver, event_sender)
        self._tasks = [
            asyncio.create_task(self._implementation.run()),
            asyncio.create_task(self._collect(event_receiver)),
        ]
        if self.worker:
            self._tasks.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        """Stop the node (as a process exit would; files on disk stay)."""
        if self._implementation is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._implementation.shutdown(), 5)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(BaseException):
                await task
        self._tasks = []
        self._implementation = None

    async def reannounce(self) -> None:
        """What the node does when a new master is elected."""
        assert self._implementation is not None
        await self._implementation.reannounce()

    async def restart(self) -> None:
        await self.stop()
        # A restarted node keeps the cluster's view of its downloads until it
        # announces new ones, exactly as the master's state would.
        await self.start()

    # --- commands ---------------------------------------------------------------

    async def send(self, command: DownloadCommand) -> None:
        assert self._command_sender is not None
        self.commands_sent.append(command)
        await self._command_sender.send(
            ForwarderDownloadCommand(origin=SystemId(), command=command)
        )

    async def start_download(
        self, model_id: str, shard: ShardMetadata | None = None
    ) -> None:
        card = self.cards[ModelId(model_id)]
        await self.send(
            StartDownload(
                target_node_id=NODE_ID, shard_metadata=shard or shard_for(card)
            )
        )

    async def cancel(self, model_id: str) -> None:
        await self.send(
            CancelDownload(target_node_id=NODE_ID, model_id=ModelId(model_id))
        )

    async def delete(self, model_id: str) -> None:
        await self.send(
            DeleteDownload(target_node_id=NODE_ID, model_id=ModelId(model_id))
        )

    def want(self, model_id: str, shard: ShardMetadata | None = None) -> None:
        """Ask for the model the way a worker with a placed instance does."""
        card = self.cards[ModelId(model_id)]
        self.wanted[card.model_id] = shard or shard_for(card)

    def unwant(self, model_id: str) -> None:
        self.wanted.pop(ModelId(model_id), None)

    # --- observation ------------------------------------------------------------

    def status(self, model_id: str) -> DownloadProgress | None:
        for progress in self.state.downloads.get(NODE_ID, []):
            if progress.shard_metadata.model_card.model_id == model_id:
                return progress
        return None

    def statuses(self, model_id: str) -> list[DownloadProgress]:
        return self.history.get(ModelId(model_id), [])

    def failures(self, model_id: str) -> list[DownloadFailed]:
        return [s for s in self.statuses(model_id) if isinstance(s, DownloadFailed)]

    async def wait_for(
        self,
        model_id: str,
        kind: type[DownloadProgress] | tuple[type[DownloadProgress], ...],
        timeout: float = 10.0,
        predicate: Callable[[DownloadProgress], bool] = lambda _: True,
        fail_fast: bool = False,
    ) -> DownloadProgress:
        """Wait for the status to be ``kind``.

        With ``fail_fast``, give up as soon as the download is reported failed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(model_id)
            if isinstance(status, kind) and predicate(status):
                return status
            if fail_fast and (failures := self.failures(model_id)):
                raise AssertionError(
                    f"{model_id} was reported failed: {failures[0].error_message!r}"
                )
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"{model_id} never reached {kind} in {timeout}s; status is {_describe(self.status(model_id))}, "
            f"history: {[type(s).__name__ for s in self.statuses(model_id)]}"
        )

    async def wait_for_failure(
        self, model_id: str, timeout: float = 10.0
    ) -> DownloadFailed:
        """Wait until the download has been reported failed (it may not stay so)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if failures := self.failures(model_id):
                return failures[0]
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"{model_id} was never reported failed in {timeout}s; status is "
            f"{_describe(self.status(model_id))}"
        )

    async def quiet(self, seconds: float) -> None:
        """Wait until the hub saw no request for ``seconds``."""
        last = len(self.hub.requests)
        quiet_since = time.monotonic()
        while time.monotonic() - quiet_since < seconds:
            await asyncio.sleep(0.02)
            if len(self.hub.requests) != last:
                last = len(self.hub.requests)
                quiet_since = time.monotonic()

    def model_dirs(self, model_id: str) -> list[Path]:
        name = ModelId(model_id).normalize()
        return [
            d / name
            for d in (*self.models_dirs, *self.read_only_dirs)
            if (d / name).is_dir()
        ]

    def files_on_disk(
        self, model_id: str, models_dir: Path | None = None
    ) -> dict[str, bytes]:
        directory = (models_dir or self.models_dir) / ModelId(model_id).normalize()
        files: dict[str, bytes] = {}
        if not directory.is_dir():
            return files
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_file() and (
                not relative.startswith(".") or relative == ".gitattributes"
            ):
                files[relative] = path.read_bytes()
        return files

    def partial_files(self) -> list[Path]:
        return [
            p
            for d in self.models_dirs
            if d.is_dir()
            for p in d.rglob("*")
            if p.name.endswith(".partial") or ".partial." in p.name
        ]

    # --- internals --------------------------------------------------------------

    async def _collect(self, events: Receiver[Event]) -> None:
        with events:
            async for event in events:
                if not isinstance(event, NodeDownloadProgress):
                    continue
                self.events.append(event)
                self.state = apply_node_download_progress(event, self.state)
                progress = event.download_progress
                self.history.setdefault(
                    progress.shard_metadata.model_card.model_id, []
                ).append(progress)

    async def _worker_loop(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            for model_id, shard in list(self.wanted.items()):
                status = self.status(model_id)
                if isinstance(status, (DownloadOngoing, DownloadCompleted)):
                    continue
                failed = isinstance(status, DownloadFailed)
                backoff = self._failed_backoff if failed else self._backoff
                if backoff.should_proceed(model_id):
                    self._backoff.record_attempt(model_id)
                    if failed:
                        self._failed_backoff.record_attempt(model_id)
                    await self.send(
                        StartDownload(target_node_id=NODE_ID, shard_metadata=shard)
                    )


def _describe(status: DownloadProgress | None) -> str:
    match status:
        case None:
            return "None"
        case DownloadOngoing(download_progress=p):
            return f"DownloadOngoing({p.downloaded.in_bytes}/{p.total.in_bytes})"
        case DownloadFailed(error_message=message):
            return f"DownloadFailed({message!r})"
        case DownloadCompleted() | DownloadPending():
            return type(status).__name__


@contextlib.contextmanager
def _environment(node: DownloadNode) -> Iterator[None]:
    """Process-wide settings every implementation reads: env vars and the card list."""
    cache = model_cards._CardCache()  # pyright: ignore[reportPrivateUsage]
    cache.cc = dict(node.cards)
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ["HF_ENDPOINT"] = node.hub.url
    os.environ["HF_HOME"] = str(node.root / "hf_home")
    for key in ("HF_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    try:
        with (
            patch.object(model_cards, "card_cache", cache),
            _implementation_patches(node),
        ):
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_ENV = (
    "HF_ENDPOINT",
    "HF_HOME",
    "HF_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


# ---------------------------------------------------------------------------------
# Implementation adapter: how this implementation is built, configured and faulted.
# ---------------------------------------------------------------------------------

from exo.download import store as _store  # noqa: E402
from exo.download import transfer as _transfer  # noqa: E402
from exo.download.coordinator import DownloadCoordinator  # noqa: E402
from exo.download.transfer import DownloadSettings  # noqa: E402


def settings_for(timing: Timing) -> DownloadSettings:
    scale = timing.backoff_scale
    production = DownloadSettings()
    return DownloadSettings(
        idle_timeout=timing.stall_timeout,
        connect_timeout=timing.connect_timeout,
        metadata_timeout=timing.metadata_timeout,
        retry_base=production.retry_base * scale,
        retry_cap=production.retry_cap * scale,
        rate_limit_cap=production.rate_limit_cap * scale,
        server_wait_scale=scale,
        give_up_after=production.give_up_after * scale,
        rescan_interval=timing.rescan_interval,
        unreachable_after=production.unreachable_after * scale,
        # file_request_timeout has no equivalent: there is no cap on a whole file
    )


@final
class _Implementation:
    def __init__(
        self,
        node: DownloadNode,
        commands: Receiver[ForwarderDownloadCommand],
        events: Sender[Event],
    ) -> None:
        self.coordinator = DownloadCoordinator(
            NODE_ID,
            settings_for(node.timing),
            download_command_receiver=commands,
            event_sender=events,
            offline=node.offline,
        )

    async def run(self) -> None:
        await self.coordinator.run()

    async def shutdown(self) -> None:
        await self.coordinator.shutdown()

    async def reannounce(self) -> None:
        await self.coordinator.reannounce()


@contextlib.contextmanager
def _implementation_patches(node: DownloadNode) -> Iterator[None]:
    real_write_at = _transfer._write_at  # pyright: ignore[reportPrivateUsage]

    def write_at(
        descriptor: int, data: bytes, offset: int, state: _transfer.FileState
    ) -> None:
        budget = node.write_budget
        if budget is not None:
            if budget < len(data):
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            node.write_budget = budget - len(data)
        real_write_at(descriptor, data, offset, state)

    def disk_usage(path: str | os.PathLike[str]) -> object:
        usage = _real_disk_usage(path)
        free = next(
            (f for d, f in node.free_space.items() if Path(path).is_relative_to(d)),
            None,
        )
        if free is None:
            return usage
        return DiskUsage(usage.total, usage.total - free, free)

    with (
        patch.object(_store, "EXO_MODELS_DIRS", node.models_dirs),
        patch.object(_store, "EXO_DEFAULT_MODELS_DIR", node.models_dir),
        patch.object(_store, "EXO_MODELS_READ_ONLY_DIRS", node.read_only_dirs),
        patch.object(_transfer, "_write_at", write_at),
        patch("shutil.disk_usage", disk_usage),
    ):
        yield


_real_disk_usage = shutil.disk_usage
