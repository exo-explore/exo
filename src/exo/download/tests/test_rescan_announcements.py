"""The periodic rescan must not flood the event log with unchanged download statuses."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.tests.test_download_status_not_lost import (
    MODEL_DIR,
    MODEL_ID,
    NODE_ID,
    SHARD,
    FakeShardDownloader,
    _collect_events,  # pyright: ignore[reportPrivateUsage]
    _setup_coordinator,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.utils.channels import Receiver


class CountingShardDownloader(FakeShardDownloader):
    """Reports one model that isn't complete, with `downloaded` bytes already on disk."""

    def __init__(self, downloaded: Memory) -> None:
        super().__init__(status="not_started")
        self.downloaded = downloaded
        self.scans = 0

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        self.scans += 1
        yield (
            MODEL_DIR,
            RepoDownloadProgress(
                repo_id=str(MODEL_ID),
                repo_revision="main",
                shard=SHARD,
                completed_files=0,
                total_files=13,
                downloaded=self.downloaded,
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_mb(100),
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                status="not_started",
            ),
        )


def _announcements(events: list[Event]) -> list[NodeDownloadProgress]:
    return [
        e
        for e in events
        if isinstance(e, NodeDownloadProgress)
        and e.download_progress.shard_metadata.model_card.model_id == MODEL_ID
    ]


async def _run_for(
    coordinator: DownloadCoordinator, event_recv: Receiver[Event], seconds: float
) -> list[Event]:
    task = asyncio.create_task(coordinator.run())
    try:
        return await _collect_events(event_recv, timeout=seconds)
    finally:
        await coordinator.shutdown()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.fixture
def fast_rescans():
    with (
        patch("exo.download.coordinator._RESCAN_INTERVAL_SECONDS", 0.01),
        patch("exo.download.coordinator.resolve_existing_model", return_value=None),
    ):
        yield


@pytest.mark.usefixtures("fast_rescans")
async def test_model_never_downloaded_is_announced_once() -> None:
    downloader = CountingShardDownloader(downloaded=Memory.from_bytes(0))
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    with patch("exo.download.coordinator._FULL_RESCAN_EVERY", 1_000_000):
        events = await _run_for(coordinator, event_recv, 0.5)

    assert downloader.scans > 5
    announced = _announcements(events)
    assert len(announced) == 1
    assert isinstance(announced[0].download_progress, DownloadPending)


@pytest.mark.usefixtures("fast_rescans")
async def test_full_rescan_repeats_model_never_downloaded() -> None:
    downloader = CountingShardDownloader(downloaded=Memory.from_bytes(0))
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    with patch("exo.download.coordinator._FULL_RESCAN_EVERY", 3):
        events = await _run_for(coordinator, event_recv, 0.5)

    full_rescans = -(-downloader.scans // 3)  # scans 0, 3, 6, ...
    assert full_rescans >= 2
    # Stopping mid-scan can leave the last full rescan unannounced
    assert len(_announcements(events)) in (full_rescans - 1, full_rescans)


@pytest.mark.usefixtures("fast_rescans")
async def test_model_with_local_files_is_announced_every_rescan() -> None:
    """So the master regains it within a rescan if it dropped this node's state."""
    downloader = CountingShardDownloader(downloaded=Memory.from_mb(95))
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    with patch("exo.download.coordinator._FULL_RESCAN_EVERY", 1_000_000):
        events = await _run_for(coordinator, event_recv, 0.5)

    assert downloader.scans > 5
    # Stopping mid-scan can leave the last rescan unannounced
    assert len(_announcements(events)) in (downloader.scans - 1, downloader.scans)


@pytest.mark.usefixtures("fast_rescans")
async def test_changed_status_is_announced() -> None:
    downloader = CountingShardDownloader(downloaded=Memory.from_bytes(0))
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    with patch("exo.download.coordinator._FULL_RESCAN_EVERY", 1_000_000):
        task = asyncio.create_task(coordinator.run())
        try:
            first = _announcements(await _collect_events(event_recv, timeout=0.2))
            downloader.downloaded = Memory.from_mb(50)
            second = _announcements(await _collect_events(event_recv, timeout=0.2))
        finally:
            await coordinator.shutdown()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert len(first) == 1
    assert isinstance(first[0].download_progress, DownloadPending)
    assert first[0].download_progress.downloaded == Memory.from_bytes(0)
    assert second, "the changed status must be announced"
    assert all(
        isinstance(e.download_progress, DownloadPending)
        and e.download_progress.downloaded == Memory.from_mb(50)
        for e in second
    )


async def test_start_download_repeats_known_status() -> None:
    """A worker asks to download a model it can't see in the cluster state; if this
    node already has it, the status must be re-announced rather than silently dropped."""
    coordinator, _cmd_send, event_recv = _setup_coordinator(FakeShardDownloader())
    completed = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=SHARD,
        total=Memory.from_mb(100),
        model_directory=str(MODEL_DIR),
    )
    coordinator.download_status[MODEL_ID] = completed

    await coordinator._start_download(SHARD)  # pyright: ignore[reportPrivateUsage]

    announced = _announcements(event_recv.collect())
    assert [e.download_progress for e in announced] == [completed]
