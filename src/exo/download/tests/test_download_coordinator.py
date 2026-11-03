"""The coordinator's own rules: what it announces, when, and how it retries."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download import coordinator
from exo.download.tests.fake_hub import (
    FakeHub,
    FakeRepo,
    content_for,
    model_repo_files,
)
from exo.download.tests.harness import DownloadNode, Timing, card_for, expected_files
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadPending,
)

REPO = "fake-org/tiny"
SHARD1 = "model-00001-of-00002.safetensors"


@pytest.fixture
async def hub() -> AsyncIterator[FakeHub]:
    async with FakeHub() as hub:
        yield hub


@pytest.fixture
def repo(hub: FakeHub) -> FakeRepo:
    return hub.add_repo(FakeRepo(REPO, model_repo_files(REPO)))


@pytest.fixture
def rare_full_rescans() -> Iterator[None]:
    with patch.object(coordinator, "_FULL_RESCAN_EVERY", 1_000_000):
        yield


def announcements(node: DownloadNode) -> int:
    return sum(
        1
        for e in node.events
        if e.download_progress.shard_metadata.model_card.model_id == REPO
    )


FAST_RESCANS = Timing(rescan_interval=0.05)


@pytest.mark.usefixtures("rare_full_rescans")
async def test_model_never_downloaded_is_announced_once_from_its_card(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=FAST_RESCANS
    ) as node:
        status = await node.wait_for(REPO, DownloadPending)
        await asyncio.sleep(0.5)
        assert announcements(node) == 1
        assert isinstance(status, DownloadPending)
        assert status.total == card_for(repo).storage_size
        assert hub.requests == [], "no Hub request for a model nobody asked for"


async def test_full_rescans_repeat_models_never_downloaded(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    with patch.object(coordinator, "_FULL_RESCAN_EVERY", 2):
        async with DownloadNode(
            hub, tmp_path, [card_for(repo)], worker=False, timing=FAST_RESCANS
        ) as node:
            await asyncio.sleep(0.5)
            assert announcements(node) >= 3


@pytest.mark.usefixtures("rare_full_rescans")
async def test_model_with_files_is_announced_every_rescan(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    model_dir = tmp_path / "models" / ModelId(REPO).normalize()
    model_dir.mkdir(parents=True)
    (model_dir / (SHARD1 + ".partial")).write_bytes(b"x" * 1000)
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=FAST_RESCANS
    ) as node:
        status = await node.wait_for(REPO, DownloadPending)
        assert (
            isinstance(status, DownloadPending) and status.downloaded.in_bytes == 1000
        )
        await asyncio.sleep(0.5)
        assert announcements(node) >= 4


async def test_completed_survives_rescans_and_start_repeats_it(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=FAST_RESCANS
    ) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        await asyncio.sleep(0.3)
        after = node.statuses(REPO)[
            [type(s) for s in node.statuses(REPO)].index(DownloadCompleted) :
        ]
        assert all(isinstance(s, DownloadCompleted) for s in after)
        before = announcements(node)
        await node.start_download(REPO)
        await asyncio.sleep(0.1)
        assert announcements(node) > before


async def test_reannounce_keeps_the_download_going(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.start_download(REPO)
        reached = asyncio.Event()
        hub.when_sent(SHARD1, 50_000, reached.set)
        await reached.wait()
        before = announcements(node)
        await node.reannounce()
        await asyncio.sleep(0.05)
        assert announcements(node) > before
        await node.wait_for(REPO, DownloadCompleted, timeout=10, fail_fast=True)
        assert hub.body_bytes(REPO) == sum(
            len(c) for c in expected_files(repo).values()
        )
        assert hub.count(kind="tree") == 1


async def test_a_failed_download_an_instance_needs_is_retried_until_it_succeeds(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with DownloadNode(hub, tmp_path, [card_for(repo)]) as node:
        node.want(REPO)  # as the worker does for a placed instance
        node.write_budget = 100_000
        await node.wait_for(REPO, DownloadFailed)
        node.write_budget = None
        await node.wait_for(REPO, DownloadCompleted, timeout=10)
        assert node.files_on_disk(REPO) == expected_files(repo)


async def test_a_failed_download_nothing_needs_is_not_retried(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        node.write_budget = 100_000
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        node.write_budget = None
        fetched = hub.body_bytes(REPO)
        await asyncio.sleep(3)  # five minutes and more, at this timing
        assert isinstance(node.status(REPO), DownloadFailed)
        assert hub.body_bytes(REPO) == fetched
        # a new request (the dashboard's download button) retries it at once
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=10)


async def test_cancelling_a_failed_download_shows_what_is_on_disk(
    hub: FakeHub, tmp_path: Path
) -> None:
    repo = hub.add_repo(FakeRepo(REPO, model_repo_files(REPO), gated=True))
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        await node.cancel(REPO)
        await node.wait_for(REPO, DownloadPending)


async def test_cancel_of_an_unknown_model_does_nothing(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.cancel("fake-org/unknown")
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)


async def test_resuming_keeps_the_pinned_commit(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    first_commit = repo.main.sha
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.start_download(REPO)
        reached = asyncio.Event()
        hub.when_sent(SHARD1, 50_000, reached.set)
        await reached.wait()
        await node.cancel(REPO)
        await node.wait_for(REPO, DownloadPending)
        repo.commit({"config.json": b'{"v": 2}', SHARD1: content_for("v2", 1000)})
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=10)
        assert node.files_on_disk(REPO) == expected_files(repo, first_commit)
        assert hub.count(kind="tree") == 1
