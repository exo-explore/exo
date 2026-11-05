"""Edge cases from an independent review of the downloader: broken folders, links,
shared vision repos, racing downloads, odd listings, outages and leftovers."""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from exo.download.download_utils import download_file_with_retry
from exo.download.tests.fake_hub import (
    Corrupt,
    FakeHub,
    FakeRepo,
    Fault,
    FirstByteDelay,
    Reset,
    content_for,
    model_repo_files,
    server_error,
)
from exo.download.tests.harness import (
    NODE_ID,
    DownloadNode,
    Timing,
    card_for,
    expected_files,
)
from exo.master.placement import cancel_unnecessary_downloads
from exo.shared.models.model_cards import ModelCard, VisionCardConfig
from exo.shared.types.commands import CancelDownload
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
)

REPO = "fake-org/tiny"
OTHER = "fake-org/other"


@pytest.fixture
async def hub() -> AsyncIterator[FakeHub]:
    async with FakeHub() as hub:
        yield hub


@pytest.fixture
def repo(hub: FakeHub) -> FakeRepo:
    return hub.add_repo(FakeRepo(REPO, model_repo_files(REPO)))


def coordinator_crash(node: DownloadNode) -> BaseException | None:
    """The exception the coordinator's run() died with, if it died."""
    task = node._tasks[0]  # pyright: ignore[reportPrivateUsage]
    if task.done() and not task.cancelled():
        return task.exception()
    return None


# --- deleting -------------------------------------------------------------------------


async def test_delete_of_a_symlinked_model_dir_keeps_the_coordinator_alive(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """Users link a model folder in from another disk. Deleting it must not kill
    the coordinator (in exo, an exception out of the coordinator ends the node)."""
    other = hub.add_repo(FakeRepo(OTHER, model_repo_files(OTHER)))
    elsewhere = tmp_path / "external-disk" / ModelId(REPO).normalize()
    elsewhere.mkdir(parents=True)
    for path, content in expected_files(repo).items():
        (elsewhere / path).write_bytes(content)
    node = DownloadNode(hub, tmp_path, [card_for(repo), card_for(other)], worker=False)
    (node.models_dir / ModelId(REPO).normalize()).symlink_to(elsewhere)
    async with node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        await node.delete(REPO)
        await asyncio.sleep(0.3)
        assert coordinator_crash(node) is None, (
            f"coordinator died: {coordinator_crash(node)!r}"
        )
        await node.start_download(OTHER)
        await node.wait_for(OTHER, DownloadCompleted, timeout=5)


async def test_delete_of_a_dir_it_cannot_fully_remove_keeps_the_coordinator_alive(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """E.g. a sub-folder created by `sudo exo` once. The delete may fail, the
    coordinator must not."""
    other = hub.add_repo(FakeRepo(OTHER, model_repo_files(OTHER)))
    node = DownloadNode(hub, tmp_path, [card_for(repo), card_for(other)], worker=False)
    async with node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        locked = node.models_dir / ModelId(REPO).normalize() / "locked"
        locked.mkdir()
        (locked / "file").write_bytes(b"x")
        locked.chmod(0o500)
        try:
            await node.delete(REPO)
            await asyncio.sleep(0.3)
            assert coordinator_crash(node) is None, (
                f"coordinator died: {coordinator_crash(node)!r}"
            )
            await node.start_download(OTHER)
            await node.wait_for(OTHER, DownloadCompleted, timeout=5)
        finally:
            locked.chmod(0o700)


# --- a failed download nobody wants any more ------------------------------------------


async def test_failed_download_does_not_come_back_after_its_instance_is_deleted(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """The master cancels downloads no instance needs (cancel_unnecessary_downloads,
    on a stop), but only those it sees as DownloadOngoing. A download that
    is DownloadFailed at that moment is not cancelled; it must not then restart on
    its own and fetch the whole model."""
    hub.bandwidth = 2_000_000
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        node.write_budget = 100_000  # the disk fills: the download fails
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        # The user deletes the instance; the master cancels what it sees downloading.
        for command in cancel_unnecessary_downloads({}, node.state.downloads):
            await node.send(command)
        node.write_budget = None  # space is freed later
        # A few auto-retry periods (failed_retry_base is 0.6 s at this timing scale)
        await asyncio.sleep(3)
        assert not isinstance(
            node.status(REPO), (DownloadOngoing, DownloadCompleted)
        ), (
            f"a download nobody asked for restarted: {type(node.status(REPO)).__name__},"
            f" {hub.body_bytes(REPO)} bytes fetched"
        )


async def test_master_cancel_of_a_retrying_download_is_honoured(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """Control for the test above: when the master does send CancelDownload, the
    auto-retry stops."""
    hub.bandwidth = 2_000_000
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        node.write_budget = 100_000
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        await node.send(CancelDownload(target_node_id=NODE_ID, model_id=ModelId(REPO)))
        node.write_budget = None
        await asyncio.sleep(3)
        assert isinstance(node.status(REPO), DownloadPending)


# --- paths from the listing -----------------------------------------------------------


async def test_listed_absolute_path_is_not_written_outside_the_model_dir(
    hub: FakeHub, tmp_path: Path
) -> None:
    """HF_ENDPOINT can point at any mirror. A listing with an absolute path must not
    make exo write outside its models folder."""
    outside = tmp_path / "outside" / "planted.json"
    outside.parent.mkdir()
    files = model_repo_files(REPO)
    files[str(outside)] = b'{"planted": true}'
    repo = hub.add_repo(FakeRepo(REPO, files))
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, (DownloadCompleted, DownloadFailed), timeout=10)
        assert not outside.exists(), "a file was written outside the models folder"


# --- two downloads at once ------------------------------------------------------------


def vision_card(model_id: str, vision_repo: str, storage: int) -> ModelCard:
    card = card_for(FakeRepo(model_id, {}))
    return card.model_copy(
        update={
            "storage_size": card.storage_size.from_bytes(storage),
            "vision": VisionCardConfig(
                image_token_id=1, model_type="fake", weights_repo=vision_repo
            ),
        }
    )


async def test_two_models_sharing_a_vision_repo_download_together(
    hub: FakeHub, tmp_path: Path
) -> None:
    """Two cards with the same vision weights repo (as two Kimi K2.6 variants
    have), downloaded at the same time: both must complete, neither may fail."""
    vision = "fake-org/shared-vision"
    hub.add_repo(
        FakeRepo(
            vision,
            {
                "config.json": b'{"model_type": "fake"}',
                "vision.safetensors": content_for("vision", 600_000),
            },
        )
    )
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    b = hub.add_repo(FakeRepo("fake-org/b", model_repo_files("fake-org/b")))
    hub.bandwidth = 1_500_000
    cards = [vision_card(a.repo_id, vision, 1), vision_card(b.repo_id, vision, 1)]
    async with DownloadNode(
        hub, tmp_path, cards, worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(a.repo_id)
        await node.start_download(b.repo_id)
        await node.wait_for(a.repo_id, DownloadCompleted, timeout=15, fail_fast=True)
        await node.wait_for(b.repo_id, DownloadCompleted, timeout=15, fail_fast=True)


async def test_two_downloads_started_together_do_not_both_take_the_same_space(
    hub: FakeHub, tmp_path: Path
) -> None:
    """Two models that each need 60% of the free space, started together: the
    second must be refused before it fetches anything (it may retry later, once
    the first is done)."""
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    b = hub.add_repo(FakeRepo("fake-org/b", model_repo_files("fake-org/b")))
    hub.bandwidth = 1_000_000
    need = sum(len(c) for c in expected_files(a).values())
    async with DownloadNode(
        hub,
        tmp_path,
        [card_for(a), card_for(b)],
        worker=False,
        timing=Timing(rescan_interval=60),
    ) as node:
        node.free_space[node.models_dir] = int(need / 0.6)
        await node.start_download(a.repo_id)
        await node.start_download(b.repo_id)
        first, second = await asyncio.wait_for(
            _first_failure_or_completion(node, a.repo_id, b.repo_id), 10
        )
        refused = node.failures(second)[0]
        assert "disk space" in refused.error_message
        assert hub.body_bytes(second) == 0, "the refused download fetched bytes"
        await node.wait_for(first, DownloadCompleted, timeout=10)


async def _first_failure_or_completion(
    node: DownloadNode, a: str, b: str
) -> tuple[str, str]:
    """(the model that got the space, the one refused)."""
    while True:
        for winner, loser in ((a, b), (b, a)):
            if node.failures(loser):
                return winner, loser
        await asyncio.sleep(0.01)


# --- legacy dirs ------------------------------------------------------------------------


async def test_offline_model_left_complete_by_older_exo_with_a_stray_partial(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """Older exo saw a model as complete when its index and weights were there; a
    leftover ``.partial`` of some small file didn't matter. Offline, such a model
    must still be usable."""
    model_dir = tmp_path / "models" / ModelId(REPO).normalize()
    model_dir.mkdir(parents=True)
    for path, content in expected_files(repo).items():
        (model_dir / path).write_bytes(content)
    (model_dir / "README.md.partial").write_bytes(b"# half")
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, offline=True
    ) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, (DownloadCompleted, DownloadFailed), timeout=5)
        assert isinstance(node.status(REPO), DownloadCompleted), node.status(REPO)


async def test_one_unreadable_model_dir_does_not_stop_the_rescan_of_others(
    hub: FakeHub, tmp_path: Path
) -> None:
    """A partly downloaded model dir with a folder lacking its x bit (say after a
    `chmod -R 600`) must only affect that model: the rescan must still announce
    the models after it."""
    bad = hub.add_repo(FakeRepo("aaa-org/bad", model_repo_files("aaa-org/bad")))
    good = hub.add_repo(FakeRepo("zzz-org/good", model_repo_files("zzz-org/good")))
    good_dir = tmp_path / "models" / ModelId(good.repo_id).normalize()
    good_dir.mkdir(parents=True)
    for path, content in expected_files(good).items():
        (good_dir / path).write_bytes(content)
    bad_dir = tmp_path / "models" / ModelId(bad.repo_id).normalize()
    (bad_dir / "sub").mkdir(parents=True)
    (bad_dir / "sub" / "file").write_bytes(b"x")
    (bad_dir / "sub").chmod(0o600)
    try:
        async with DownloadNode(
            hub,
            tmp_path,
            [card_for(bad), card_for(good)],
            worker=False,
            timing=Timing(rescan_interval=0.2),
        ) as node:
            await node.wait_for(good.repo_id, DownloadCompleted, timeout=3)
    finally:
        (bad_dir / "sub").chmod(0o700)


async def test_start_of_a_model_whose_dir_has_an_unsearchable_folder(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """A partly downloaded model dir with a folder lacking its x bit (say after a
    `chmod -R 600`). StartDownload for it may fail, but must not kill the
    coordinator."""
    other = hub.add_repo(FakeRepo(OTHER, model_repo_files(OTHER)))
    model_dir = tmp_path / "models" / ModelId(REPO).normalize()
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "sub" / "file").write_bytes(b"x")
    (model_dir / "config.json").write_bytes(expected_files(repo)["config.json"])
    (model_dir / "sub").chmod(0o600)
    try:
        async with DownloadNode(
            hub,
            tmp_path,
            [card_for(repo), card_for(other)],
            worker=False,
            timing=Timing(rescan_interval=60),
        ) as node:
            await node.start_download(REPO)
            await asyncio.sleep(0.3)
            assert coordinator_crash(node) is None, (
                f"coordinator died: {coordinator_crash(node)!r}"
            )
            await node.start_download(OTHER)
            await node.wait_for(OTHER, DownloadCompleted, timeout=5)
    finally:
        (model_dir / "sub").chmod(0o700)


async def test_no_file_descriptors_leak_across_cancels(
    hub: FakeHub, tmp_path: Path
) -> None:
    repo = hub.add_repo(
        FakeRepo(REPO, model_repo_files(REPO, shards=8, shard_size=4_000_000))
    )
    hub.bandwidth = 4_000_000
    before = len(os.listdir("/dev/fd"))
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        for _ in range(20):
            await node.start_download(REPO)
            await asyncio.sleep(0.05)
            await node.cancel(REPO)
            await node.wait_until_idle(REPO, timeout=5)
    await asyncio.sleep(0.2)
    assert len(os.listdir("/dev/fd")) <= before + 2


# --- pinned commits, leftovers, outages -----------------------------------------------

SHARD1 = "model-00001-of-00002.safetensors"


async def test_pinned_commit_force_pushed_away_mid_download_recovers(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """The commit a download pinned disappears upstream (a force push) while a file
    is being resumed: exo pins main again and finishes with the new commit's bytes."""
    from exo.download.tests.fake_hub import Fault, Reset

    hub.bandwidth = 400_000
    hub.inject(Fault(Reset(after_bytes=100_000), path=SHARD1))

    def force_push() -> None:
        repo.commit({SHARD1: content_for("rewritten", 200_000)})
        repo.commits = repo.commits[-1:]  # history rewritten: the old commit is gone

    hub.when_sent(SHARD1, 90_000, force_push)
    async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=15, fail_fast=True)
        assert node.files_on_disk(REPO) == expected_files(repo)
        assert node.partial_files() == []


async def test_power_loss_garbage_in_a_partial_does_not_use_up_the_integrity_budget(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """After a power cut a .partial can hold zeros where data was never flushed (or
    older exo left a .partial of another revision). That costs one refetch of the
    file; one bad response after that must not fail the download for good:
    IntegrityError is never retried automatically and workers don't re-request a
    DownloadFailed, so the instance waits for a human. Main retried 5 times."""
    from exo.download.tests.fake_hub import Corrupt, Fault

    model_dir = tmp_path / "models" / ModelId(REPO).normalize()
    model_dir.mkdir(parents=True)
    good = expected_files(repo)[SHARD1]
    (model_dir / (SHARD1 + ".partial")).write_bytes(
        good[:100_000] + bytes(len(good) - 100_000)
    )
    hub.inject(Fault(Corrupt(offset=5), path=SHARD1))
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=10)


async def test_an_unreachable_hub_is_reported_before_the_give_up_time(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """With Hugging Face unreachable, users should learn why nothing happens well
    before give_up_after (30 min in production; 18 s at this timing). Instead the
    cluster sees DownloadOngoing 0/0 bytes (which the dashboard's main page skips)
    and the reason (Progress.last_error) is never sent. Main was no better: it
    flip-flopped between Ongoing 0/0 and Pending without a reason."""
    await hub.go_down("refuse")
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(REPO)
        await asyncio.sleep(3)  # ~5 production minutes at this timing scale
        status = node.status(REPO)
        assert isinstance(status, DownloadFailed), (
            f"after ~5 production minutes the cluster sees {type(status).__name__}"
            + (
                f" {status.download_progress.downloaded.in_bytes}/"
                f"{status.download_progress.total.in_bytes}"
                if isinstance(status, DownloadOngoing)
                else ""
            )
        )


async def test_cancel_while_the_hub_is_a_black_hole_is_prompt(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    await hub.go_down("blackhole")
    async with DownloadNode(
        hub,
        tmp_path,
        [card_for(repo)],
        worker=False,
        timing=Timing(metadata_timeout=30, stall_timeout=30),
    ) as node:
        await node.start_download(REPO)
        await asyncio.sleep(0.3)
        await node.cancel(REPO)
        await node.wait_for_unlisted(REPO, timeout=1)
        await node.delete(REPO)
        await asyncio.sleep(0.2)
        assert coordinator_crash(node) is None


async def test_a_partial_download_in_a_dir_that_filled_up_moves_to_one_with_room(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """Main: a dir holding part of the model is used only if the rest fits there;
    otherwise the download goes to another models dir with room."""
    second = tmp_path / "second"
    second.mkdir()
    async with DownloadNode(
        hub,
        tmp_path,
        [card_for(repo)],
        worker=False,
        extra_dirs=(second,),
        timing=Timing(rescan_interval=60),
    ) as node:
        node.write_budget = 150_000  # the first dir fills up part way
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        await node.cancel(REPO)
        await node.wait_until_idle(REPO)
        node.write_budget = None
        node.free_space[node.models_dir] = 1_000  # still full
        await node.start_download(REPO)
        status = await node.wait_for(
            REPO, (DownloadCompleted, DownloadFailed), timeout=10
        )
        assert isinstance(status, DownloadCompleted), status


async def test_a_model_with_separate_vision_weights_downloads_both(
    hub: FakeHub, tmp_path: Path
) -> None:
    vision = "fake-org/vision"
    hub.add_repo(
        FakeRepo(
            vision,
            {
                "config.json": b'{"model_type": "fake"}',
                "README.md": b"# not wanted",
                "vision.safetensors": content_for("vision", 300_000),
            },
        )
    )
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    async with DownloadNode(
        hub,
        tmp_path,
        [vision_card(a.repo_id, vision, 1)],
        worker=False,
        timing=Timing(rescan_interval=0.2),
    ) as node:
        await node.start_download(a.repo_id)
        await node.wait_for(a.repo_id, DownloadCompleted, timeout=10, fail_fast=True)
        assert set(node.files_on_disk(vision)) == {"config.json", "vision.safetensors"}
        ongoing = [
            s for s in node.statuses(a.repo_id) if isinstance(s, DownloadOngoing)
        ]
        assert any(
            f"{vision}/vision.safetensors" in s.download_progress.files for s in ongoing
        )
        await asyncio.sleep(0.6)  # rescans keep it complete
        assert isinstance(node.status(a.repo_id), DownloadCompleted)


async def test_offline_legacy_model_with_separate_vision_weights_is_complete(
    hub: FakeHub, tmp_path: Path
) -> None:
    """A Kimi-style model (separate vision weights repo) downloaded by older exo:
    the main dir has its index and weights, the vision dir has what older exo
    fetched for it (``*.safetensors`` and ``config.json``, no index). Main reported
    it complete, offline too (from its cached file list). It must stay usable
    without the network."""
    vision = "fake-org/vision"
    vision_files = {
        "config.json": b'{"model_type": "fake"}',
        "vision.safetensors": content_for("vision", 300_000),
    }
    hub.add_repo(FakeRepo(vision, vision_files))
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    models = tmp_path / "models"
    for repo_id, files in (
        (a.repo_id, expected_files(a)),
        (vision, vision_files),
    ):
        directory = models / ModelId(repo_id).normalize()
        directory.mkdir(parents=True)
        for path, content in files.items():
            (directory / path).write_bytes(content)
    async with DownloadNode(
        hub,
        tmp_path,
        [vision_card(a.repo_id, vision, 1)],
        worker=False,
        offline=True,
    ) as node:
        await node.start_download(a.repo_id)
        status = await node.wait_for(
            a.repo_id, (DownloadCompleted, DownloadFailed), timeout=5
        )
        assert isinstance(status, DownloadCompleted), status


async def test_online_legacy_model_with_separate_vision_weights_is_adopted_without_refetching(
    hub: FakeHub, tmp_path: Path
) -> None:
    """Same model online: before any request it is shown as Pending (it was
    Completed on main), and a start adopts it with metadata requests only."""
    vision = "fake-org/vision"
    vision_files = {
        "config.json": b'{"model_type": "fake"}',
        "vision.safetensors": content_for("vision", 300_000),
    }
    hub.add_repo(FakeRepo(vision, vision_files))
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    models = tmp_path / "models"
    for repo_id, files in ((a.repo_id, expected_files(a)), (vision, vision_files)):
        directory = models / ModelId(repo_id).normalize()
        directory.mkdir(parents=True)
        for path, content in files.items():
            (directory / path).write_bytes(content)
    async with DownloadNode(
        hub, tmp_path, [vision_card(a.repo_id, vision, 1)], worker=False
    ) as node:
        await asyncio.sleep(0.3)  # the first rescan
        first = node.status(a.repo_id)
        await node.start_download(a.repo_id)
        await node.wait_for(a.repo_id, DownloadCompleted, timeout=5, fail_fast=True)
        assert hub.body_bytes() == 0
        assert isinstance(first, DownloadCompleted), (
            f"a complete model was first shown as {type(first).__name__}"
        )


# ===== second review pass =====


# --- hash failures that are never counted -------------------------------------------------


async def test_a_file_that_is_always_wrong_fails_even_on_a_flaky_link(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """A server that always sends the wrong bytes for a file (a broken mirror, a
    stale CDN object) on a link that drops each fresh request part way. Every
    attempt that completes the file is a resume, so its mismatch is never counted
    (``fetched_whole`` is only true for attempts from byte 0): the file is fetched
    again and again forever, the status stays Ongoing and never says why."""
    hub.bandwidth = 2_000_000
    # (the fake Hub applies the first matching fault: fresh requests are cut, resumed
    # ones carry the bad byte)
    hub.inject(Fault(Reset(after_bytes=100_000), path=SHARD1, ranged=False, times=None))
    hub.inject(Fault(Corrupt(offset=200_000), path=SHARD1, times=None))
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(REPO)
        await asyncio.sleep(6)
        fetched = hub.body_bytes(REPO, SHARD1)
        assert isinstance(node.status(REPO), DownloadFailed), (
            f"still {type(node.status(REPO)).__name__} after fetching {SHARD1}"
            f" {fetched // len(expected_files(repo)[SHARD1])} times"
        )


# --- older exo's vision folders ------------------------------------------------------------


async def test_a_half_downloaded_legacy_vision_folder_is_not_complete(
    hub: FakeHub, tmp_path: Path
) -> None:
    """Older exo interrupted while fetching a vision repo with two weight files:
    ``config.json`` and the first file are there, the second is a ``.partial``.
    The model isn't complete; offline, it must not be reported Completed."""
    vision = "fake-org/vision"
    vision_files = {
        "config.json": b'{"model_type": "fake"}',
        "vision-1.safetensors": content_for("v1", 100_000),
        "vision-2.safetensors": content_for("v2", 100_000),
    }
    hub.add_repo(FakeRepo(vision, vision_files))
    a = hub.add_repo(FakeRepo("fake-org/a", model_repo_files("fake-org/a")))
    models = tmp_path / "models"
    main_dir = models / ModelId(a.repo_id).normalize()
    main_dir.mkdir(parents=True)
    for path, content in expected_files(a).items():
        (main_dir / path).write_bytes(content)
    vision_dir = models / ModelId(vision).normalize()
    vision_dir.mkdir(parents=True)
    (vision_dir / "config.json").write_bytes(vision_files["config.json"])
    (vision_dir / "vision-1.safetensors").write_bytes(
        vision_files["vision-1.safetensors"]
    )
    (vision_dir / "vision-2.safetensors.partial").write_bytes(
        vision_files["vision-2.safetensors"][:10_000]
    )
    async with DownloadNode(
        hub, tmp_path, [vision_card(a.repo_id, vision, 1)], worker=False, offline=True
    ) as node:
        await node.start_download(a.repo_id)
        status = await node.wait_for(
            a.repo_id, (DownloadCompleted, DownloadFailed), timeout=5
        )
        assert isinstance(status, DownloadFailed), (
            "a vision folder missing a weight file was reported complete"
        )


# --- proxies from the environment ----------------------------------------------------------


async def test_a_netrc_entry_does_not_break_downloads_with_a_token(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """``trust_env=True`` (for the proxy variables) also makes aiohttp read
    ~/.netrc. With HF_TOKEN set and a netrc entry for the Hub's host, aiohttp
    refuses every request ("Cannot combine AUTHORIZATION header with AUTH
    argument"), which surfaces as "Download failed unexpectedly: ValueError."
    git and curl read the same file, so such entries exist."""
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login someone password something\n")
    netrc.chmod(0o600)
    saved = {k: os.environ.get(k) for k in ("NETRC", "HF_TOKEN")}
    try:
        async with DownloadNode(hub, tmp_path, [card_for(repo)], worker=False) as node:
            os.environ["NETRC"] = str(netrc)
            os.environ["HF_TOKEN"] = "hf_token_for_the_test"
            await node.start_download(REPO)
            status = await node.wait_for(
                REPO, (DownloadCompleted, DownloadFailed), timeout=10
            )
            assert isinstance(status, DownloadCompleted), status
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- model card fetches ------------------------------------------------------------------


async def test_fetching_one_card_file_gives_up_like_before(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """``download_file_with_retry`` backs API calls (adding a custom model, loading
    an unknown card). Main gave up after 3 tries (seconds for a 5xx); it now
    retries a file for give_up_after (30 minutes in production) of no progress,
    so the HTTP request waits that long."""
    hub.inject(
        Fault(server_error(503), where="resolve", path="config.json", times=None)
    )
    target = tmp_path / "card-dir"
    target.mkdir()
    saved = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = hub.url
    try:
        with pytest.raises(Exception) as raised:  # noqa: B017 - any error, not a hang
            await asyncio.wait_for(
                download_file_with_retry(ModelId(REPO), "main", "config.json", target),
                15,
            )
        assert not isinstance(raised.value, TimeoutError), (
            "still retrying after 15 s (main gave up within seconds)"
        )
    finally:
        if saved is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = saved


# --- the window before the files are listed ---------------------------------------------


async def test_deleting_the_instance_while_the_files_are_listed_stops_the_download(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """The master cancels downloads no instance needs, but only those it sees as
    DownloadOngoing. The coordinator now keeps the old status (Pending, or Failed
    for a retry) until the files are listed, so an instance deleted in that window
    leaves a download running for nobody. (On faf9614d the window was at most the
    1 s progress interval; on main it was the first file-list fetch.)"""
    hub.inject(Fault(FirstByteDelay(1.0), where="tree"))  # a slow listing
    async with DownloadNode(
        hub,
        tmp_path,
        [card_for(repo)],
        worker=False,
        timing=Timing(rescan_interval=60, metadata_timeout=3),
    ) as node:
        await node.start_download(REPO)
        await asyncio.sleep(0.5)  # listing the files
        for command in cancel_unnecessary_downloads({}, node.state.downloads):
            await node.send(command)
        await asyncio.sleep(3)
        assert hub.body_bytes(REPO) == 0, (
            f"{hub.body_bytes(REPO)} bytes fetched for a deleted instance;"
            f" status {type(node.status(REPO)).__name__}"
        )


# --- file I/O under cancel (expected to pass: records what was verified) ------------------


async def test_delete_during_slow_writes_leaves_nothing_behind(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """Writes take 0.2 s each (a slow disk). A delete mid-download waits for the
    writes in flight, and nothing is written, created or reopened after it."""
    import time as _time

    from exo.download import transfer

    hub.bandwidth = 2_000_000
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        patched = transfer._write_at  # pyright: ignore[reportPrivateUsage]

        def slow_write(
            descriptor: int, data: bytes, offset: int, state: transfer.FileState
        ) -> None:
            _time.sleep(0.2)
            patched(descriptor, data, offset, state)

        transfer._write_at = slow_write  # pyright: ignore[reportPrivateUsage]
        try:
            before = len(os.listdir("/dev/fd"))
            await node.start_download(REPO)
            await asyncio.sleep(0.5)
            await node.delete(REPO)
            await node.wait_for_unlisted(REPO, timeout=5)
            await asyncio.sleep(1.0)
            assert node.model_dirs(REPO) == [], "files came back after the delete"
            assert len(os.listdir("/dev/fd")) <= before + 2
        finally:
            transfer._write_at = patched  # pyright: ignore[reportPrivateUsage]


# --- resuming with the Hub unreachable ------------------------------------------------------------------


async def test_an_unreachable_hub_is_reported_when_resuming_too(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """ "Can't reach Hugging Face" after 2 minutes only covers listing the files. A
    download resumed from its manifest (after a restart, or a pause) needs no
    listing, so with the Hub unreachable it shows Ongoing, with no reason, until
    give_up_after (30 minutes in production; 18 s here)."""
    hub.bandwidth = 300_000
    async with DownloadNode(
        hub, tmp_path, [card_for(repo)], worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(REPO)
        reached = asyncio.Event()
        hub.when_sent(SHARD1, 50_000, reached.set)
        await reached.wait()
        await node.cancel(REPO)
        await node.wait_until_idle(REPO)
        await hub.go_down("refuse")
        await node.start_download(REPO)
        await asyncio.sleep(3)  # ~5 production minutes at this timing scale
        assert isinstance(node.status(REPO), DownloadFailed), (
            f"after ~5 production minutes: {type(node.status(REPO)).__name__}"
        )
