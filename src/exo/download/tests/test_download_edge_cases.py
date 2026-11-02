"""Edge cases from an independent review of the downloader: broken folders, links,
shared vision repos, racing downloads, odd listings, outages and leftovers."""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from exo.download.tests.fake_hub import (
    FakeHub,
    FakeRepo,
    content_for,
    model_repo_files,
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
    on DeleteInstance), but only those it sees as DownloadOngoing. A download that
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
            await node.wait_for(REPO, DownloadPending, timeout=5)
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
        await node.wait_for(REPO, DownloadPending, timeout=1)
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
        await node.wait_for(REPO, DownloadPending)
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
