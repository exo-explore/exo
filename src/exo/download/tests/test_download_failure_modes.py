"""Download failure modes, end to end: exo's real download stack against a fake Hub.

Each scenario checks what a user would notice: whether the download finishes,
whether the bytes on disk are right, how many bytes were fetched twice, what
status the cluster sees (and that it is never stuck), and that no ``.partial``
files are left behind.

Set ``EXO_DOWNLOAD_SCENARIO_REPORT`` to a file path to get one JSON line per
scenario with those measurements (run with ``--runxfail`` to measure every one,
and ``--junitxml`` for the pass/fail of each).
"""

import asyncio
import json
import os
import stat
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast

import pytest

from exo.download.tests.fake_hub import (
    Corrupt,
    FakeHub,
    FakeRepo,
    Fault,
    FirstByteDelay,
    IgnoreRange,
    RangeNotSatisfiable,
    Reset,
    ShortBody,
    Stall,
    Throttle,
    content_for,
    model_repo_files,
    rate_limited,
    server_error,
)
from exo.download.tests.harness import (
    DownloadNode,
    Timing,
    card_for,
    expected_files,
    shard_for,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
)

REPO = "fake-org/tiny"
SHARD1 = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
SHARD_SIZE = 256 * 1024
# Bytes a resumed download may fetch twice: what was in flight when it broke.
IN_FLIGHT = 128 * 1024


@pytest.fixture
async def hub() -> AsyncIterator[FakeHub]:
    async with FakeHub() as hub:
        yield hub


@pytest.fixture
def repo(hub: FakeHub) -> FakeRepo:
    return hub.add_repo(FakeRepo(REPO, model_repo_files(REPO, shard_size=SHARD_SIZE)))


_nodes: list[DownloadNode] = []


def node_for(
    hub: FakeHub,
    repo: FakeRepo,
    tmp_path: Path,
    *others: FakeRepo,
    **kwargs: object,
) -> DownloadNode:
    cards = [card_for(r) for r in (repo, *others)]
    node = DownloadNode(hub, tmp_path, cards, **kwargs)  # pyright: ignore[reportArgumentType]
    _nodes.append(node)
    return node


@pytest.fixture(autouse=True)
def scenario_report(
    request: pytest.FixtureRequest, hub: FakeHub, tmp_path: Path
) -> Iterator[None]:
    _nodes.clear()
    started = time.monotonic()
    yield
    path = os.environ.get("EXO_DOWNLOAD_SCENARIO_REPORT")
    if not path or not _nodes:
        return
    node = _nodes[-1]
    repo = hub.repos.get(REPO)
    statuses = node.statuses(REPO)
    failures = [s for s in statuses if isinstance(s, DownloadFailed)]
    completed = any(isinstance(s, DownloadCompleted) for s in statuses)
    model_dirs = node.model_dirs(REPO)
    on_disk = node.files_on_disk(REPO, model_dirs[0].parent) if model_dirs else {}
    commits = (
        [] if repo is None else [expected_files(repo, c.sha) for c in repo.commits]
    )
    line: dict[str, object] = {
        "test": cast(pytest.Item, request.node).name,
        "seconds": round(time.monotonic() - started, 2),
        "completed": completed,
        "bytes_correct": completed and on_disk in commits,
        "refetched_bytes": None if repo is None else refetched(hub, repo),
        "failed_reports": len(failures),
        "first_error": failures[0].error_message[:160] if failures else None,
        "final_status": type(node.status(REPO)).__name__,
        "partial_files": len(
            [p for p in tmp_path.rglob("*") if p.name.endswith(".partial")]
        ),
    }
    with open(path, "a") as report_file:
        report_file.write(json.dumps(line) + "\n")


def total_bytes(repo: FakeRepo) -> int:
    return sum(len(c) for c in expected_files(repo).values())


def refetched(hub: FakeHub, repo: FakeRepo) -> int:
    return hub.body_bytes(REPO) - total_bytes(repo)


def sent(hub: FakeHub, path: str, after_bytes: int) -> asyncio.Event:
    reached = asyncio.Event()
    hub.when_sent(path, after_bytes, reached.set)
    return reached


async def assert_complete(
    node: DownloadNode, repo: FakeRepo, timeout: float = 10, *, may_fail: bool = False
) -> None:
    """The download completes with the right bytes and no .partial files left.

    Unless ``may_fail``, it must never be reported failed on the way.
    """
    status = await node.wait_for(
        REPO, DownloadCompleted, timeout=timeout, fail_fast=not may_fail
    )
    assert isinstance(status, DownloadCompleted)
    assert node.files_on_disk(REPO) == expected_files(repo)
    assert node.partial_files() == []


def put_files(directory: Path, files: dict[str, bytes]) -> None:
    for path, content in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(content)


# --- the baseline ------------------------------------------------------------------


async def test_clean_download(hub: FakeHub, repo: FakeRepo, tmp_path: Path) -> None:
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) == 0
        assert node.failures(REPO) == []
        status = node.status(REPO)
        assert isinstance(status, DownloadCompleted)
        assert status.total.in_bytes == total_bytes(repo)
        assert status.model_directory == str(
            node.models_dir / ModelId(REPO).normalize()
        )


# --- network faults ------------------------------------------------------------------


async def test_reset_mid_file_resumes(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Reset(after_bytes=100_000), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) <= IN_FLIGHT
        assert any(
            r.file_path == SHARD1 and (r.range_start or 0) > 0 for r in hub.requests
        ), "the second request should resume with a Range"
        assert node.failures(REPO) == []


async def test_short_body_resumes(hub: FakeHub, repo: FakeRepo, tmp_path: Path) -> None:
    hub.inject(Fault(ShortBody(after_bytes=100_000), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) <= IN_FLIGHT


async def test_repeated_resets_with_progress_never_fail(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Reset(after_bytes=20_000), path=SHARD1, times=10))
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await assert_complete(node, repo, timeout=15)
        assert node.failures(REPO) == []


@pytest.mark.parametrize("seconds", [0.1, 0.4])
async def test_stall_shorter_than_the_idle_timeout_is_waited_out(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path, seconds: float
) -> None:
    hub.inject(Fault(Stall(after_bytes=10_000, seconds=seconds), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) == 0
        assert hub.count(kind="cdn", path=SHARD1) == 1


@pytest.mark.parametrize("seconds", [1.5, None])
async def test_stall_longer_than_the_idle_timeout_is_cut_and_resumed(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path, seconds: float | None
) -> None:
    hub.inject(Fault(Stall(after_bytes=100_000, seconds=seconds), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert hub.count(kind="cdn", path=SHARD1) == 2
        assert refetched(hub, repo) <= IN_FLIGHT


async def test_repeated_stalls_never_fail(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Stall(after_bytes=0), path=SHARD1, times=6))
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await assert_complete(node, repo, timeout=20)
        assert node.failures(REPO) == []


async def test_slow_first_byte_is_waited_out(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(FirstByteDelay(0.3), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert hub.count(kind="cdn", path=SHARD1) == 1


async def test_slow_link_longer_than_the_request_timeout(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Throttle(64 * 1024), path=SHARD1, times=None))
    timing = Timing(file_request_timeout=0.5)
    async with node_for(hub, repo, tmp_path, timing=timing) as node:
        node.want(REPO)
        await assert_complete(node, repo, timeout=20)
        assert node.failures(REPO) == []
        assert refetched(hub, repo) <= IN_FLIGHT * 2


async def test_flapping_link(hub: FakeHub, repo: FakeRepo, tmp_path: Path) -> None:
    hub.bandwidth = 200_000
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await sent(hub, SHARD1, 50_000).wait()
        for _ in range(4):
            await hub.go_down()
            await asyncio.sleep(0.4)
            await hub.come_up()
            await asyncio.sleep(0.3)
        await assert_complete(node, repo, timeout=20)
        assert node.failures(REPO) == []
        assert refetched(hub, repo) <= 4 * IN_FLIGHT


async def test_rate_limit_burst_is_waited_out(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(rate_limited(20), where="head", path=SHARD1, times=3))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert node.failures(REPO) == []


async def test_long_rate_limit_burst_is_waited_out(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(rate_limited(5), where="head", path=SHARD1, times=8))
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await assert_complete(node, repo, timeout=15)
        assert node.failures(REPO) == []


async def test_server_error_burst(hub: FakeHub, repo: FakeRepo, tmp_path: Path) -> None:
    hub.inject(Fault(server_error(503), path=SHARD1, times=3))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert node.failures(REPO) == []


async def test_long_server_error_burst(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(server_error(502), path=SHARD1, times=8))
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await assert_complete(node, repo, timeout=15)
        assert node.failures(REPO) == []


async def test_endpoint_down_then_back(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    # an outage of a few minutes shows as Failed with the reason; asking again
    # (an instance needing it, or the user) completes it once the Hub is back
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await hub.go_down()
        await node.start_download(REPO)
        await asyncio.sleep(2)
        await hub.come_up()
        failed = await node.wait_for_failure(REPO)
        assert failed.error_message.startswith("Can't reach Hugging Face")
        await node.start_download(REPO)
        await assert_complete(node, repo, timeout=6, may_fail=True)


async def test_short_outage_at_the_start_is_not_a_failure(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await hub.go_down()
        await node.start_download(REPO)
        await asyncio.sleep(0.5)
        await hub.come_up()
        await assert_complete(node, repo, timeout=6)


async def test_endpoint_down_then_back_with_worker_retrying(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path) as node:
        await hub.go_down()
        node.want(REPO)
        await asyncio.sleep(2)
        await hub.come_up()
        await assert_complete(node, repo, timeout=15, may_fail=True)


async def test_link_dies_mid_download_then_returns(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await sent(hub, SHARD1, 100_000).wait()
        await hub.go_down("blackhole")
        await asyncio.sleep(5)
        await hub.come_up()
        await assert_complete(node, repo, timeout=20)
        assert node.failures(REPO) == []
        assert refetched(hub, repo) <= 2 * IN_FLIGHT


# --- process lifecycle --------------------------------------------------------------


async def test_restart_mid_download_resumes_from_partial(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 400_000
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await sent(hub, SHARD1, 150_000).wait()
        await node.restart()
        await assert_complete(node, repo, timeout=15)
        assert refetched(hub, repo) <= 2 * IN_FLIGHT
        assert any(
            r.file_path == SHARD1 and (r.range_start or 0) >= 100_000
            for r in hub.requests
        ), "the restarted node should resume from the .partial file"


async def test_cancel_stops_the_download_and_start_resumes(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await sent(hub, SHARD1, 100_000).wait()
        await node.cancel(REPO)
        await node.wait_for(REPO, DownloadPending, timeout=5)
        await asyncio.sleep(0.2)
        bytes_after_cancel = hub.body_bytes(REPO)
        await asyncio.sleep(1.5)
        assert hub.body_bytes(REPO) == bytes_after_cancel, (
            "bytes kept flowing after cancel"
        )
        status = node.status(REPO)
        assert isinstance(status, DownloadPending), (
            f"cancelled download came back as {status}"
        )
        assert status.downloaded.in_bytes > 0
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) <= 2 * IN_FLIGHT


async def test_cancel_then_start_repeatedly(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 600_000
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        for round_ in range(4):
            await node.start_download(REPO)
            await sent(hub, SHARD1, 40_000 * (round_ + 1)).wait()
            await node.cancel(REPO)
            await node.wait_for(REPO, DownloadPending, timeout=5)
        await node.start_download(REPO)
        await assert_complete(node, repo)


async def test_delete_during_download_removes_everything(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await sent(hub, SHARD1, 100_000).wait()
        await node.delete(REPO)
        await node.wait_for_unlisted(REPO, timeout=5)
        await asyncio.sleep(1.0)
        bytes_after_delete = hub.body_bytes(REPO)
        await asyncio.sleep(1.0)
        assert hub.body_bytes(REPO) == bytes_after_delete, (
            "bytes kept flowing after delete"
        )
        assert node.model_dirs(REPO) == [], "files are back after the delete"
        assert not isinstance(node.status(REPO), DownloadOngoing)
        await node.start_download(REPO)
        await assert_complete(node, repo)


async def test_duplicate_start_download_fetches_once(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await node.start_download(REPO)
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) == 0


async def test_two_shards_of_one_model_download_once(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    card = card_for(repo)
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO, shard_for(card, 0, 1))
        await node.start_download(REPO, shard_for(card, 1, 2))
        await assert_complete(node, repo)
        assert refetched(hub, repo) == 0


async def test_one_slow_model_does_not_hold_up_another(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    other = hub.add_repo(FakeRepo("fake-org/other", model_repo_files("fake-org/other")))
    hub.inject(Fault(Stall(), where="tree", repo=REPO, times=None))
    timing = Timing(metadata_timeout=3)
    async with node_for(
        hub, repo, tmp_path, other, worker=False, timing=timing
    ) as node:
        await node.start_download(REPO)
        await asyncio.sleep(0.1)
        await node.start_download(other.repo_id)
        await node.wait_for(other.repo_id, DownloadCompleted, timeout=2)


async def test_master_change_mid_download_resumes(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    """The node's download side restarts mid-download (exo did this on every new
    master until the coordinator was kept across elections); it resumes."""
    hub.bandwidth = 400_000
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        for after in (60_000, 120_000):
            await sent(hub, SHARD1, after).wait()
            await node.restart()
        await assert_complete(node, repo, timeout=15)
        assert refetched(hub, repo) <= 3 * IN_FLIGHT


# --- disk ------------------------------------------------------------------------------


async def test_disk_full_fails_with_the_reason(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        node.write_budget = 200_000
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert "space" in failed.error_message.lower()


async def test_start_download_retries_a_failed_download(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    # a slow link keeps the bytes in flight when the disk fills small
    hub.bandwidth = 1_000_000
    async with node_for(
        hub, repo, tmp_path, worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        node.write_budget = 200_000
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadFailed)
        node.write_budget = None
        await node.start_download(REPO)
        await assert_complete(node, repo, timeout=5, may_fail=True)
        assert refetched(hub, repo) <= 2 * IN_FLIGHT


async def test_failure_reason_stays_visible(hub: FakeHub, tmp_path: Path) -> None:
    repo = hub.add_repo(FakeRepo(REPO, model_repo_files(REPO), gated=True))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await node.wait_for_failure(REPO)
        await asyncio.sleep(3 * node.timing.rescan_interval)
        # it may be retried, but it is never shown as merely pending
        after_failure = node.statuses(REPO)[
            node.statuses(REPO).index(node.failures(REPO)[0]) :
        ]
        assert not any(isinstance(s, DownloadPending) for s in after_failure)
        await node.wait_for(REPO, DownloadFailed, timeout=2)


async def test_not_enough_space_fails_before_fetching(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        node.free_space[node.models_dir] = 1000
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert (
            "free" in failed.error_message.lower()
            or "space" in failed.error_message.lower()
        )
        assert hub.body_bytes(REPO) == 0


@pytest.fixture
def read_only_dir(tmp_path: Path) -> Iterator[Path]:
    directory = tmp_path / "read-only"
    directory.mkdir()
    yield directory
    for path in [directory, *directory.rglob("*")]:
        if path.is_dir():
            path.chmod(stat.S_IRWXU)


def make_read_only(directory: Path) -> None:
    for path in [*directory.rglob("*"), directory]:
        path.chmod(0o555 if path.is_dir() else 0o444)


async def test_complete_model_in_read_only_dir_is_used(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path, read_only_dir: Path
) -> None:
    put_files(read_only_dir / ModelId(REPO).normalize(), expected_files(repo))
    make_read_only(read_only_dir)
    async with node_for(
        hub, repo, tmp_path, worker=False, read_only_dirs=(read_only_dir,)
    ) as node:
        await node.start_download(REPO)
        status = await node.wait_for(REPO, DownloadCompleted)
        assert isinstance(status, DownloadCompleted)
        assert status.read_only
        assert status.model_directory == str(read_only_dir / ModelId(REPO).normalize())
        assert hub.body_bytes() == 0


async def test_partial_model_in_read_only_dir_downloads_to_writable_dir(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path, read_only_dir: Path
) -> None:
    files = expected_files(repo)
    put_files(
        read_only_dir / ModelId(REPO).normalize(),
        {p: files[p] for p in ("config.json", "model.safetensors.index.json", SHARD1)},
    )
    make_read_only(read_only_dir)
    async with node_for(
        hub, repo, tmp_path, worker=False, read_only_dirs=(read_only_dir,)
    ) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        status = node.status(REPO)
        assert isinstance(status, DownloadCompleted) and not status.read_only


async def test_delete_never_touches_read_only_dir(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path, read_only_dir: Path
) -> None:
    put_files(read_only_dir / ModelId(REPO).normalize(), expected_files(repo))
    async with node_for(
        hub, repo, tmp_path, worker=False, read_only_dirs=(read_only_dir,)
    ) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        await node.delete(REPO)
        await asyncio.sleep(0.3)
        assert node.files_on_disk(REPO, read_only_dir) == expected_files(repo)


async def test_second_models_dir_is_used_when_the_first_is_full(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    second = tmp_path / "second"
    second.mkdir()
    async with node_for(
        hub, repo, tmp_path, worker=False, extra_dirs=(second,)
    ) as node:
        node.free_space[node.models_dir] = 1000
        await node.start_download(REPO)
        status = await node.wait_for(REPO, DownloadCompleted)
        assert node.files_on_disk(REPO, second) == expected_files(repo)
        assert isinstance(status, DownloadCompleted)
        assert status.model_directory == str(second / ModelId(REPO).normalize())


async def test_download_resumes_in_the_dir_holding_its_partial_files(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    second = tmp_path / "second"
    target = second / ModelId(REPO).normalize()
    put_files(target, {SHARD1 + ".partial": expected_files(repo)[SHARD1][:100_000]})
    async with node_for(
        hub, repo, tmp_path, worker=False, extra_dirs=(second,)
    ) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        assert node.files_on_disk(REPO, second) == expected_files(repo)
        assert refetched(hub, repo) == -100_000


# --- other -------------------------------------------------------------------------------


async def test_offline_with_the_model_on_disk(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    put_files(tmp_path / "models" / ModelId(REPO).normalize(), expected_files(repo))
    async with node_for(hub, repo, tmp_path, worker=False, offline=True) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, DownloadCompleted)
        assert hub.requests == []


async def test_offline_without_the_model_fails_without_network(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    async with node_for(hub, repo, tmp_path, worker=False, offline=True) as node:
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert "offline" in failed.error_message.lower()
        assert hub.requests == []


async def test_upstream_shrinks_a_file_mid_download(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    first_commit = repo.main.sha
    hub.inject(Fault(Reset(after_bytes=200_000), path=SHARD1))

    def push_smaller_shard() -> None:
        repo.commit({SHARD1: content_for("smaller", 50_000)})

    hub.when_sent(SHARD1, 190_000, push_smaller_shard)
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=15, fail_fast=True)
        on_disk = node.files_on_disk(REPO)
        assert on_disk in (expected_files(repo, first_commit), expected_files(repo))
        assert node.partial_files() == []


async def test_upstream_replaces_a_file_mid_download(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 300_000
    first_commit = repo.main.sha
    hub.inject(Fault(Reset(after_bytes=200_000), path=SHARD1))

    def push_new_shard() -> None:
        repo.commit({SHARD1: content_for("replaced", SHARD_SIZE + 10_000)})

    hub.when_sent(SHARD1, 190_000, push_new_shard)
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await node.wait_for(REPO, DownloadCompleted, timeout=15, fail_fast=True)
        on_disk = node.files_on_disk(REPO)
        assert on_disk in (expected_files(repo, first_commit), expected_files(repo))
        assert hub.body_bytes(REPO) - total_bytes(repo) <= 2 * IN_FLIGHT


async def test_upstream_change_never_mixes_revisions(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 400_000
    first_commit = repo.main.sha
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        await sent(hub, SHARD1, 150_000).wait()
        await node.stop()
        config = repo.main.files["config.json"].content
        repo.commit(
            {
                # same size: only the content changed
                "config.json": config.replace(
                    b'"hidden_size": 64', b'"hidden_size": 65'
                ),
                SHARD2: content_for("new shard 2", SHARD_SIZE),
            }
        )
        await node.start()
        await node.wait_for(REPO, DownloadCompleted, timeout=15, fail_fast=True)
        assert node.files_on_disk(REPO) in (
            expected_files(repo, first_commit),
            expected_files(repo),
        )


async def test_corrupt_body_once_refetches_that_file(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Corrupt(offset=1234), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) == SHARD_SIZE


async def test_always_corrupt_file_fails_with_the_reason(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Corrupt(offset=1234), path=SHARD1, times=None))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert "hash" in failed.error_message.lower()
        assert hub.count(kind="cdn", path=SHARD1) <= 5
        await node.quiet(0.5)
        assert node.partial_files() == []


async def test_a_failed_file_stops_the_whole_download(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 200_000
    hub.inject(Fault(Corrupt(offset=10), path="tokenizer.json", times=None))
    async with node_for(
        hub, repo, tmp_path, worker=False, timing=Timing(rescan_interval=60)
    ) as node:
        await node.start_download(REPO)
        await node.wait_for_failure(REPO)
        await asyncio.sleep(0.2)
        bytes_after_failure = hub.body_bytes(REPO)
        await asyncio.sleep(0.3)  # under a retry's backoff
        assert hub.body_bytes(REPO) == bytes_after_failure, "files kept downloading"
        assert isinstance(node.status(REPO), DownloadFailed)


async def test_index_naming_a_missing_file_fails(hub: FakeHub, tmp_path: Path) -> None:
    files = model_repo_files(REPO, shards=3, shard_size=50_000)
    del files["model-00003-of-00003.safetensors"]
    repo = hub.add_repo(FakeRepo(REPO, files))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await node.wait_for(REPO, (DownloadFailed, DownloadCompleted), timeout=10)
        failed = await node.wait_for_failure(REPO, timeout=0.1)
        assert "model-00003-of-00003.safetensors" in failed.error_message


async def test_a_missing_tokenizer_is_downloaded_not_reported_complete(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(Fault(Stall(), path="tokenizer.json", times=None))
    async with node_for(hub, repo, tmp_path) as node:
        node.want(REPO)
        model_dir = node.models_dir / ModelId(REPO).normalize()
        deadline = time.monotonic() + 10
        while not ((model_dir / SHARD1).exists() and (model_dir / SHARD2).exists()):
            assert time.monotonic() < deadline, "the weights never finished"
            await asyncio.sleep(0.02)
        await node.stop()
        hub.clear_faults()
        await node.start()
        await assert_complete(node, repo)


async def test_file_missing_upstream_fails_fast_and_says_so(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.inject(
        Fault(
            server_error(404),
            where="resolve",
            path="tokenizer.json",
            times=None,
        )
    )
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert "tokenizer.json" in failed.error_message
        assert "not found" in failed.error_message.lower()
        assert hub.count(kind="resolve", path="tokenizer.json") <= 2


async def test_gated_model_fails_with_auth_message(
    hub: FakeHub, tmp_path: Path
) -> None:
    repo = hub.add_repo(FakeRepo(REPO, model_repo_files(REPO), gated=True))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO)
        assert (
            "HF_TOKEN" in failed.error_message
            or "access" in failed.error_message.lower()
        )
        await node.quiet(0.3)
        assert hub.count(kind="resolve", path=SHARD1) <= 1


async def test_unknown_repo_fails_and_says_so(hub: FakeHub, tmp_path: Path) -> None:
    repo = FakeRepo(REPO, model_repo_files(REPO))  # not on the hub
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        failed = await node.wait_for_failure(REPO, timeout=5)
        assert REPO in failed.error_message


async def test_server_ignoring_range_restarts_the_file_cleanly(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    target = tmp_path / "models" / ModelId(REPO).normalize()
    put_files(target, {SHARD1 + ".partial": expected_files(repo)[SHARD1][:100_000]})
    hub.inject(Fault(IgnoreRange(), path=SHARD1, times=None))
    hub.bandwidth = 1_000_000
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert refetched(hub, repo) <= 0, "fetched more than one copy of the file"
        for status in node.statuses(REPO):
            if isinstance(status, DownloadOngoing):
                progress = status.download_progress
                assert progress.downloaded.in_bytes <= progress.total.in_bytes


async def test_oversized_partial_is_discarded(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    target = tmp_path / "models" / ModelId(REPO).normalize()
    put_files(target, {SHARD1 + ".partial": content_for("old", SHARD_SIZE + 5000)})
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)


async def test_416_on_resume_recovers(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    target = tmp_path / "models" / ModelId(REPO).normalize()
    put_files(target, {SHARD1 + ".partial": expected_files(repo)[SHARD1][:100_000]})
    hub.inject(Fault(RangeNotSatisfiable(), path=SHARD1))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)


async def test_progress_is_truthful_and_paced(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.bandwidth = 120_000
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        started = time.monotonic()
        await node.start_download(REPO)
        await assert_complete(node, repo)
        elapsed = time.monotonic() - started
        ongoing = [s for s in node.statuses(REPO) if isinstance(s, DownloadOngoing)]
        assert len(ongoing) >= 2, "no progress shown during a multi-second download"
        assert len(ongoing) <= elapsed * 3 + 2, "progress events are too frequent"
        downloaded = [s.download_progress.downloaded.in_bytes for s in ongoing]
        assert downloaded == sorted(downloaded)
        for status in ongoing:
            progress = status.download_progress
            assert progress.downloaded.in_bytes <= progress.total.in_bytes
            # 0 of 0 while the files are being listed, then the real total
            assert progress.total.in_bytes in (0, total_bytes(repo))
        assert ongoing[-1].download_progress.total.in_bytes == total_bytes(repo)


async def test_file_list_pages_are_all_followed(
    hub: FakeHub, repo: FakeRepo, tmp_path: Path
) -> None:
    hub.page_size = 3
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)


async def test_ignored_folders_are_not_downloaded(hub: FakeHub, tmp_path: Path) -> None:
    files = model_repo_files(REPO) | {
        "original/consolidated.pth": content_for("orig", 1000),
        "metal/model.bin": content_for("metal", 1000),
    }
    repo = hub.add_repo(FakeRepo(REPO, files))
    async with node_for(hub, repo, tmp_path, worker=False) as node:
        await node.start_download(REPO)
        await assert_complete(node, repo)
        assert hub.count(path="original/*") == 0 and hub.count(path="metal/*") == 0
