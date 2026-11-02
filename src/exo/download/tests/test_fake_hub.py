"""The fake Hub behaves like the real one where exo depends on it."""

import asyncio
import hashlib
import json

import aiohttp
import pytest

from exo.download.tests.fake_hub import (
    Corrupt,
    FakeHub,
    FakeRepo,
    Fault,
    IgnoreRange,
    RangeNotSatisfiable,
    Reset,
    ShortBody,
    Stall,
    model_repo_files,
    rate_limited,
)

REPO = "fake-org/tiny"
SHARD = "model-00001-of-00002.safetensors"


@pytest.fixture
async def hub():
    async with FakeHub(page_size=4) as hub:
        hub.add_repo(FakeRepo(REPO, model_repo_files(REPO)))
        yield hub


async def test_tree_lists_files_with_lfs_and_paginates(hub: FakeHub) -> None:
    entries: list[dict[str, object]] = []
    url: str | None = f"{hub.url}/api/models/{REPO}/tree/main?recursive=true"
    async with aiohttp.ClientSession() as session:
        while url is not None:
            async with session.get(url) as response:
                assert response.status == 200
                entries += json.loads(await response.text())  # pyright: ignore[reportAny]
                url = response.links.get("next", {}).get("url")  # pyright: ignore[reportAssignmentType]
                url = str(url) if url is not None else None
    by_path = {str(e["path"]): e for e in entries}
    assert set(by_path) == set(hub.repos[REPO].main.files)
    assert "lfs" in by_path[SHARD]
    assert "lfs" not in by_path["config.json"]


async def test_resolve_redirects_lfs_to_cdn_with_linked_headers(hub: FakeHub) -> None:
    blob = hub.repos[REPO].main.files[SHARD]
    async with (
        aiohttp.ClientSession() as session,
        session.head(f"{hub.url}/{REPO}/resolve/main/{SHARD}") as head,
    ):
        assert head.status == 302
        assert head.headers["X-Linked-Size"] == str(blob.size)
        assert head.headers["X-Linked-Etag"] == f'"{blob.sha256}"'
        assert head.headers["X-Repo-Commit"] == hub.repos[REPO].main.sha
    async with (
        aiohttp.ClientSession() as session,
        session.get(f"{hub.url}/{REPO}/resolve/main/{SHARD}") as get,
    ):
        body = await get.read()
    assert hashlib.sha256(body).hexdigest() == blob.sha256
    assert hub.body_bytes(REPO, SHARD) == blob.size


async def test_small_files_use_git_blob_sha1(hub: FakeHub) -> None:
    blob = hub.repos[REPO].main.files["config.json"]
    async with (
        aiohttp.ClientSession() as session,
        session.head(f"{hub.url}/{REPO}/resolve/main/config.json") as head,
    ):
        assert head.status == 200
        assert head.headers["ETag"] == f'"{blob.git_sha1}"'
        assert head.headers["Content-Length"] == str(blob.size)


async def test_range_requests(hub: FakeHub) -> None:
    blob = hub.repos[REPO].main.files[SHARD]
    url = f"{hub.url}/{REPO}/resolve/main/{SHARD}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"Range": "bytes=100-"}) as r:
            assert r.status == 206
            assert await r.read() == blob.content[100:]
        async with session.get(url, headers={"Range": f"bytes={blob.size}-"}) as r:
            assert r.status == 416
        hub.inject(Fault(IgnoreRange()))
        async with session.get(url, headers={"Range": "bytes=100-"}) as r:
            assert r.status == 200
            assert await r.read() == blob.content
        hub.inject(Fault(RangeNotSatisfiable()))
        async with session.get(url, headers={"Range": "bytes=100-"}) as r:
            assert r.status == 416


async def test_body_faults(hub: FakeHub) -> None:
    url = f"{hub.url}/{REPO}/resolve/main/{SHARD}"
    blob = hub.repos[REPO].main.files[SHARD]
    async with aiohttp.ClientSession() as session:
        hub.inject(Fault(Reset(after_bytes=1000)))
        with pytest.raises(aiohttp.ClientError):
            async with session.get(url) as r:
                await r.read()
        hub.inject(Fault(ShortBody(after_bytes=1000)))
        with pytest.raises(aiohttp.ClientPayloadError):
            async with session.get(url) as r:
                await r.read()
        hub.inject(Fault(Stall(after_bytes=10)))
        with pytest.raises(asyncio.TimeoutError):
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(sock_read=0.2)
            ) as r:
                await r.read()
        hub.inject(Fault(Corrupt(offset=5)))
        async with session.get(url) as r:
            body = await r.read()
            assert len(body) == blob.size and body != blob.content


async def test_errors(hub: FakeHub) -> None:
    hub.add_repo(FakeRepo("fake-org/gated", {"config.json": b"{}"}, gated=True))
    async with aiohttp.ClientSession() as session:
        hub.inject(Fault(rate_limited(7), where="head"))
        async with session.head(f"{hub.url}/{REPO}/resolve/main/config.json") as r:
            assert r.status == 429
            assert r.headers["RateLimit"] == '"api";r=0;t=7'
        async with session.head(f"{hub.url}/{REPO}/resolve/main/missing.json") as r:
            assert r.status == 404
            assert r.headers["X-Error-Code"] == "EntryNotFound"
        async with session.head(
            f"{hub.url}/fake-org/gated/resolve/main/config.json"
        ) as r:
            assert r.status == 401
        async with session.head(
            f"{hub.url}/fake-org/gated/resolve/main/config.json",
            headers={"Authorization": "Bearer x"},
        ) as r:
            assert r.status == 403


async def test_down_and_up(hub: FakeHub) -> None:
    url = f"{hub.url}/{REPO}/resolve/main/config.json"
    await hub.go_down()
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientConnectionError):
            async with session.head(url) as r:
                pass
        await hub.come_up()
        async with session.head(url) as r:
            assert r.status == 200


async def test_upstream_change_moves_main(hub: FakeHub) -> None:
    repo = hub.repos[REPO]
    first = repo.main.sha
    repo.commit({"config.json": b'{"changed": true}'})
    assert repo.main.sha != first
    assert repo.at(first) is not None
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{hub.url}/{REPO}/resolve/{first}/config.json") as r:
            assert await r.read() == model_repo_files(REPO)["config.json"]
        async with session.get(f"{hub.url}/{REPO}/resolve/main/config.json") as r:
            assert await r.read() == b'{"changed": true}'
