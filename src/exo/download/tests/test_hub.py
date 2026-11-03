"""Talking to the Hub: pinning, listing, and what each failure tells the user."""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from exo.download.hub import (
    Hub,
    NotFoundError,
    PermanentError,
    RateLimitedError,
    TransientError,
    parse_retry_after,
)
from exo.download.tests.fake_hub import (
    FakeHub,
    FakeRepo,
    Fault,
    model_repo_files,
    rate_limited,
    server_error,
)
from exo.shared.types.common import ModelId

REPO = ModelId("fake-org/tiny")


class TestParseRetryAfter:
    def test_hf_ratelimit_header(self) -> None:
        assert parse_retry_after({"RateLimit": '"api";r=0;t=243'}) == 243.0

    def test_other_buckets(self) -> None:
        assert parse_retry_after({"RateLimit": '"resolvers";r=0;t=120'}) == 120.0

    def test_extra_whitespace(self) -> None:
        assert parse_retry_after({"RateLimit": '"api"; r=0; t=42'}) == 42.0

    def test_falls_back_to_retry_after(self) -> None:
        assert parse_retry_after({"Retry-After": "60"}) == 60.0

    def test_nothing_usable(self) -> None:
        assert parse_retry_after({}) is None
        assert parse_retry_after({"RateLimit": "garbage"}) is None


@pytest.fixture
def environment(tmp_path: Path) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in ("HF_ENDPOINT", "HF_HOME", "HF_TOKEN")}
    os.environ["HF_HOME"] = str(tmp_path)
    os.environ.pop("HF_TOKEN", None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
async def hub(environment: None) -> AsyncIterator[tuple[FakeHub, Hub]]:
    del environment
    async with FakeHub(page_size=3) as fake:
        os.environ["HF_ENDPOINT"] = fake.url
        fake.add_repo(FakeRepo(str(REPO), model_repo_files(str(REPO))))
        async with Hub.connect(
            connect_timeout=1, idle_timeout=1, metadata_timeout=1
        ) as client:
            yield fake, client


async def test_pins_and_lists_every_page(hub: tuple[FakeHub, Hub]) -> None:
    fake, client = hub
    commit = await client.resolve_commit(REPO)
    assert commit == fake.repos[str(REPO)].main.sha
    files = {f.path: f for f in await client.list_files(REPO, commit)}
    assert set(files) == set(fake.repos[str(REPO)].main.files)
    shard = fake.repos[str(REPO)].main.files["model-00001-of-00002.safetensors"]
    assert files["model-00001-of-00002.safetensors"].sha256 == shard.sha256
    config = fake.repos[str(REPO)].main.files["config.json"]
    assert files["config.json"].sha256 is None
    assert files["config.json"].git_sha1 == config.git_sha1


async def test_gated_model(hub: tuple[FakeHub, Hub]) -> None:
    fake, client = hub
    fake.repos[str(REPO)].gated = True
    with pytest.raises(PermanentError) as raised:
        async with client.open(REPO, "main", "config.json"):
            pass
    assert raised.value.message == (
        "fake-org/tiny is gated. Accept its terms at huggingface.co/fake-org/tiny, "
        "then set HF_TOKEN."
    )


async def test_unknown_repo(hub: tuple[FakeHub, Hub]) -> None:
    _, client = hub
    with pytest.raises(NotFoundError) as raised:
        await client.resolve_commit(ModelId("fake-org/nope"))
    assert raised.value.message == (
        "fake-org/nope was not found on Hugging Face. If it is private, set HF_TOKEN."
    )


async def test_missing_file(hub: tuple[FakeHub, Hub]) -> None:
    _, client = hub
    with pytest.raises(NotFoundError) as raised:
        async with client.open(REPO, "main", "nope.json"):
            pass
    assert (
        raised.value.message
        == "nope.json is listed in fake-org/tiny but was not found (404)."
    )


async def test_rate_limit_and_server_errors_are_transient(
    hub: tuple[FakeHub, Hub],
) -> None:
    fake, client = hub
    fake.inject(Fault(rate_limited(7), where="revision"))
    with pytest.raises(RateLimitedError) as limited:
        await client.resolve_commit(REPO)
    assert limited.value.retry_after == 7
    fake.inject(Fault(server_error(503), where="revision"))
    with pytest.raises(TransientError) as server:
        await client.resolve_commit(REPO)
    assert server.value.message == "Hugging Face server error 503"


async def test_unreachable_hub_is_transient(hub: tuple[FakeHub, Hub]) -> None:
    fake, client = hub
    await fake.go_down()
    with pytest.raises(TransientError) as raised:
        await client.resolve_commit(REPO)
    assert raised.value.message == "can't connect to 127.0.0.1"
