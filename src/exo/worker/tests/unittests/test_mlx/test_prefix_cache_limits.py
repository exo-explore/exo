"""The prefix cache stays bounded: a long-running node must not slow down as it fills up."""

from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

import exo.worker.engines.mlx.cache as cache_module
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.types import KVCacheType


class FakeCache:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


def add(prefix_cache: KVPrefixCache, first_token: int, nbytes: int = 10) -> None:
    tokens = mx.array([first_token, first_token + 1, first_token + 2])
    prefix_cache.add_kv_cache(tokens, cast(KVCacheType, [FakeCache(nbytes)]))


def first_tokens(prefix_cache: KVPrefixCache) -> list[int]:
    return [int(prompt[0].item()) for prompt in prefix_cache.prompts]


@pytest.fixture(autouse=True)
def no_memory_pressure() -> Iterator[None]:
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.1):
        yield


def test_the_cache_keeps_at_most_max_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 3)
    prefix_cache = KVPrefixCache(None)

    for first_token in (0, 10, 20, 30, 40):
        add(prefix_cache, first_token)

    assert first_tokens(prefix_cache) == [20, 30, 40]


def test_the_cache_stays_within_its_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_BYTES", 250)
    prefix_cache = KVPrefixCache(None)

    for first_token in (0, 10, 20, 30, 40):
        add(prefix_cache, first_token, nbytes=100)

    assert first_tokens(prefix_cache) == [30, 40]


def test_the_least_recently_used_entry_goes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    prefix_cache = KVPrefixCache(None)
    add(prefix_cache, 0)
    add(prefix_cache, 10)
    prefix_cache._last_used[0] = 100  # pyright: ignore[reportPrivateUsage]

    add(prefix_cache, 20)

    assert first_tokens(prefix_cache) == [0, 20]


def update(prefix_cache: KVPrefixCache, index: int, nbytes: int) -> None:
    tokens = mx.concatenate([prefix_cache.prompts[index], mx.array([-1])])
    prefix_cache.update_kv_cache(
        index,
        tokens,
        cast(KVCacheType, [FakeCache(nbytes)]),
        None,
        restore_pos=len(prefix_cache.prompts[index]),
    )


def test_an_update_that_grows_an_entry_evicts_older_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_BYTES", 250)
    prefix_cache = KVPrefixCache(None)
    add(prefix_cache, 0, nbytes=100)
    add(prefix_cache, 10, nbytes=100)

    update(prefix_cache, 1, nbytes=200)

    assert first_tokens(prefix_cache) == [10]


def test_an_update_keeps_the_updated_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "_MAX_BYTES", 250)
    prefix_cache = KVPrefixCache(None)
    add(prefix_cache, 0, nbytes=100)

    update(prefix_cache, 0, nbytes=300)

    assert first_tokens(prefix_cache) == [0]


def test_an_update_of_a_full_cache_evicts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    prefix_cache = KVPrefixCache(None)
    add(prefix_cache, 0)
    add(prefix_cache, 10)

    update(prefix_cache, 1, nbytes=10)

    assert first_tokens(prefix_cache) == [0, 10]


@pytest.mark.parametrize(
    ("total_gb", "threshold"), [(16, 0.75), (64, 0.9375), (512, 0.94)]
)
def test_eviction_starts_when_less_than_4gb_is_left(
    total_gb: int, threshold: float
) -> None:
    memory = MagicMock(total=total_gb * 2**30)
    with patch.object(cache_module.psutil, "virtual_memory", return_value=memory):
        default = cache_module._default_memory_threshold()  # pyright: ignore[reportPrivateUsage]
    assert abs(default - threshold) < 1e-9
