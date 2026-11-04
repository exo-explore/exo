"""Concurrent agent sessions that share a system prompt keep their own prefix-cache entries."""

from collections.abc import Iterator
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

import exo.worker.engines.mlx.cache as cache_module
from exo.worker.engines.mlx.cache import CacheSnapshot, KVPrefixCache


def _kv_entry(tokens: int) -> list[KVCache]:
    entry = KVCache()
    keys = mx.zeros((1, 1, tokens, 8), dtype=mx.float16)
    entry.update_and_fetch(keys, keys)  # pyright: ignore[reportUnknownMemberType]
    return [entry]


def _snapshots(*positions: int) -> list[CacheSnapshot]:
    return [CacheSnapshot(states=[], token_count=p) for p in positions]


@pytest.fixture(autouse=True)
def no_memory_pressure() -> Iterator[None]:
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.0):
        yield


def test_only_a_continuation_replaces_an_entry() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    kv_prefix_cache.add_kv_cache(mx.zeros((4437,), dtype=mx.int32), _kv_entry(4435))
    # The same session's next turn restores up to two tokens before the end.
    assert kv_prefix_cache.continues_entry(0, 4435)
    # Another session that only shares the 4096-token system prompt.
    assert not kv_prefix_cache.continues_entry(0, 4096)


def test_restores_from_the_entry_that_restores_most() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    session = mx.arange(6000, dtype=mx.int32)
    # An earlier run shares 5800 tokens but only has a snapshot at 4096
    # before them; this session's own entry ends at 5500.
    stale = mx.concatenate([session[:5800], mx.full((2200,), -1, mx.int32)])
    kv_prefix_cache.add_kv_cache(stale, [ArraysCache(1)], _snapshots(4096, 7999))
    kv_prefix_cache.add_kv_cache(
        session[:5500], [ArraysCache(1)], _snapshots(4096, 5498)
    )
    query = session
    with patch.object(cache_module, "make_kv_cache", return_value=[]):
        _, remaining, index, _ = kv_prefix_cache.get_kv_cache(None, query)  # pyright: ignore[reportArgumentType]
    assert index == 1
    assert len(remaining) == len(query) - 5498
