# pyright: reportPrivateUsage=false
"""Prefix-cache hits roll DeepseekV4Cache entries back to their snapshot.

`DeepseekV4Cache.trim()` is a no-op, so a partial prefix hit that only called
`trim()` handed the next request a cache still holding the previous request's
tokens.
"""

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.types import KVCacheType

SLIDING_WINDOW = 128
HEAD_DIM = 8
NUM_LAYERS = 2
SNAPSHOT_TOKENS = 12
TOTAL_TOKENS = 20


def _append(cache: KVCacheType, num_tokens: int, fill: float) -> None:
    for c in cache:
        assert isinstance(c, DeepseekV4Cache)
        kv = mx.full((1, 1, num_tokens, HEAD_DIM), fill, dtype=mx.bfloat16)
        c.update_and_fetch(kv, kv)
        mx.eval(c.local.keys, c.local.values)


def _local_keys(c: object) -> mx.array:
    assert isinstance(c, DeepseekV4Cache)
    assert c.local.keys is not None
    return c.local.keys[..., : c.local.offset, :]


def test_trim_cache_restores_deepseek_v4_snapshot() -> None:
    cache: KVCacheType = [DeepseekV4Cache(SLIDING_WINDOW) for _ in range(NUM_LAYERS)]
    _append(cache, SNAPSHOT_TOKENS, 1.0)
    snapshot = snapshot_ssm_states(cache)
    _append(cache, TOTAL_TOKENS - SNAPSHOT_TOKENS, 2.0)

    trim_cache(cache, TOTAL_TOKENS - SNAPSHOT_TOKENS, snapshot)

    for c in cache:
        assert isinstance(c, DeepseekV4Cache)
        assert c.local.offset == SNAPSHOT_TOKENS
        assert mx.array_equal(_local_keys(c), mx.ones_like(_local_keys(c)))


def test_trim_cache_without_snapshot_resets_deepseek_v4_cache() -> None:
    cache: KVCacheType = [DeepseekV4Cache(SLIDING_WINDOW) for _ in range(NUM_LAYERS)]
    _append(cache, TOTAL_TOKENS, 1.0)

    trim_cache(cache, TOTAL_TOKENS - SNAPSHOT_TOKENS)

    for c in cache:
        assert isinstance(c, DeepseekV4Cache)
        assert c.local.offset == 0


def test_partial_prefix_hit_does_not_keep_previous_prompt() -> None:
    cache: KVCacheType = [DeepseekV4Cache(SLIDING_WINDOW) for _ in range(NUM_LAYERS)]
    _append(cache, SNAPSHOT_TOKENS, 1.0)
    snapshot = snapshot_ssm_states(cache)
    _append(cache, TOTAL_TOKENS - SNAPSHOT_TOKENS, 2.0)

    cached_prompt = mx.arange(TOTAL_TOKENS)
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(cached_prompt, cache, [snapshot])

    # Shares the first SNAPSHOT_TOKENS tokens, then diverges.
    new_prompt = mx.concatenate([cached_prompt[:SNAPSHOT_TOKENS], mx.arange(100, 110)])
    restored, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
        None,  # type: ignore[arg-type]  # only used when nothing matches
        new_prompt,
    )

    assert matched_index == 0
    assert not is_exact
    assert remaining.tolist() == new_prompt[SNAPSHOT_TOKENS:].tolist()
    for c in restored:
        assert isinstance(c, DeepseekV4Cache)
        assert c.local.offset == SNAPSHOT_TOKENS
        assert mx.array_equal(_local_keys(c), mx.ones_like(_local_keys(c)))

    # The stored entry is untouched.
    for c in prefix_cache.caches[0]:
        assert isinstance(c, DeepseekV4Cache)
        assert c.local.offset == TOTAL_TOKENS
