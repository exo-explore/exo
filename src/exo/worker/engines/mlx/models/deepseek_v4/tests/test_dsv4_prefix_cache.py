"""DeepSeek V4 prefix-cache cases beyond test_dsv4_model's partial hit at a chunk boundary:
trim_cache with an empty snapshot and with none, a prompt that shares only a few leading
tokens (restored from the 0-token snapshot), shared prefixes that end inside a chunk
(including one past the first row of the ratio-128 compressor) and an exact hit. Each
restored cache must continue exactly as the same chunks prefilled without the prefix cache.
The model has V4's real sliding window and compression ratios, at small widths."""

from collections.abc import Iterator
from typing import cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    cache_length,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.models.deepseek_v4 import model as dsv4
from exo.worker.engines.mlx.models.deepseek_v4.model import DeepseekV4Cache
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.engines.mlx.types import Model as ExoModel

CHUNK = 64
_BRANCH_KEYS = ("compressor", "indexer")


@pytest.fixture(scope="module")
def model() -> Iterator[dsv4.Model]:
    args = dsv4.ModelArgs(
        vocab_size=1024,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=64,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        sliding_window=128,
        compress_ratios=[0, 0, 4, 128],
    )
    mx.random.seed(0)
    model = dsv4.Model(args)
    mx.eval(model.parameters())
    yield model


def _caches(model: dsv4.Model) -> list[DeepseekV4Cache]:
    return [c for c in model.make_cache() if isinstance(c, DeepseekV4Cache)]


def _filled_cache(window: int, tokens: int) -> DeepseekV4Cache:
    cache = DeepseekV4Cache(window)
    for _ in range(tokens):
        step = mx.random.normal((1, 1, 1, 8))
        cache.update_and_fetch(step, step)
    cache._branches["compressor"].pool = mx.zeros((1, tokens // 4, 8))
    cache._branches["indexer"].pool = mx.zeros((1, tokens // 4, 4))
    return cache


def test_trim_cache_restores_deepseek_v4_layers_from_an_empty_snapshot() -> None:
    empty_snapshot = snapshot_ssm_states(
        cast(KVCacheType, [DeepseekV4Cache(8), DeepseekV4Cache(8)])
    )
    assert empty_snapshot.token_count == 0
    cache = cast(KVCacheType, [_filled_cache(8, 40), _filled_cache(8, 40)])
    assert cache_length(cache) == 40

    trim_cache(cache, 40, empty_snapshot)

    assert cache_length(cache) == 0
    for c in cache:
        assert isinstance(c, DeepseekV4Cache)
        assert c.empty()
        for key in _BRANCH_KEYS:
            assert c._branches[key].pool is None


def test_trim_cache_without_a_snapshot_empties_deepseek_v4_layers() -> None:
    cache = cast(KVCacheType, [_filled_cache(8, 40)])

    trim_cache(cache, 40)

    assert cache_length(cache) == 0
    assert isinstance(cache[0], DeepseekV4Cache)
    assert cache[0]._branches["compressor"].pool is None


def _prefill_with_snapshots(
    model: dsv4.Model, tokens: mx.array, cache: list[DeepseekV4Cache]
) -> list[CacheSnapshot]:
    """As exo's prefill: a snapshot before the first chunk and after every chunk."""
    snapshots = [snapshot_ssm_states(cast(KVCacheType, cache))]
    for start in range(0, tokens.shape[0], CHUNK):
        model(tokens[None, start : start + CHUNK], cache=cache)
        mx.eval([c.local.keys for c in cache if c.local.keys is not None])
        snapshots.append(snapshot_ssm_states(cast(KVCacheType, cache)))
    return snapshots


def _last_logits(
    model: dsv4.Model, tokens: mx.array, cache: list[DeepseekV4Cache]
) -> mx.array:
    if tokens.shape[0] > 1:
        model(tokens[None, :-1], cache=cache)
    return model(tokens[None, -1:], cache=cache)[0, -1].astype(mx.float32)


def _lookup(
    model: dsv4.Model,
    first: mx.array,
    second: mx.array,
    drop_last_snapshot: bool = False,
) -> tuple[list[DeepseekV4Cache], mx.array, bool]:
    """Prefill `first` into a prefix cache and look `second` up in it."""
    first_cache = _caches(model)
    snapshots = _prefill_with_snapshots(model, first, first_cache)
    if drop_last_snapshot:
        # exo's prefill() hands the prefix cache every snapshot but the last
        snapshots = snapshots[:-1]
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(first, cast(KVCacheType, first_cache), snapshots)
    cache, remaining, matched, exact = prefix_cache.get_kv_cache(
        cast(ExoModel, cast(object, model)), second
    )
    assert matched == 0
    assert cache_length(cache) == second.shape[0] - remaining.shape[0]
    return cast(list[DeepseekV4Cache], cache), remaining, exact


def _from_scratch(model: dsv4.Model, tokens: mx.array, restore_pos: int) -> mx.array:
    """The last logits of `tokens` prefilled in the same chunks without the prefix cache."""
    cache = _caches(model)
    if restore_pos:
        _prefill_with_snapshots(model, tokens[:restore_pos], cache)
    return _last_logits(model, tokens[restore_pos:], cache)


def test_prefix_hit_sharing_a_few_leading_tokens_starts_from_an_empty_cache(
    model: dsv4.Model,
) -> None:
    """Two prompts that share only a chat template's first tokens: the second is prefilled
    on an empty cache (restored from the 0-token snapshot), not on top of the first."""
    shared = mx.array([5, 6, 7])
    first = mx.concatenate([shared, mx.random.randint(8, 1024, (300,))])
    second = mx.concatenate([shared, mx.random.randint(8, 1024, (280,))])

    cache, remaining, exact = _lookup(model, first, second)

    assert not exact
    assert remaining.shape[0] == second.shape[0]
    reused = _last_logits(model, remaining, cache)
    assert mx.array_equal(reused, _from_scratch(model, second, 0))


@pytest.mark.parametrize(("shared_tokens", "restore_pos"), [(200, 192), (280, 256)])
def test_prefix_hit_ending_inside_a_chunk_restores_the_snapshot_before_it(
    model: dsv4.Model, shared_tokens: int, restore_pos: int
) -> None:
    """A shared prefix that ends inside a chunk restores the snapshot after the last whole
    chunk in it. At 256 tokens the ratio-128 compressor has emitted rows, so its pool is
    part of what must come back."""
    first = mx.random.randint(8, 1024, (303,))
    second = mx.concatenate([first[:shared_tokens], mx.random.randint(8, 1024, (120,))])

    cache, remaining, exact = _lookup(model, first, second)

    assert not exact
    assert cache_length(cache) == restore_pos
    reused = _last_logits(model, remaining, cache)
    assert mx.array_equal(reused, _from_scratch(model, second, restore_pos))


def test_exact_prefix_hit_restores_the_last_kept_snapshot(model: dsv4.Model) -> None:
    """The same prompt again: with the snapshots prefill() keeps (all but the last), the
    cache is restored at the last whole chunk before the end and the rest is prefilled."""
    prompt = mx.random.randint(8, 1024, (300,))

    cache, remaining, exact = _lookup(model, prompt, prompt, drop_last_snapshot=True)

    assert exact
    assert cache_length(cache) == 256
    assert remaining.shape[0] == 300 - 256
    reused = _last_logits(model, remaining, cache)
    assert mx.array_equal(reused, _from_scratch(model, prompt, 256))
