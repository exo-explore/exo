# pyright: reportPrivateUsage=false
"""Snapshots of DeepseekV4Cache entries are exact, detached, and batched."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.engines.mlx.cache import snapshot_ssm_states

SLIDING_WINDOW = 128
HEAD_DIM = 8
NUM_LAYERS = 6


def _make_cache(seed: int) -> DeepseekV4Cache:
    mx.random.seed(seed)
    cache = DeepseekV4Cache(SLIDING_WINDOW)
    keys = mx.random.normal((1, 1, 20, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, 1, 20, HEAD_DIM)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    for branch in cache._branches.values():
        branch.buffer_kv = mx.random.normal((1, 3, HEAD_DIM)).astype(mx.bfloat16)
        branch.buffer_gate = mx.random.normal((1, 3, HEAD_DIM))
        branch.pool = mx.random.normal((1, 5, HEAD_DIM)).astype(mx.bfloat16)
        branch.buffer_lengths = [3]
        branch.pool_lengths = [5]
    arrays = [cache.local.keys, cache.local.values, *_branch_arrays(cache)]
    mx.eval([a for a in arrays if a is not None])
    return cache


def _independent(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    copy = a * 1
    mx.eval(copy)
    return copy


def _branch_arrays(cache: DeepseekV4Cache) -> list[mx.array | None]:
    arrays: list[mx.array | None] = []
    for branch in cache._branches.values():
        arrays.extend(
            [
                branch.buffer_kv,
                branch.buffer_gate,
                branch.prev_kv,
                branch.prev_gate,
                branch.pool,
            ]
        )
    return arrays


def test_snapshot_matches_and_is_detached() -> None:
    caches = [_make_cache(seed) for seed in range(NUM_LAYERS)]
    expected_keys = [_independent(c.local.keys) for c in caches]
    expected_branches = [[_independent(a) for a in _branch_arrays(c)] for c in caches]

    snapshot = snapshot_ssm_states(caches)
    assert snapshot.token_count == 20

    for c in caches:
        new_keys = mx.ones((1, 1, 1, HEAD_DIM), dtype=mx.bfloat16)
        c.update_and_fetch(new_keys, new_keys)
        for branch in c._branches.values():
            assert branch.pool is not None
            branch.pool[:] = 0

    for state, keys, branches in zip(
        snapshot.states, expected_keys, expected_branches, strict=True
    ):
        assert isinstance(state, DeepseekV4Cache)
        assert isinstance(state.local, RotatingKVCache)
        assert state.local.keys is not None
        assert keys is not None
        assert state.local.keys.dtype == mx.bfloat16
        assert state.local._idx == 20
        assert state.local.offset == 20
        assert mx.array_equal(state.local.keys, keys)
        for copied, original in zip(_branch_arrays(state), branches, strict=True):
            if original is None:
                assert copied is None
            else:
                assert copied is not None
                assert copied.dtype == original.dtype
                assert mx.array_equal(copied, original)
        for branch in state._branches.values():
            assert branch.buffer_lengths == [3]
            assert branch.pool_lengths == [5]


def test_snapshot_syncs_twice_regardless_of_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caches = [_make_cache(seed) for seed in range(NUM_LAYERS)]
    original_eval = mx.eval
    calls: list[int] = []

    def counting_eval(*args: "mx.MX_ARRAY_TREE | None") -> None:
        calls.append(1)
        original_eval(*args)

    monkeypatch.setattr(mx, "eval", counting_eval)
    snapshot_ssm_states(caches)

    assert len(calls) == 2
