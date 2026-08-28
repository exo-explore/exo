"""Unit tests for the speculative-decode snapshot/rollback helpers.

These exercise :mod:`exo.worker.engines.mlx.spec_cache` directly with fake
cache entries that mimic the mlx-lm cache API (``is_trimmable`` / ``trim`` /
``state`` / ``meta_state``), so no real model or GatedDeltaNet is needed.
"""

# The snapshot stores cloned cache state as ``tuple[Any, ...]`` by design,
# so reads off it are Any -- mirror the pragma used by the mlx modules.
# pyright: reportAny=false

from __future__ import annotations

from typing import Any

import mlx.core as mx

from exo.worker.engines.mlx.spec_cache import (
    CacheSnapshot,
    rollback_after_verify,
    snapshot_untrimmable_cache_lazy,
)


class _TrimmableEntry:
    """Stands in for an attention KV cache: trimmable, no recurrent state."""

    def __init__(self) -> None:
        self.trim_calls: list[int] = []

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        self.trim_calls.append(n)


class _NonTrimmableEntry:
    """Stands in for a GatedDeltaNet ``ArraysCache``: not trimmable, carries
    recurrent state in a list that the cache reads from in place."""

    def __init__(self, state: list[Any], meta_state: Any) -> None:
        self.state = state
        self.meta_state = meta_state

    def is_trimmable(self) -> bool:
        return False


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a, b))


def test_snapshot_skips_trimmable_and_clones_non_trimmable() -> None:
    trimmable = _TrimmableEntry()
    non_trimmable = _NonTrimmableEntry(
        state=[mx.array([1, 2, 3])], meta_state=mx.array([9])
    )
    snap = snapshot_untrimmable_cache_lazy([trimmable, non_trimmable])

    assert isinstance(snap, CacheSnapshot)
    # Trimmable slot carries None (offset trim is sufficient rollback).
    assert snap.states[0] is None
    assert snap.meta_states[0] is None
    # Non-trimmable slot carries a clone of the recurrent state.
    assert snap.states[1] is not None
    assert _arrays_equal(snap.states[1][0], mx.array([1, 2, 3]))


def test_snapshot_clone_is_isolated_from_later_mutation() -> None:
    """A snapshot must not alias the live cache: replacing the cache's state
    leaves (as the verify forward does) must not change the snapshot."""
    entry = _NonTrimmableEntry(state=[mx.array([1, 2, 3])], meta_state=None)
    snap = snapshot_untrimmable_cache_lazy([entry])

    # Simulate the verify forward replacing the recurrent-state leaf.
    entry.state = [mx.array([7, 8, 9])]

    assert _arrays_equal(snap.states[0][0], mx.array([1, 2, 3]))


def test_rollback_trims_trimmable_and_restores_non_trimmable() -> None:
    trimmable = _TrimmableEntry()
    original_state = [mx.array([1, 2, 3])]
    entry = _NonTrimmableEntry(state=original_state, meta_state=mx.array([5]))
    cache: list[Any] = [trimmable, entry]

    snap = snapshot_untrimmable_cache_lazy(cache)

    # Verify forward advances state past the speculative trajectory.
    entry.state[:] = [mx.array([7, 8, 9])]
    entry.meta_state = mx.array([6])

    rollback_after_verify(cache, snap, verified_tokens=2)

    # Trimmable entry trimmed by the requested count.
    assert trimmable.trim_calls == [2]
    # Non-trimmable recurrent state + meta restored from the snapshot.
    assert _arrays_equal(entry.state[0], mx.array([1, 2, 3]))
    assert _arrays_equal(entry.meta_state, mx.array([5]))
    # Container identity preserved (cache reads from the same list object).
    assert entry.state is original_state


def test_rollback_with_zero_verified_tokens_skips_trim_but_restores() -> None:
    trimmable = _TrimmableEntry()
    entry = _NonTrimmableEntry(state=[mx.array([1, 2, 3])], meta_state=None)
    cache: list[Any] = [trimmable, entry]

    snap = snapshot_untrimmable_cache_lazy(cache)
    entry.state[:] = [mx.array([7, 8, 9])]

    rollback_after_verify(cache, snap, verified_tokens=0)

    assert trimmable.trim_calls == []
    assert _arrays_equal(entry.state[0], mx.array([1, 2, 3]))
