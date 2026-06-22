"""Snapshot/rollback helpers for speculative-decode cache state.

MLX's ``trim_prompt_cache`` only rolls back attention KV offsets, but the
Qwen3.5/3.6 GatedDeltaNet ``ArraysCache`` recurrent state cannot be
reconstructed from the offset alone. The native-MTP draft+verify loop
therefore clones that recurrent state before each batched verify forward
and restores it when a draft is rejected. Without this, a batched verify
followed by an offset-only trim produces materially wrong logits.

Adapted from MTPLX's ``snapshot_untrimmable_cache`` /
``rollback_after_verify`` (Apache 2.0).
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportAny=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class CacheSnapshot:
    """Frozen pair of (state, meta_state) tuples — one slot per cache entry.

    Trimmable entries (KV caches) carry `None` in both tuples; non-trimmable
    entries (GatedDeltaNet `ArraysCache`) carry a deep-cloned snapshot.
    """

    states: tuple[Any, ...]
    meta_states: tuple[Any, ...]


def _is_trimmable(entry: Any) -> bool:
    """Mirror MTPLX's check: call `entry.is_trimmable()`, treat raise/missing as
    non-trimmable. mlx-lm 0.31.3's `_BaseCache.is_trimmable` returns False;
    KV-bearing subclasses override to True. `ArraysCache` (GDN recurrent
    state) inherits the base False.
    """
    method = getattr(entry, "is_trimmable", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _clone_tree(value: Any) -> Any:
    """Recursive deep clone of containers + mx.array leaves.

    For mx.array leaves we materialize a fresh contiguous expression and
    force evaluation so subsequent in-place writes by the cache cannot
    mutate the snapshot through aliasing. Cost: one `mx.eval` per leaf.
    """
    if value is None:
        return None
    if isinstance(value, mx.array):
        leaf = mx.contiguous(value)
        mx.eval(leaf)
        return leaf
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    return value


def _clone_tree_lazy(value: Any) -> Any:
    """Recursive clone expression without forcing leaf evaluation.

    MLX arrays are immutable values, and cache updates replace cache leaves
    rather than mutating the old array storage in place. For speculative
    snapshots this lets accept rounds discard the snapshot without paying the
    full synchronization cost.
    """
    if value is None:
        return None
    if isinstance(value, mx.array):
        return mx.contiguous(value)
    if isinstance(value, tuple):
        return tuple(_clone_tree_lazy(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree_lazy(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_tree_lazy(v) for k, v in value.items()}
    return value


def snapshot_untrimmable_cache_lazy(cache: list[Any]) -> CacheSnapshot:
    """Clone state + meta_state of every non-trimmable cache entry.

    Trimmable entries get ``None`` (their offset trim is sufficient
    rollback). Call this BEFORE the batched verify forward each spec-decode
    round. The clone is lazy: MLX arrays are immutable and cache updates
    replace leaves rather than mutating storage in place, so an accept round
    can discard the snapshot without paying the full synchronization cost.
    Measured logit/GDN-exact for K=1 accept and reject transactions.
    """
    states: list[Any] = []
    meta_states: list[Any] = []
    for entry in cache:
        if _is_trimmable(entry):
            states.append(None)
            meta_states.append(None)
        else:
            states.append(_clone_tree_lazy(getattr(entry, "state", None)))
            meta_states.append(_clone_tree_lazy(getattr(entry, "meta_state", None)))
    return CacheSnapshot(states=tuple(states), meta_states=tuple(meta_states))


def _restore_state_preserving_container(entry: Any, state: Any) -> None:
    """Write `state` back into `entry` without breaking container identity.

    `ArraysCache.state` is the same `list` the cache reads from internally.
    Mutating it in place preserves identity; falling back to the property
    setter is fine for entries that own one.
    """
    cloned = _clone_tree(state)
    if hasattr(entry, "replace_state"):
        entry.replace_state(cloned)
        return
    current = getattr(entry, "state", None)
    if (
        isinstance(current, list)
        and isinstance(cloned, list)
        and len(current) == len(cloned)
    ):
        current[:] = cloned
        return
    entry.state = cloned


def restore_cache(cache: list[Any], snapshot: CacheSnapshot) -> None:
    """Restore non-trimmable cache state from a snapshot. Trimmable entries
    untouched. Use directly when you've already done your own trim, or call
    `rollback_after_verify` to combine trim + restore.
    """
    for entry, state, meta_state in zip(
        cache, snapshot.states, snapshot.meta_states, strict=True
    ):
        if state is not None:
            _restore_state_preserving_container(entry, state)
        if meta_state is not None:
            entry.meta_state = _clone_tree(meta_state)


def rollback_after_verify(
    cache: list[Any],
    snapshot: CacheSnapshot,
    verified_tokens: int,
) -> None:
    """Undo a speculative target verify pass.

    `verified_tokens` is the count of tokens to TRIM from trimmable entries
    (matches MTPLX's API and mlx-lm's `trim(n)` "remove n from end"
    convention). For a K=1 verify of `[primary, draft]` advancing the
    cache by 2:
        accept:  rollback_after_verify(cache, snap, verified_tokens=0)
            — keep the full 2-step advance, just rewrite GDN state.
            But the GDN state at pos+2 IS what we want post-accept, so
            callers typically do NOT call rollback on accept; they re-run
            the snapshot at the top of the next round.
        reject:  rollback_after_verify(cache, snap, verified_tokens=2)
            — trim KV by 2, restore GDN state, then re-forward `[primary]`
            alone to advance back to pos+1.

    Mirrors `mtplx/cache_state.py:rollback_after_verify`.
    """
    if verified_tokens > 0:
        for entry in cache:
            if _is_trimmable(entry) and hasattr(entry, "trim"):
                entry.trim(verified_tokens)
    restore_cache(cache, snapshot)
