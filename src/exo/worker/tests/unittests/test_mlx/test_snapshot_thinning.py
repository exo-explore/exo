# pyright: reportPrivateUsage=false
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    _thin_snapshots,
)


def _positions(snapshots: list[CacheSnapshot] | None) -> list[int]:
    return [s.token_count for s in snapshots or []]


def _snapshots(*positions: int) -> list[CacheSnapshot]:
    return [CacheSnapshot(states=[], token_count=p) for p in positions]


def test_thin_keeps_two_latest_and_halving_positions() -> None:
    # One 34K-token prefill: a snapshot per 4096-token chunk and one at the end.
    chunks = [*range(4096, 34055, 4096), 34053]
    kept = _thin_snapshots(_snapshots(*chunks))
    assert _positions(kept) == [4096, 8192, 16384, 32768, 34053]


def test_thin_keeps_short_lists() -> None:
    assert _positions(_thin_snapshots(_snapshots(100, 200))) == [100, 200]
    assert _thin_snapshots([]) == []


def test_agent_session_snapshots_stay_bounded() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    cache = [ArraysCache(1)]
    with patch(
        "exo.worker.engines.mlx.cache.get_memory_used_percentage", return_value=0.0
    ):
        kv_prefix_cache.add_kv_cache(
            mx.zeros((34055,), dtype=mx.int32),
            cache,
            _snapshots(*range(4096, 34055, 4096), 34053),
        )
    length = 34055
    for _ in range(200):
        restore_pos = length
        length += 1700
        kv_prefix_cache.update_kv_cache(
            0,
            mx.zeros((length,), dtype=mx.int32),
            cache,
            _snapshots(length - 3, length - 2),
            restore_pos=restore_pos,
        )
    positions = _positions(kv_prefix_cache._snapshots[0])
    assert positions[-2:] == [length - 3, length - 2]
    # Memory grows with position: stays within ~3x the latest snapshot.
    assert sum(positions) <= 3 * positions[-1]
    assert len(positions) <= 12
