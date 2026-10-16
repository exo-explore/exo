# pyright: reportPrivateUsage=false
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    thin_snapshots,
)


def _positions(snapshots: list[CacheSnapshot] | None) -> list[int]:
    return [s.token_count for s in snapshots or []]


def _snapshots(*positions: int) -> list[CacheSnapshot]:
    return [CacheSnapshot(states=[], token_count=p) for p in positions]


def test_thin_keeps_two_latest_and_one_per_power_of_two_range() -> None:
    # One 34K-token prefill: a snapshot per 4096-token chunk and one at the end.
    chunks = [*range(4096, 34055, 4096), 34053]
    kept = thin_snapshots(_snapshots(*chunks))
    assert _positions(kept) == [4096, 12288, 28672, 32768, 34053]


def test_thinning_during_prefill_matches_thinning_at_the_end() -> None:
    # 100K-token prefill: thinned after every chunk, as prefill() does.
    chunks = [*range(4096, 105884, 4096), 105882]
    running: list[CacheSnapshot] = []
    for snap in _snapshots(*chunks):
        running = thin_snapshots([*running, snap])
        assert sum(_positions(running)) <= 5 * running[-1].token_count
    assert _positions(running) == _positions(thin_snapshots(_snapshots(*chunks)))


def test_thin_keeps_short_lists() -> None:
    assert _positions(thin_snapshots(_snapshots(100, 200))) == [100, 200]
    assert thin_snapshots([]) == []


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
    # Memory grows with position: stays within 5x the latest snapshot.
    assert sum(positions) <= 5 * positions[-1]
    assert len(positions) <= 12
