"""A model's own cache class that can't be trimmed (SnapshotCache) is snapshotted and
restored by the prefix cache, never trimmed."""

from typing import Self

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.cache import (
    cache_length,
    copy_snapshot_entry,
    has_non_kv_caches,
    is_non_trimmable_cache_entry,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.types import KVCacheType, SnapshotCache


class _WindowCache:
    """Keeps the last rows it was given, like a sliding window."""

    def __init__(self) -> None:
        self.offset = 0
        self.rows: mx.array | None = None

    def update(self, new: mx.array) -> None:
        rows = new if self.rows is None else mx.concatenate([self.rows, new])
        self.rows = rows[-4:]
        self.offset += new.shape[0]

    @property
    def nbytes(self) -> int:
        return 0 if self.rows is None else self.rows.nbytes

    def prefix_cache_snapshot(self) -> Self:
        copy = type(self)()
        copy.offset = self.offset
        if self.rows is not None:
            copy.rows = mx.array(self.rows)
            mx.eval(copy.rows)
        return copy


def test_window_cache_is_a_snapshot_cache_and_kv_cache_is_not() -> None:
    assert isinstance(_WindowCache(), SnapshotCache)
    assert not isinstance(KVCache(), SnapshotCache)
    assert is_non_trimmable_cache_entry(_WindowCache())
    assert not is_non_trimmable_cache_entry(KVCache())


def test_snapshot_and_restore() -> None:
    cache = _WindowCache()
    cache.update(mx.arange(6))
    caches: KVCacheType = [cache]
    assert has_non_kv_caches(caches)
    snapshot = snapshot_ssm_states(caches)
    assert snapshot.token_count == 6

    cache.update(mx.arange(10, 13))
    assert cache_length(caches) == 9

    restored = copy_snapshot_entry(snapshot.states[0])
    assert isinstance(restored, _WindowCache)
    assert restored.offset == 6
    assert restored.rows is not None and restored.rows.tolist() == [2, 3, 4, 5]
    # the snapshot is not shared with what was restored from it
    assert restored is not snapshot.states[0]

    live: list[_WindowCache] = [cache]
    trim_cache(live, 3, snapshot)
    assert live[0].offset == 6 and live[0] is not cache


def test_trimming_without_a_snapshot_is_refused() -> None:
    cache = _WindowCache()
    cache.update(mx.arange(3))
    with pytest.raises(ValueError):
        trim_cache([cache], 1, None)
