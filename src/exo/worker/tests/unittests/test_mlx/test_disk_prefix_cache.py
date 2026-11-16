# pyright: reportPrivateUsage=false
"""Prefix-cache entries on disk (disk_cache, EXO_PREFIX_CACHE_DISK=1): an entry waits in memory
until the runner is idle, what is written reads back the same, updates rewrite only what
changed, memory pressure leaves the entries on disk alone, and a failed write keeps the entry in
memory."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

import exo.worker.engines.mlx.cache as cache_module
import exo.worker.engines.mlx.disk_cache as disk_cache
from exo.worker.engines.mlx.cache import KVPrefixCache, snapshot_ssm_states
from exo.worker.engines.mlx.disk_cache import DiskEntry
from exo.worker.engines.mlx.types import KVCacheType, Model


class IndexedKVCache(KVCache):
    """A KVCache with one more per-token array, as qwen4_exp's QSAKVCache"""

    disk_cache_arrays = ("index_keys",)

    def __init__(self) -> None:
        super().__init__()
        self.index_keys: mx.array | None = None


@pytest.fixture(autouse=True)
def on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(disk_cache, "ENABLED", True)
    monkeypatch.setattr(disk_cache, "ROOT", tmp_path / "prefix-cache")
    monkeypatch.setattr(disk_cache, "MIN_TOKENS", 8)
    monkeypatch.setattr(disk_cache, "_BLOCK_TOKENS", 4)
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.1):
        yield


def _kv(cls: type[KVCache], tokens: int, seed: int) -> KVCache:
    mx.random.seed(seed)
    c = cls()
    c.update_and_fetch(
        mx.random.normal((1, 2, tokens, 8)), mx.random.normal((1, 2, tokens, 8))
    )
    if isinstance(c, IndexedKVCache):
        c.index_keys = mx.random.normal((1, tokens, 4))
    return c


def _arrays(seed: int) -> ArraysCache:
    mx.random.seed(seed)
    c = ArraysCache(2)
    c.cache = [mx.random.normal((1, 3, 6)), mx.random.normal((1, 4, 4, 4))]
    return c


def _same_cache(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ArraysCache):
        assert isinstance(b, ArraysCache)
        return all(
            x is not None and y is not None and bool(mx.array_equal(x, y).item())
            for x, y in zip(a.cache, b.cache, strict=True)
        )
    assert isinstance(a, KVCache) and isinstance(b, KVCache)
    assert a.keys is not None and b.keys is not None
    assert a.values is not None and b.values is not None
    n = a.offset
    same = (
        n == b.offset
        and bool(mx.array_equal(a.keys[..., :n, :], b.keys[..., :n, :]).item())
        and bool(mx.array_equal(a.values[..., :n, :], b.values[..., :n, :]).item())
    )
    if isinstance(a, IndexedKVCache):
        assert isinstance(b, IndexedKVCache)
        assert a.index_keys is not None and b.index_keys is not None
        same = same and bool(
            mx.array_equal(a.index_keys[:, :n], b.index_keys[:, :n]).item()
        )
    return same


def test_what_is_written_reads_back_the_same(tmp_path: Path) -> None:
    cache: list[object] = [_kv(KVCache, 10, 1), _arrays(2), _kv(IndexedKVCache, 10, 3)]
    assert disk_cache.supports(cache)
    snapshot = snapshot_ssm_states(cast(KVCacheType, cache))
    entry = disk_cache.save(tmp_path, None, cache, 0, [snapshot], [], 10)
    assert entry.kinds == ["kv", "arrays", "kv"] and entry.has_ssm
    loaded = disk_cache.load(entry)
    assert all(_same_cache(a, b) for a, b in zip(cache, loaded, strict=True))
    states = disk_cache.load_snapshot_states(entry, snapshot.token_count)
    assert states[0] is None and states[2] is None
    assert _same_cache(cache[1], states[1])


def test_other_cache_types_stay_in_memory() -> None:
    class Unknown(KVCache):  # extra state disk_cache doesn't know of
        pass

    assert not disk_cache.supports([Unknown()])
    assert not disk_cache.supports([])


def test_an_update_rewrites_only_the_blocks_after_the_reused_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = _kv(KVCache, 10, 4)
    entry = disk_cache.save(tmp_path, None, [short], 0, [], [], 10)
    assert entry.blocks == [3]  # 4-token blocks: 0-3, 4-7, 8-9
    written: list[str] = []
    real_save = disk_cache._save

    def recording(path: Path, arrays: dict[str, mx.array]) -> None:
        written.append(path.name)
        real_save(path, arrays)

    monkeypatch.setattr(disk_cache, "_save", recording)
    longer = _kv(KVCache, 10, 4)
    longer.update_and_fetch(mx.ones((1, 2, 7, 8)), mx.ones((1, 2, 7, 8)))
    entry = disk_cache.save(tmp_path, entry, [longer], 9, [], [], 17)
    assert written == ["L0_B2.safetensors", "L0_B3.safetensors", "L0_B4.safetensors"]
    assert _same_cache(longer, disk_cache.load(entry)[0])


def _prompt(first: int, length: int) -> mx.array:
    return mx.arange(first, first + length)


def test_a_long_entry_waits_in_memory_then_is_kept_on_disk_and_read_back() -> None:
    prefix_cache = KVPrefixCache(None)
    cache = [_kv(KVCache, 12, 5), _kv(IndexedKVCache, 12, 6)]
    prompt = _prompt(0, 12)
    prefix_cache.add_kv_cache(prompt, cast(KVCacheType, cache))
    # While its request is served, the entry is in memory, as without the disk cache
    assert not isinstance(prefix_cache.caches[0], DiskEntry)
    assert prefix_cache._entry_bytes[0] > 0 and prefix_cache._pending[0] is not None
    assert not any(disk_cache.ROOT.rglob("*.safetensors"))

    prefix_cache.write_pending_to_disk()
    stored = prefix_cache.caches[0]
    assert isinstance(stored, DiskEntry)
    assert prefix_cache._entry_bytes == [0] and prefix_cache._disk_bytes[0] > 0
    assert prefix_cache._pending == [None]

    restored, remaining, index, exact = prefix_cache.get_kv_cache(
        cast(Model, cast(object, None)), prompt
    )
    assert index == 0 and exact and remaining.size == 1  # the last token is read again
    for original, read in zip(cache, restored, strict=True):
        assert isinstance(read, KVCache) and read.offset == 11
        assert original.keys is not None and read.keys is not None
        assert bool(
            mx.array_equal(original.keys[..., :11, :], read.keys[..., :11, :]).item()
        )

    prefix_cache.clear()
    assert not stored.path.exists()


def test_a_short_entry_stays_in_memory() -> None:
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(_prompt(0, 4), cast(KVCacheType, [_kv(KVCache, 4, 7)]))
    prefix_cache.write_pending_to_disk()
    assert not isinstance(prefix_cache.caches[0], DiskEntry)


def test_memory_pressure_evicts_only_the_entries_in_memory() -> None:
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(_prompt(0, 4), cast(KVCacheType, [_kv(KVCache, 4, 8)]))
    prefix_cache.add_kv_cache(
        _prompt(100, 12), cast(KVCacheType, [_kv(KVCache, 12, 9)])
    )
    prefix_cache.write_pending_to_disk()
    on_disk = prefix_cache.caches[1]
    assert isinstance(on_disk, DiskEntry)
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.99):
        prefix_cache._evict_if_needed()
    assert prefix_cache.caches == [on_disk] and on_disk.path.exists()


def test_the_disk_budget_evicts_entries_on_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(_prompt(0, 12), cast(KVCacheType, [_kv(KVCache, 12, 10)]))
    prefix_cache.write_pending_to_disk()
    first = prefix_cache.caches[0]
    assert isinstance(first, DiskEntry)
    monkeypatch.setattr(disk_cache, "MAX_BYTES", first.nbytes + 1)
    prefix_cache.add_kv_cache(
        _prompt(100, 12), cast(KVCacheType, [_kv(KVCache, 12, 11)])
    )
    assert len(prefix_cache.caches) == 1 and not first.path.exists()


def test_a_failed_write_keeps_the_entry_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disk_full(*_: object) -> DiskEntry:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(disk_cache, "save", disk_full)
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(_prompt(0, 12), cast(KVCacheType, [_kv(KVCache, 12, 12)]))
    prefix_cache.write_pending_to_disk()
    assert not isinstance(prefix_cache.caches[0], DiskEntry)
    assert prefix_cache._entry_bytes[0] > 0
    assert prefix_cache._pending == [None] and prefix_cache._disk_bytes == [0]


def test_a_new_cache_removes_the_directories_of_exited_processes() -> None:
    disk_cache.ROOT.mkdir(parents=True)
    exited = disk_cache.ROOT / "999999-deadbeef"  # no such process
    running = disk_cache.ROOT / f"{os.getpid()}-cafe0000"
    exited.mkdir()
    running.mkdir()
    mine = disk_cache.new_root()
    assert not exited.exists() and running.exists() and mine.exists()


def test_an_update_is_written_once_idle_and_only_after_the_reused_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(_prompt(0, 10), cast(KVCacheType, [_kv(KVCache, 10, 13)]))
    prefix_cache.write_pending_to_disk()
    written: list[str] = []
    real_save = disk_cache._save

    def recording(path: Path, arrays: dict[str, mx.array]) -> None:
        written.append(path.name)
        real_save(path, arrays)

    monkeypatch.setattr(disk_cache, "_save", recording)
    longer = _kv(KVCache, 10, 13)
    longer.update_and_fetch(mx.ones((1, 2, 7, 8)), mx.ones((1, 2, 7, 8)))
    prefix_cache.update_kv_cache(
        0, _prompt(0, 17), cast(KVCacheType, [longer]), None, restore_pos=9
    )
    assert written == []  # nothing before the reply

    prefix_cache.write_pending_to_disk()
    # 4-token blocks: those from token 8 on (the reused prefix ends at 9)
    assert written == ["L0_B2.safetensors", "L0_B3.safetensors", "L0_B4.safetensors"]
    stored = prefix_cache.caches[0]
    assert isinstance(stored, DiskEntry)
    assert _same_cache(longer, disk_cache.load(stored)[0])


def _hybrid(tokens: int, seed: int) -> list[object]:
    return [_kv(KVCache, tokens, seed), _arrays(seed + 100)]


def test_a_snapshot_on_disk_is_used_while_its_update_waits() -> None:
    prefix_cache = KVPrefixCache(None)
    at_8 = snapshot_ssm_states(cast(KVCacheType, _hybrid(8, 14)))
    at_12 = snapshot_ssm_states(cast(KVCacheType, _hybrid(12, 15)))
    prefix_cache.add_kv_cache(
        _prompt(0, 12), cast(KVCacheType, _hybrid(12, 15)), [at_8, at_12]
    )
    prefix_cache.write_pending_to_disk()
    first = prefix_cache.caches[0]
    assert isinstance(first, DiskEntry) and first.snapshot_counts == [8, 12]

    # The conversation goes on from token 10: the snapshot at 8 stays valid, the one at 12
    # doesn't, and the update's own snapshot is at 16
    longer = _hybrid(16, 16)
    at_16 = snapshot_ssm_states(cast(KVCacheType, longer))
    prefix_cache.update_kv_cache(
        0, _prompt(0, 16), cast(KVCacheType, longer), [at_16], restore_pos=10
    )
    branch = mx.concatenate([_prompt(0, 10), mx.array([999, 998])])

    def restore() -> tuple[int, object]:
        restored, remaining, index, _ = prefix_cache.get_kv_cache(
            cast(Model, cast(object, None)), branch
        )
        assert index == 0
        return int(branch.size - remaining.size), restored[1]

    # Before the write, the snapshot at 8 is read from the files of the entry being extended
    position, state = restore()
    assert position == 8 and _same_cache(state, at_8.states[1])

    prefix_cache.write_pending_to_disk()
    stored = prefix_cache.caches[0]
    assert isinstance(stored, DiskEntry) and stored.snapshot_counts == [8, 16]
    assert not (stored.path / "S12.safetensors").exists()
    position, state = restore()
    assert position == 8 and _same_cache(state, at_8.states[1])
