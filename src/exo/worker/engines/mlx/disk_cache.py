"""Prefix-cache entries kept on disk instead of in GPU memory (EXO_PREFIX_CACHE_DISK=1).

The prefix cache keeps a full copy of every cached conversation in (GPU) memory: a copy of the
active conversation next to the live cache serving it, plus every inactive conversation. At long
context two conversations no longer fit next to the model, so memory pressure evicts them and
returning to one reads its whole prompt again. With EXO_PREFIX_CACHE_DISK=1, an entry of at
least EXO_PREFIX_CACHE_DISK_MIN_TOKENS tokens is written to disk instead and read back when a
request matches it, so between requests memory holds no cached conversation. Entries belong to
the process that wrote them (a directory per process, removed once it has exited); they are not
a cache across restarts.

The write waits until the reply is sent: while a request is served, its entry is held in memory
as without this option, and KVPrefixCache.write_pending_to_disk writes it once the runner has no
request in flight. Writes are incremental: each attention layer's keys and values are stored in
blocks of _BLOCK_TOKENS tokens, and an update rewrites only the blocks from the reused prefix
on. Snapshots of the recurrent state are written once each. A KVCache subclass that keeps more per-token
arrays (B, L, ...) names them in ``disk_cache_arrays``. Entries made of other cache types
(rotating, quantized, CacheList, ...) are not supported here and stay in memory.
"""

import importlib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.shared.constants import EXO_CACHE_HOME
from exo.worker.runner.bootstrap import logger

ENABLED = os.environ.get("EXO_PREFIX_CACHE_DISK", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
MIN_TOKENS = int(os.environ.get("EXO_PREFIX_CACHE_DISK_MIN_TOKENS", "16384"))
MAX_BYTES = int(float(os.environ.get("EXO_PREFIX_CACHE_DISK_GB", "40")) * 2**30)
ROOT = Path(
    os.environ.get("EXO_PREFIX_CACHE_DIR", str(EXO_CACHE_HOME / "prefix-cache"))
)
_BLOCK_TOKENS = 2048

DiskCache = KVCache | ArraysCache


class Snapshot(Protocol):
    """cache.CacheSnapshot, as this module reads it."""

    @property
    def states(self) -> Sequence[object]: ...
    @property
    def token_count(self) -> int: ...


@dataclass
class DiskEntry:
    path: Path
    kinds: list[str]  # per layer: "kv" or "arrays"
    classes: list[str]  # per layer: "module:qualname" of its cache class
    arrays_sizes: list[int]  # per layer: number of arrays slots (0 for kv layers)
    extra_arrays: list[tuple[str, ...]]  # per layer: its class's disk_cache_arrays
    length: int = 0  # tokens in the cache (as cache_length)
    kv_lengths: list[int] = field(default_factory=list[int])  # per layer offset
    blocks: list[int] = field(
        default_factory=list[int]
    )  # per layer: block files on disk
    snapshot_counts: list[int] = field(default_factory=list[int])
    nbytes: int = 0

    @property
    def has_ssm(self) -> bool:
        return "arrays" in self.kinds


def _class_path(obj: object) -> str:
    return f"{type(obj).__module__}:{type(obj).__qualname__}"


def _load_class(path: str) -> type:
    module, qualname = path.split(":")
    target: object = importlib.import_module(module)
    for part in qualname.split("."):
        target = cast(object, getattr(target, part))
    assert isinstance(target, type)
    return target


def _extra_arrays(c: object) -> tuple[str, ...]:
    return cast(tuple[str, ...], getattr(type(c), "disk_cache_arrays", ()))


def supports(cache: Sequence[object]) -> bool:
    """Whether every entry is a cache type this module can write and read back: ArraysCache,
    KVCache, or a KVCache subclass whose extra state is the arrays it names in
    ``disk_cache_arrays``."""
    for c in cache:
        if type(c) is ArraysCache:
            if c.left_padding is not None or c.lengths is not None:
                return False
        elif type(c) is KVCache or (
            isinstance(c, KVCache) and hasattr(type(c), "disk_cache_arrays")
        ):
            continue
        else:
            return False
    return bool(cache)


def new_root() -> Path:
    """A directory for this cache, after removing those of processes that have exited (their
    entries went with the process that indexed them)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    for old in ROOT.iterdir():
        pid = old.name.split("-", 1)[0]
        if pid.isdigit() and not _alive(int(pid)):
            shutil.rmtree(old, ignore_errors=True)
    path = ROOT / f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _save(path: Path, arrays: dict[str, mx.array]) -> None:
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), arrays)  # pyright: ignore[reportUnknownMemberType]
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, mx.array]:
    return cast(dict[str, mx.array], mx.load(str(path)))


def _block_file(entry: DiskEntry, layer: int, block: int) -> Path:
    return entry.path / f"L{layer}_B{block}.safetensors"


def _snapshot_file(entry: DiskEntry, count: int) -> Path:
    return entry.path / f"S{count}.safetensors"


def save(
    root: Path,
    previous: DiskEntry | None,
    cache: Sequence[object],
    keep_tokens: int,
    new_snapshots: Sequence[Snapshot],
    keep_snapshot_counts: Sequence[int],
    length: int,
) -> DiskEntry:
    """Write `cache` to disk. With `previous` (the entry this cache was restored from), its
    blocks before `keep_tokens` (the reused prefix) are kept, as are its snapshots listed in
    `keep_snapshot_counts`; `new_snapshots` are written. Returns the entry.

    On an OSError a new entry's directory is removed; the files of `previous` may be partly
    rewritten, so the caller deletes it."""
    if previous is None:
        path = root / uuid.uuid4().hex[:12]
        path.mkdir(parents=True)
        try:
            return _write(
                DiskEntry(
                    path=path,
                    kinds=["arrays" if type(c) is ArraysCache else "kv" for c in cache],
                    classes=[_class_path(c) for c in cache],
                    arrays_sizes=[
                        len(c.cache) if isinstance(c, ArraysCache) else 0 for c in cache
                    ],
                    extra_arrays=[_extra_arrays(c) for c in cache],
                    kv_lengths=[0] * len(cache),
                    blocks=[0] * len(cache),
                ),
                cache,
                0,
                new_snapshots,
                [],
                length,
            )
        except OSError:
            shutil.rmtree(path, ignore_errors=True)
            raise
    return _write(
        previous, cache, keep_tokens, new_snapshots, keep_snapshot_counts, length
    )


def _write(
    entry: DiskEntry,
    cache: Sequence[object],
    keep_tokens: int,
    new_snapshots: Sequence[Snapshot],
    keep_snapshot_counts: Sequence[int],
    length: int,
) -> DiskEntry:
    for i, c in enumerate(cache):
        if entry.kinds[i] == "arrays":
            assert isinstance(c, ArraysCache)
            _save(
                entry.path / f"L{i}_arrays.safetensors",
                {f"a{k}": a for k, a in enumerate(c.cache) if a is not None},
            )
            continue
        assert isinstance(c, KVCache)
        n = int(c.offset)
        first = min(keep_tokens, entry.kv_lengths[i], n) // _BLOCK_TOKENS
        last = (n + _BLOCK_TOKENS - 1) // _BLOCK_TOKENS
        for b in range(first, last):
            start, stop = b * _BLOCK_TOKENS, min((b + 1) * _BLOCK_TOKENS, n)
            assert c.keys is not None and c.values is not None
            arrays = {
                "k": c.keys[..., start:stop, :],
                "v": c.values[..., start:stop, :],
            }
            for name in entry.extra_arrays[i]:
                extra = cast(mx.array | None, getattr(c, name))
                if extra is not None:
                    arrays[f"x_{name}"] = extra[:, start:stop]
            _save(_block_file(entry, i, b), arrays)
        for b in range(last, entry.blocks[i]):  # what a longer earlier entry left
            _block_file(entry, i, b).unlink(missing_ok=True)
        entry.kv_lengths[i] = n
        entry.blocks[i] = last

    for count in set(entry.snapshot_counts) - set(keep_snapshot_counts):
        _snapshot_file(entry, count).unlink(missing_ok=True)
    counts = [c for c in entry.snapshot_counts if c in keep_snapshot_counts]
    for snap in new_snapshots:
        arrays = {
            f"L{i}_a{k}": a
            for i, state in enumerate(snap.states)
            if isinstance(state, ArraysCache)
            for k, a in enumerate(state.cache)
            if a is not None
        }
        _save(_snapshot_file(entry, snap.token_count), arrays)
        if snap.token_count not in counts:
            counts.append(snap.token_count)
    entry.snapshot_counts = sorted(counts)
    entry.length = length
    entry.nbytes = sum(f.stat().st_size for f in entry.path.iterdir())
    (entry.path / "entry.json").write_text(json.dumps({"length": entry.length}))
    return entry


def load(entry: DiskEntry) -> list[DiskCache]:
    """The cache as written, read back into memory. Each layer is evaluated as it is read, so
    the block files and the joined arrays coexist for one layer at a time, not the whole cache
    (at 210K tokens that would be several GB more than a long conversation leaves free)."""
    cache: list[DiskCache] = []
    for i, kind in enumerate(entry.kinds):
        if kind == "arrays":
            arrays_cls = cast(type[ArraysCache], _load_class(entry.classes[i]))
            arrays_cache = arrays_cls(entry.arrays_sizes[i])
            saved = _read(entry.path / f"L{i}_arrays.safetensors")
            arrays_cache.cache = [
                saved.get(f"a{k}") for k in range(entry.arrays_sizes[i])
            ]
            mx.eval(*_arrays_of(arrays_cache))
            cache.append(arrays_cache)
            continue
        kv_cls = cast(type[KVCache], _load_class(entry.classes[i]))
        kv = kv_cls()
        if entry.blocks[i]:
            parts = [_read(_block_file(entry, i, b)) for b in range(entry.blocks[i])]
            kv.keys = mx.concatenate([p["k"] for p in parts], axis=2)
            kv.values = mx.concatenate([p["v"] for p in parts], axis=2)
            for name in entry.extra_arrays[i]:
                if f"x_{name}" in parts[0]:
                    setattr(
                        kv,
                        name,
                        mx.concatenate([p[f"x_{name}"] for p in parts], axis=1),
                    )
            mx.eval(*_arrays_of(kv))
            del parts
        kv.offset = entry.kv_lengths[i]
        cache.append(kv)
    return cache


def load_snapshot_states(entry: DiskEntry, count: int) -> list[ArraysCache | None]:
    """The recurrent-state snapshot at `count` tokens: one ArraysCache (or None) per layer."""
    saved = _read(_snapshot_file(entry, count))
    states: list[ArraysCache | None] = []
    for i, kind in enumerate(entry.kinds):
        if kind != "arrays":
            states.append(None)
            continue
        state = ArraysCache(entry.arrays_sizes[i])
        state.cache = [saved.get(f"L{i}_a{k}") for k in range(entry.arrays_sizes[i])]
        states.append(state)
    return states


def delete(entry: DiskEntry) -> None:
    shutil.rmtree(entry.path, ignore_errors=True)


def _arrays_of(c: DiskCache) -> list[mx.array]:
    if isinstance(c, ArraysCache):
        return [a for a in c.cache if a is not None]
    out = [a for a in (c.keys, c.values) if a is not None]
    for name in _extra_arrays(c):
        extra = cast(mx.array | None, getattr(c, name, None))
        if extra is not None:
            out.append(extra)
    return out


def log_saved(entry: DiskEntry, written_from: int, seconds: float) -> None:
    logger.info(
        f"KV cache entry on disk: {entry.length} tokens, {entry.nbytes / 2**20:.0f} MB "
        f"(rewrote from token {written_from}, in {seconds:.1f} s)"
    )
