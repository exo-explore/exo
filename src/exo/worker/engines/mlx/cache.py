import contextlib
import gc

# KV disk persistence (offload) helpers
import hashlib
import json
import os
import time as _time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
import numpy as np
import psutil
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    ConcatenateKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_CACHE_HOME
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


# ── KV disk persistence: state (de)serialization helpers ──
#
# Cache states are not pure array trees: GLM/DeepSeek-V3.2 DSA indexer caches
# hold zero-width value arrays (safetensors rejects size-0 tensors) and
# DeepseekV4Cache branch state holds Nones, ints and int-lists. Such leaves are
# recorded as JSON "placeholders" in the slot's _meta.json and re-inserted on
# load, so the .safetensors file itself stays byte-compatible with stock
# mlx-lm save_prompt_cache for standard architectures.

_CACHE_CLASS_REGISTRY: dict[str, type] = {
    "KVCache": KVCache,
    "RotatingKVCache": RotatingKVCache,
    "QuantizedKVCache": QuantizedKVCache,
    "ArraysCache": ArraysCache,
    "CacheList": CacheList,
    "ConcatenateKVCache": ConcatenateKVCache,
    "ChunkedKVCache": ChunkedKVCache,
    "DeepseekV4Cache": DeepseekV4Cache,
}


def _is_int_list_leaf(x: object) -> bool:
    # mlx tree_flatten silently drops empty lists, so int-lists (DeepseekV4Cache
    # buffer/pool lengths, possibly []) must be kept whole as single leaves.
    return isinstance(x, list) and all(
        isinstance(i, int) for i in cast("list[object]", x)
    )


def _partition_cache_state(
    cache: KVCacheType,
) -> tuple[dict[str, mx.array], dict[str, dict[str, object]]]:
    """Split flattened per-layer cache state into safetensors-storable arrays
    plus JSON placeholder specs for the leaves safetensors cannot hold."""
    states: list[object] = [cast(object, c.state) for c in cache]
    flat = cast(
        "list[tuple[str, object]]",
        tree_flatten(states, is_leaf=_is_int_list_leaf),
    )
    arrays: dict[str, mx.array] = {}
    placeholders: dict[str, dict[str, object]] = {}
    for path, leaf in flat:
        if isinstance(leaf, mx.array):
            if leaf.size > 0:
                arrays[path] = leaf
            else:
                placeholders[path] = {
                    "kind": "empty",
                    "shape": list(leaf.shape),
                    "dtype": str(leaf.dtype).split(".")[-1],
                }
        elif leaf is None:
            placeholders[path] = {"kind": "none"}
        elif isinstance(leaf, (bool, int, float, str, list)):
            placeholders[path] = {"kind": "json", "value": leaf}
        else:
            raise TypeError(f"unserializable cache state leaf at {path}: {type(leaf)}")
    return arrays, placeholders


def _materialize_placeholder(spec: dict[str, object]) -> object:
    kind = spec.get("kind")
    if kind == "none":
        return None
    if kind == "empty":
        shape = cast("list[int]", spec["shape"])
        dtype = cast("mx.Dtype", getattr(mx, cast(str, spec["dtype"])))
        return mx.zeros(shape, dtype)
    if kind == "json":
        return spec["value"]
    raise ValueError(f"unknown cache placeholder kind: {kind}")


def _reconstruct_cache_entry(class_name: str, state: object, meta: object) -> object:
    cls = _CACHE_CLASS_REGISTRY[class_name]  # KeyError -> caller fail-safes
    if cls is DeepseekV4Cache:
        # No usable from_state: __init__ needs the sliding window, which is the
        # wrapped RotatingKVCache's max_size = meta_state[1].
        meta_strs = cast("list[str]", meta)
        v4 = DeepseekV4Cache(int(meta_strs[1]))
        v4.state = state
        v4.meta_state = meta
        return v4
    if cls is CacheList:
        # Resolve inner classes through this registry — mlx-lm's
        # CacheList.from_state only looks up names in its own module globals.
        names, metas = cast("tuple[list[str], list[object]]", meta)
        inner_states = cast("list[object]", state)
        return CacheList(
            *(
                _reconstruct_cache_entry(n, s, m)
                for n, s, m in zip(names, inner_states, metas, strict=True)
            )
        )
    from_state = cast("Callable[[object, object], object]", cls.from_state)
    return from_state(state, meta)


def _load_slot_cache(
    cache_file: str, placeholders: dict[str, dict[str, object]]
) -> list[object]:
    """exo-side load_prompt_cache: re-inserts placeholder leaves and rebuilds
    cache classes, including ones mlx-lm's loader cannot (DeepseekV4Cache)."""
    arrays, metadata = cast(
        "tuple[dict[str, mx.array], dict[str, str]]",
        cast(object, mx.load(cache_file, return_metadata=True)),
    )
    items: list[tuple[str, object]] = list(arrays.items())
    items += [
        (path, _materialize_placeholder(spec)) for path, spec in placeholders.items()
    ]
    state_tree = cast("list[object]", tree_unflatten(items))
    info, _, classes = cast(
        "tuple[list[object], object, list[str]]",
        tree_unflatten(list(metadata.items())),
    )
    return [
        _reconstruct_cache_entry(c, s, m)
        for c, s, m in zip(classes, state_tree, info, strict=True)
    ]


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None, model_id: str = ""):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group
        # KV disk persistence (offload). Opt-in:
        # set EXO_KV_DISK_PERSISTENCE=1 to enable.
        self._model_id = model_id
        self._disk_enabled = os.environ.get("EXO_KV_DISK_PERSISTENCE", "0") == "1"
        self._disk_dir = (
            self._init_disk_dir() if model_id and self._disk_enabled else None
        )
        self._disk_dirty = False
        self._flush_requested_at: float = 0.0
        self._hot_slot_disk_id: int | None = None

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. With disk on: single hot slot (flush current to disk first). Else: LRU evict."""
        if self._disk_dir and len(self.caches) > 0:
            if self._disk_dirty:
                self._flush_hot_slot()
            self.prompts.clear()
            self.caches.clear()
            self._snapshots.clear()
            self._media_regions.clear()
            self._last_used.clear()
            self.prefill_tps.clear()
            self._hot_slot_disk_id = None
        else:
            self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._disk_dirty = True
        if not self._flush_requested_at:
            self._flush_requested_at = _time.time()
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._disk_dirty = True
        if not self._flush_requested_at:
            self._flush_requested_at = _time.time()
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        # Disk fallback: in-memory hit is only a partial prefix of a different slot
        if best_index is not None and self._disk_dir:
            _cached_len = len(self.prompts[best_index])
            if best_length < _cached_len - 1:
                _disk_result = self._try_load_from_disk(
                    model, prompt_tokens, min_prefix=best_length
                )
                if _disk_result is not None:
                    return _disk_result

        if best_index is None:
            if self._disk_dir:
                _disk_result = self._try_load_from_disk(model, prompt_tokens)
                if _disk_result is not None:
                    return _disk_result
            return make_kv_cache(model), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        if has_ssm:
            target = best_length
        else:
            desired = (max_length - 1) if is_exact else best_length
            target = min(cached_length, desired)
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None, False

        prompt_cache = deepcopy(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        evicted_any = False
        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._media_regions.pop(lru_index)
            self._last_used.pop(lru_index)
            self.prefill_tps.pop(lru_index)

            evicted_any = True
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    # ── KV disk persistence (offload) ──

    def _init_disk_dir(self):
        h = hashlib.sha256(self._model_id.encode()).hexdigest()[:16]
        base = _Path(
            os.environ.get("EXO_KV_DISK_PATH", str(EXO_CACHE_HOME / "kv-cache"))
        )
        d = base / h
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"KV cache disk dir: {d}")
        return d

    def _list_disk_slots(self) -> list[int]:
        disk_dir = self._disk_dir
        if disk_dir is None:
            return []
        slots: list[int] = []
        for f in disk_dir.glob("slot_*_tokens.safetensors"):
            try:
                slots.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(slots)

    def _next_disk_slot_id(self) -> int:
        existing = self._list_disk_slots()
        return max(existing) + 1 if existing else 0

    def _flush_hot_slot(self) -> None:
        """Save the current hot slot to disk immediately (atomic write)."""
        disk_dir = self._disk_dir
        if disk_dir is None or len(self.caches) == 0:
            return
        try:
            disk_dir.mkdir(parents=True, exist_ok=True)
            slot_id = (
                self._hot_slot_disk_id
                if self._hot_slot_disk_id is not None
                else self._next_disk_slot_id()
            )
            base = disk_dir / f"slot_{slot_id}"
            # Inline save_prompt_cache, but state leaves safetensors cannot hold
            # (zero-width arrays, Nones, ints — GLM DSA / DeepSeek-V4) are
            # recorded as placeholders in _meta.json instead of being dropped.
            cache = list(self.caches[0])
            cache_data, placeholders = _partition_cache_state(cache)
            cache_info: list[Any] = [c.meta_state for c in cache]  # pyright: ignore[reportUnknownMemberType]
            cache_classes = [type(c).__name__ for c in cache]
            cache_metadata: dict[str, str] = cast(
                "dict[str, str]", dict(tree_flatten([cache_info, {}, cache_classes]))
            )
            tmp_cache = str(base) + "_tmp_cache.safetensors"
            mx.save_safetensors(tmp_cache, cache_data, cache_metadata)  # pyright: ignore[reportUnknownMemberType]
            os.rename(tmp_cache, str(base) + "_cache.safetensors")
            meta = {
                "model_id": self._model_id,
                "token_count": int(len(self.prompts[0])),
                "timestamp": _time.time(),
                "format": 2,
                "placeholders": placeholders,
            }
            # ".json.tmp" suffix, NOT "slot_N_tmp_meta.json": the latter would
            # match the slot_*_meta.json globs in _evict_stale_disk_slots.
            tmp_meta = str(base) + "_meta.json.tmp"
            with open(tmp_meta, "w") as f:
                json.dump(meta, f)
            os.rename(tmp_meta, str(base) + "_meta.json")
            # Tokens written last: slot discovery globs *_tokens.safetensors,
            # so the slot only becomes visible once all three files exist.
            tmp_tokens = str(base) + "_tmp_tokens.safetensors"
            mx.save_safetensors(tmp_tokens, {"tokens": self.prompts[0]})  # pyright: ignore[reportUnknownMemberType]
            os.rename(tmp_tokens, str(base) + "_tokens.safetensors")
            self._hot_slot_disk_id = slot_id
            self._disk_dirty = False
            self._flush_requested_at = 0.0
            logger.info(
                f"KV cache flushed to disk: slot_{slot_id} ({len(self.prompts[0])} tokens)"
            )
        except Exception as e:
            logger.warning(f"KV cache disk flush failed: {e}")

    def _evict_stale_disk_slots(self) -> None:
        """Delete stale disk slots (TTL) then evict oldest while over the size cap."""
        disk_dir = self._disk_dir
        if disk_dir is None:
            return
        max_age_hours = float(os.environ.get("EXO_KV_DISK_TTL_HOURS", "24"))
        max_size_gb = float(os.environ.get("EXO_KV_DISK_MAX_SIZE_GB", "500"))
        cutoff = _time.time() - (max_age_hours * 3600)
        for meta_file in disk_dir.glob("slot_*_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = cast("dict[str, Any]", json.load(f))
                if cast(float, meta.get("timestamp", 0)) < cutoff:
                    slot_id = int(meta_file.stem.split("_")[1])
                    if slot_id == self._hot_slot_disk_id:
                        continue
                    base = disk_dir / f"slot_{slot_id}"
                    for ext in [
                        "_cache.safetensors",
                        "_tokens.safetensors",
                        "_meta.json",
                    ]:
                        with contextlib.suppress(FileNotFoundError):
                            os.remove(str(base) + ext)
                    logger.info(f"KV cache evicted stale disk slot_{slot_id}")
            except Exception:
                continue
        max_bytes = max_size_gb * 1024 * 1024 * 1024
        while True:
            slots: list[tuple[float, int, int]] = []
            total_size = 0
            for meta_file in disk_dir.glob("slot_*_meta.json"):
                try:
                    with open(meta_file) as f:
                        meta = cast("dict[str, Any]", json.load(f))
                    slot_id = int(meta_file.stem.split("_")[1])
                    if slot_id == self._hot_slot_disk_id:
                        continue
                    base = disk_dir / f"slot_{slot_id}"
                    slot_size = sum(
                        f.stat().st_size
                        for ext in [
                            "_cache.safetensors",
                            "_tokens.safetensors",
                            "_meta.json",
                        ]
                        if (f := _Path(str(base) + ext)).exists()
                    )
                    total_size += slot_size
                    slots.append(
                        (cast(float, meta.get("timestamp", 0)), slot_id, slot_size)
                    )
                except Exception:
                    continue
            if total_size <= max_bytes or not slots:
                break
            slots.sort()
            _oldest_ts, oldest_id, oldest_size = slots[0]
            base = disk_dir / f"slot_{oldest_id}"
            for ext in ["_cache.safetensors", "_tokens.safetensors", "_meta.json"]:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(str(base) + ext)
            logger.info(
                f"KV cache evicted disk slot_{oldest_id} "
                f"({oldest_size / 1024 / 1024 / 1024:.1f} GB) — over {max_size_gb} GB limit"
            )

    def flush_to_disk(self, force: bool = False) -> None:
        """Flush the hot slot if dirty and idle for 15s (or force=True)."""
        if not self._disk_dir or not self._disk_dirty:
            return
        if not force and (_time.time() - self._flush_requested_at) < 15:
            return
        self._flush_hot_slot()
        self._evict_stale_disk_slots()

    def _search_disk(self, prompt_tokens: mx.array) -> tuple[int | None, int]:
        """Return (slot_id, prefix_length) of the best on-disk prefix match."""
        disk_dir = self._disk_dir
        if disk_dir is None:
            return None, 0
        best_id: int | None = None
        best_length = 0
        for slot_id in self._list_disk_slots():
            if slot_id == self._hot_slot_disk_id:
                continue
            token_file = disk_dir / f"slot_{slot_id}_tokens.safetensors"
            try:
                cached_tokens = mx.load(str(token_file))["tokens"]
                prefix_len = get_prefix_length(prompt_tokens, cached_tokens)
            except Exception:
                continue
            if prefix_len > best_length:
                best_length = prefix_len
                best_id = slot_id
        return best_id, best_length

    def _try_load_from_disk(
        self, model: Model, prompt_tokens: mx.array, min_prefix: int = 0
    ) -> tuple[KVCacheType, mx.array, int, bool] | None:
        """Swap in a matching disk slot. Returns (cache, remaining, index, is_exact) or None."""
        disk_dir = self._disk_dir
        disk_id, prefix_len = self._search_disk(prompt_tokens)
        if (
            disk_id is None
            or disk_dir is None
            or prefix_len <= min_prefix
            or prefix_len < 1000
        ):
            return None
        try:
            base = disk_dir / f"slot_{disk_id}"
            # Load into locals FIRST. Do NOT mutate in-memory state until the load
            # succeeds — otherwise an incompatible/corrupt slot (e.g. an architecture
            # whose cache doesn't round-trip, like GLM) would leave the cache empty
            # and crash the caller. This way a bad slot fails safe (recompute).
            meta_path = str(base) + "_meta.json"
            placeholders: dict[str, dict[str, Any]] = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    slot_meta = cast("dict[str, Any]", json.load(f))
                # Legacy (format-1) slots have no placeholders — same load path.
                placeholders = cast(
                    "dict[str, dict[str, Any]]", slot_meta.get("placeholders") or {}
                )
            cache: KVCacheType = cast(
                KVCacheType,
                _load_slot_cache(str(base) + "_cache.safetensors", placeholders),
            )
            tokens = mx.load(str(base) + "_tokens.safetensors")["tokens"]
            cached_length = cache_length(cache)
            if cached_length <= 0 or cached_length > int(tokens.shape[0]):
                logger.warning(
                    f"KV cache disk slot_{disk_id} inconsistent "
                    f"(cache {cached_length} vs tokens {int(tokens.shape[0])}) — skipping"
                )
                return None
            # Slots are stored with the cache a couple of tokens SHORTER than
            # the token file (prefill rollback), so cap the reusable prefix at
            # what the cache actually holds — same min() as get_kv_cache.
            prefix_len = min(prefix_len, cached_length)
            if prefix_len <= min_prefix:
                # After capping, no better than the in-memory match that
                # triggered this lookup — keep the in-memory entry.
                return None
            tokens_to_trim = cached_length - prefix_len
            if tokens_to_trim > 0 and has_non_kv_caches(cache):
                # Partial prefix needs trimming, but this architecture's cache
                # (e.g. DeepSeek-V4, SSM) can't be trimmed and disk slots carry
                # no snapshots — recompute instead of corrupting/crashing.
                logger.info(
                    f"KV cache disk slot_{disk_id}: partial prefix on "
                    f"non-trimmable cache — recomputing"
                )
                return None
            prompt_cache = deepcopy(cache)
        except Exception as e:
            logger.warning(f"KV cache disk load failed for slot_{disk_id}: {e}")
            return None

        # Load succeeded — now it is safe to flush the current hot slot and swap in.
        if self._disk_dirty and len(self.caches) > 0:
            self._flush_hot_slot()
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()
        self.prompts.append(tokens)
        self.caches.append(cache)
        self._snapshots.append(None)
        self._media_regions.append([])
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self.prefill_tps.append(0.0)
        self._hot_slot_disk_id = disk_id
        self._disk_dirty = False
        self._flush_requested_at = 0.0
        logger.info(f"KV cache loaded from disk: slot_{disk_id} ({len(tokens)} tokens)")
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim)
            for c in prompt_cache:
                # Mirrors get_kv_cache: these types either can't be trimmed or
                # expose offset as a read-only property (DeepseekV4Cache) —
                # unreachable here thanks to the non-trimmable guard above,
                # kept as defense.
                if isinstance(c, (ArraysCache, RotatingKVCache, DeepseekV4Cache)):
                    continue
                if hasattr(c, "offset"):
                    c.offset = prefix_len
        remaining = prompt_tokens[prefix_len:]
        return prompt_cache, remaining, 0, False

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = isinstance(c, (ArraysCache, RotatingKVCache)) or (
            isinstance(c, CacheList) and not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
        )
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
