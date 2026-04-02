import os
from copy import deepcopy

import mlx.core as mx
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.runner.bootstrap import logger

# Checkpoint 2A: disk persistence
import hashlib
import json
import time as _time
from pathlib import Path as _Path


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
        self, states: list[RotatingKVCache | ArraysCache | None], token_count: int
    ):
        self.states = states
        self.token_count = token_count


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[ArraysCache | RotatingKVCache | None] = []
    for c in cache:
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            states.append(deepcopy(c))
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


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(isinstance(c, (ArraysCache, RotatingKVCache)) for c in cache)


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None, model_id: str = ""):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._group = group
        # Disk persistence state
        self._model_id = model_id
        self._disk_dir = self._init_disk_dir() if model_id else None
        self._disk_dirty = False
        self._flush_requested_at: float = 0.0
        self._hot_slot_disk_id: int | None = None

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._last_used.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
    ):
        """Add a new cache entry. Single hot slot: flush current to disk first."""
        if self._disk_dir and len(self.caches) > 0:
            if self._disk_dirty:
                self._flush_hot_slot()
            self.prompts.clear()
            self.caches.clear()
            self._snapshots.clear()
            self._last_used.clear()
            self._hot_slot_disk_id = None
        else:
            self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
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
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._disk_dirty = True
        if not self._flush_requested_at:
            self._flush_requested_at = _time.time()
        logger.info(f"KV cache updated (index {index}, disk slot {self._hot_slot_disk_id}): {len(prompt_tokens)} tokens")

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
    ) -> tuple[KVCacheType, mx.array, int | None]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.
        """
        max_length = len(prompt_tokens)

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        # Disk fallback: cross-conversation match or no in-memory match
        if best_index is not None and self._disk_dir:
            _cached_len = len(self.prompts[best_index])
            _cached_cov = best_length / _cached_len if _cached_len > 0 else 0.0
            if best_length < _cached_len - 1:
                _disk_result = self._try_load_from_disk(model, prompt_tokens, min_prefix=best_length)
                if _disk_result is not None:
                    return _disk_result

        if best_index is None:
            if self._disk_dir:
                _disk_result = self._try_load_from_disk(model, prompt_tokens)
                if _disk_result is not None:
                    return _disk_result
            return make_kv_cache(model), prompt_tokens, None

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        target = (max_length - 1) if is_exact and not has_ssm else best_length
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None

        prompt_cache = deepcopy(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

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
            self._last_used.pop(lru_index)
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

    # ── Disk persistence methods ──

    def _init_disk_dir(self):
        h = hashlib.sha256(self._model_id.encode()).hexdigest()[:16]
        d = _Path.home() / ".exo" / "kv-cache" / h
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"KV cache disk dir: {d}")
        return d

    def _list_disk_slots(self):
        """List all disk slot IDs."""
        if not self._disk_dir:
            return []
        slots = []
        for f in self._disk_dir.glob("slot_*_tokens.safetensors"):
            try:
                slot_id = int(f.stem.split("_")[1])
                slots.append(slot_id)
            except (IndexError, ValueError):
                continue
        return sorted(slots)

    def _next_disk_slot_id(self):
        existing = self._list_disk_slots()
        return max(existing) + 1 if existing else 0

    def _flush_hot_slot(self):
        """Save current hot slot to disk immediately."""
        if len(self.caches) == 0:
            return
        try:
            from mlx_lm.models.cache import save_prompt_cache
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            slot_id = self._hot_slot_disk_id if self._hot_slot_disk_id is not None else self._next_disk_slot_id()
            base = self._disk_dir / f"slot_{slot_id}"
            # Save cache (atomic: write tmp then rename)
            tmp_cache = str(base) + "_tmp_cache.safetensors"
            save_prompt_cache(tmp_cache, list(self.caches[0]))
            os.rename(tmp_cache, str(base) + "_cache.safetensors")
            # Save tokens
            tmp_tokens = str(base) + "_tmp_tokens.safetensors"
            mx.save_safetensors(tmp_tokens, {"tokens": self.prompts[0]})
            os.rename(tmp_tokens, str(base) + "_tokens.safetensors")
            # Save metadata
            meta = {"model_id": self._model_id, "token_count": int(len(self.prompts[0])), "timestamp": _time.time()}
            with open(str(base) + "_meta.json", "w") as f:
                json.dump(meta, f)
            self._hot_slot_disk_id = slot_id
            self._disk_dirty = False
            self._flush_requested_at = 0.0
            logger.info(f"KV cache flushed to disk: slot_{slot_id} ({len(self.prompts[0])} tokens)")
        except Exception as e:
            logger.warning(f"KV cache disk flush failed: {e}")

    def _evict_stale_disk_slots(self, max_age_hours=24):
        """Delete disk slots older than max_age_hours."""
        if not self._disk_dir:
            return
        cutoff = _time.time() - (max_age_hours * 3600)
        for meta_file in self._disk_dir.glob("slot_*_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get("timestamp", 0) < cutoff:
                    slot_id = int(meta_file.stem.split("_")[1])
                    if slot_id == self._hot_slot_disk_id:
                        continue
                    base = self._disk_dir / f"slot_{slot_id}"
                    for ext in ["_cache.safetensors", "_tokens.safetensors", "_meta.json"]:
                        try:
                            os.remove(str(base) + ext)
                        except FileNotFoundError:
                            pass
                    logger.info(f"KV cache evicted stale disk slot_{slot_id}")
            except Exception:
                continue

    def flush_to_disk(self, force=False):
        """Flush hot slot to disk if dirty and idle for 15s (or force)."""
        if not self._disk_dir or not self._disk_dirty:
            return
        if not force and (_time.time() - self._flush_requested_at) < 15:
            return
        self._flush_hot_slot()
        self._evict_stale_disk_slots()

    def _search_disk(self, prompt_tokens):
        """Search disk slots for best prefix match. Returns (slot_id, prefix_length) or (None, 0)."""
        if not self._disk_dir:
            return None, 0
        best_id = None
        best_length = 0
        for slot_id in self._list_disk_slots():
            if slot_id == self._hot_slot_disk_id:
                continue
            token_file = self._disk_dir / f"slot_{slot_id}_tokens.safetensors"
            try:
                cached_tokens = mx.load(str(token_file))["tokens"]
                prefix_len = get_prefix_length(prompt_tokens, cached_tokens)
            except Exception:
                continue
            if prefix_len > best_length:
                best_length = prefix_len
                best_id = slot_id
        return best_id, best_length

    def _try_load_from_disk(self, model, prompt_tokens, min_prefix=0):
        """Search disk for matching slot. If found, swap it in and return (cache, remaining, index)."""
        disk_id, prefix_len = self._search_disk(prompt_tokens)
        if disk_id is None or prefix_len <= min_prefix or prefix_len < 1000:
            return None
        try:
            from mlx_lm.models.cache import load_prompt_cache
            # Flush current hot slot if dirty
            if self._disk_dirty and len(self.caches) > 0:
                self._flush_hot_slot()
            # Clear RAM
            self.prompts.clear()
            self.caches.clear()
            self._snapshots.clear()
            self._last_used.clear()
            # Load from disk
            base = self._disk_dir / f"slot_{disk_id}"
            cache = load_prompt_cache(str(base) + "_cache.safetensors")
            tokens = mx.load(str(base) + "_tokens.safetensors")["tokens"]
            # Install as hot slot
            self.prompts.append(tokens)
            self.caches.append(cache)
            self._snapshots.append(None)
            self._access_counter += 1
            self._last_used.append(self._access_counter)
            self._hot_slot_disk_id = disk_id
            self._disk_dirty = False
            self._flush_requested_at = 0.0
            logger.info(f"KV cache loaded from disk: slot_{disk_id} ({len(tokens)} tokens)")
            # Trim and return
            prompt_cache = deepcopy(cache)
            cached_length = cache_length(cache)
            tokens_to_trim = cached_length - prefix_len
            if tokens_to_trim > 0:
                trim_cache(prompt_cache, tokens_to_trim)
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset = prefix_len
            remaining = prompt_tokens[prefix_len:]
            return prompt_cache, remaining, 0
        except Exception as e:
            logger.warning(f"KV cache disk load failed for slot_{disk_id}: {e}")
            return None

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
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            if snapshot is not None and snapshot.states[i] is not None:
                cache[i] = deepcopy(snapshot.states[i])  # type: ignore
            else:
                c.state = [None] * len(c.state)
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
    c: KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList,
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
    return max(_entry_length(c) for c in cache)


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
