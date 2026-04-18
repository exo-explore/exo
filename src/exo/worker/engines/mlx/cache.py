import gc
import os
from copy import deepcopy
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
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
from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TURBOQUANT_ENABLED,
    TURBOQUANT_BITS,
    TURBOQUANT_SKETCH_DIM,
    TURBOQUANT_RESIDUAL,
)
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
        self, states: list[RotatingKVCache | ArraysCache | None], token_count: int
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


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[ArraysCache | RotatingKVCache | None] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(deepcopy(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
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
    def __init__(
        self,
        group: mx.distributed.Group | None,
        max_sessions: int | None = None,
        max_bytes: int | None = None,
        max_kv_tokens: int | None = None,
    ):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._pinned: set[int] = set()  # indices protected from eviction
        self._group = group
        # Per-instance caps. None = unbounded (default).
        self._max_sessions = max_sessions
        self._max_bytes = max_bytes
        self._max_kv_tokens = max_kv_tokens

    def pin(self, index: int) -> None:
        """Mark a cache entry as non-evictable."""
        if 0 <= index < len(self.caches):
            self._pinned.add(index)

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self._pinned.clear()
        self.prefill_tps.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
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

        if best_index is None:
            return make_kv_cache(model, max_kv_size=self._max_kv_tokens), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        target = (max_length - 1) if is_exact and not has_ssm else best_length
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model, max_kv_size=self._max_kv_tokens), prompt_tokens, None, False

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

    def _evict_lru_once(self, reason: str) -> bool:
        """Evict a single LRU non-pinned entry. Returns True if evicted."""
        candidates = [
            (i, ts) for i, ts in enumerate(self._last_used) if i not in self._pinned
        ]
        if not candidates:
            return False
        lru_index = min(candidates, key=lambda x: x[1])[0]
        evicted_tokens = len(self.prompts[lru_index])
        self.prompts.pop(lru_index)
        self.caches.pop(lru_index)
        self._snapshots.pop(lru_index)
        self._media_regions.pop(lru_index)
        self._last_used.pop(lru_index)
        self.prefill_tps.pop(lru_index)
        self._pinned = {p - 1 if p > lru_index else p for p in self._pinned if p != lru_index}
        logger.info(f"KV cache evicted LRU entry ({evicted_tokens} tokens) — {reason}")
        return True

    def _entry_bytes(self, cache_list: KVCacheType) -> int:
        """Best-effort byte accounting for a single cache entry's tensors."""
        total = 0
        for c in cache_list:
            for attr in ("keys", "values", "state"):
                t = getattr(c, attr, None)
                if t is None:
                    continue
                if isinstance(t, list):
                    for a in t:
                        nb = getattr(a, "nbytes", None)
                        if nb is not None:
                            total += int(nb)
                else:
                    nb = getattr(t, "nbytes", None)
                    if nb is not None:
                        total += int(nb)
        return total

    def _total_bytes(self) -> int:
        return sum(self._entry_bytes(c) for c in self.caches)

    def _evict_if_needed(self):
        """Evict LRU entries to satisfy: memory threshold, session count, total bytes.

        Pinned entries (via pin()) are never evicted.
        """
        if len(self.caches) == 0:
            return

        evicted_any = False
        # 1. Memory-pressure eviction (existing behavior)
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            if not self._evict_lru_once("memory pressure"):
                break
            evicted_any = True

        # 2. Per-instance session count cap
        if self._max_sessions is not None:
            while len(self.caches) > self._max_sessions:
                if not self._evict_lru_once(f"session cap {self._max_sessions}"):
                    break
                evicted_any = True

        # 3. Per-instance byte cap
        if self._max_bytes is not None:
            while len(self.caches) > 0 and self._total_bytes() > self._max_bytes:
                if not self._evict_lru_once(f"byte cap {self._max_bytes}"):
                    break
                evicted_any = True

        if evicted_any:
            # Force Python GC to release array references, then clear Metal buffer cache
            gc.collect()
            mx.clear_cache()

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


def _model_is_pipeline_parallel(model: Model) -> bool:
    """True iff the model has pipeline-parallel layer wrappers installed.

    Only the PP path is safe to combine with QuantizedKVCache right now:
    the single-node BatchGenerator code path in mlx-lm calls
    ``_merge_caches`` on every step (even for a single in-flight request),
    and QuantizedKVCache does not implement ``merge``. Attempting to use
    a quantized cache in that path crashes with::

        <class 'mlx_lm.models.cache.QuantizedKVCache'> does not yet
        support batching with history

    Detecting PP mode by layer type is cheap and avoids threading the
    distributed group through every cache call site.
    """
    try:
        from exo.worker.engines.mlx.auto_parallel import (
            PipelineFirstLayer,
            PipelineLastLayer,
        )
    except Exception:
        return False
    layers = getattr(model, "layers", None)
    if layers is None:
        return False
    for layer in layers:  # type: ignore[reportUnknownVariableType]
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        caches: KVCacheType = model.make_cache()  # type: ignore
        # Per-instance per-request token cap: wrap plain KVCache entries in
        # RotatingKVCache so a single request can't grow KV memory unbounded.
        # Skips quantization/turboquant/step-size paths since they target full
        # KVCache only and we're replacing those entries.
        if max_kv_size is not None:
            replaced = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    caches[i] = RotatingKVCache(max_size=max_kv_size, keep=keep)
                    replaced += 1
            if replaced:
                logger.info(
                    f"Capped KV cache at max_kv_size={max_kv_size} for {replaced}/{len(caches)} layers"
                )
            return caches
        if TURBOQUANT_ENABLED:
            from exo.worker.engines.mlx.turboquant_cache import (
                TurboQuantConfig,
                TurboQuantKVCache,
                patch_attention_dispatch,
            )
            patch_attention_dispatch()
            tq_config = TurboQuantConfig(
                bits=TURBOQUANT_BITS,
                group_size=CACHE_GROUP_SIZE,
                sketch_dim=TURBOQUANT_SKETCH_DIM,
                use_residual=TURBOQUANT_RESIDUAL,
            )
            replaced = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    caches[i] = TurboQuantKVCache(tq_config)
                    replaced += 1
            logger.info(
                f"Using TurboQuant KV cache (bits={TURBOQUANT_BITS}, sketch_dim={TURBOQUANT_SKETCH_DIM}) "
                f"for {replaced}/{len(caches)} layers"
            )
        elif KV_CACHE_BITS is not None and _model_is_pipeline_parallel(model):
            # Replace KVCache entries with QuantizedKVCache, but keep
            # ArraysCache (DeltaNet/SSM) and other cache types unchanged.
            # Restricted to PP mode: the single-node BatchGenerator path
            # calls _merge_caches which requires cache.merge(), and
            # QuantizedKVCache does not implement it.
            quantized = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    qc = QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                    qc.step = 16384
                    caches[i] = qc
                    quantized += 1
            logger.info(f"Using quantized KV cache (bits={KV_CACHE_BITS}, group_size={CACHE_GROUP_SIZE}) for {quantized}/{len(caches)} layers")
        else:
            if KV_CACHE_BITS is not None:
                logger.info(
                    f"EXO_KV_CACHE_BITS={KV_CACHE_BITS} ignored in single-node mode "
                    f"(QuantizedKVCache has no merge() support, required by BatchGenerator)"
                )
            else:
                logger.info("Using MLX LM's make cache")
            # Increase KVCache step size to reduce Metal allocator fragmentation.
            # Default step=256 causes a mx.concatenate expansion every prefill chunk,
            # fragmenting memory (~11 GB overhead at 24k tokens). A larger step lets
            # the cache pre-allocate and write in-place for most of the prefill.
            for c in caches:
                if isinstance(c, KVCache):
                    c.step = 16384
        return caches

    if max_kv_size is None:
        logger.info("Using default KV cache")
        return [KVCache() for _ in model.layers]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
