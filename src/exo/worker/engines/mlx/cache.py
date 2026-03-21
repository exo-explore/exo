from __future__ import annotations

import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

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
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.remote_cache_tier import RemoteCacheTierProtocol


# ---------------------------------------------------------------------------
# SnapKV constants
# ---------------------------------------------------------------------------

# Minimum prompt length (tokens) before SnapKV compression is attempted.
SNAPKV_THRESHOLD: int = int(os.environ.get("EXO_SNAPKV_THRESHOLD", "2048"))

# Tokens kept at the start of the prompt (system prompt anchor).
SNAPKV_ANCHOR_TOKENS: int = int(os.environ.get("EXO_SNAPKV_ANCHOR", "64"))

# Tokens always kept at the end (most-recent context window).
SNAPKV_LOCAL_WINDOW: int = int(os.environ.get("EXO_SNAPKV_LOCAL_WINDOW", "256"))

# Maximum number of "important" tokens from the middle band to retain.
SNAPKV_MAX_CAPACITY_PROMPT: int = int(os.environ.get("EXO_SNAPKV_MAX_CAPACITY", "2048"))

# Kernel size for average-pooling key scores before top-k selection.
SNAPKV_POOLING_KERNEL_SIZE: int = int(os.environ.get("EXO_SNAPKV_KERNEL", "5"))

# Master on/off switch — must set EXO_SNAPKV=1 to enable.
_SNAPKV_ENABLED: bool = os.environ.get("EXO_SNAPKV", "0").strip() == "1"


# ---------------------------------------------------------------------------
# SnapKV runtime statistics accumulator
# ---------------------------------------------------------------------------

# Bytes per float32 element (keys and values both stored as float32).
_BYTES_PER_ELEMENT: int = 4


@dataclass
class _SnapKVStats:
    """Mutable accumulator for SnapKV compression statistics."""

    tokens_before: int = field(default=0)
    tokens_after: int = field(default=0)
    compression_calls: int = field(default=0)
    # Tracks *approximate* memory saved: tokens_removed * heads * head_dim * 2 (K+V) * bytes_per_elem.
    # We use a simple heuristic here (layer count unknown at accumulation time)
    # so memory_saved_bytes is computed from the token delta only.
    memory_saved_bytes: int = field(default=0)


_SNAPKV_STATS: _SnapKVStats = _SnapKVStats()


# ---------------------------------------------------------------------------
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
    def __init__(
        self,
        group: mx.distributed.Group | None,
        remote_tier: RemoteCacheTierProtocol | None = None,
    ):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._group = group
        self._entry_ids: list[str] = []
        self._entry_counter: int = 0
        self._remote_tier = remote_tier

    def clear(self):
        """Clear all cached prompts and caches."""
        if self._remote_tier is not None:
            for eid in self._entry_ids:
                self._remote_tier.remove(eid)
        self._entry_ids.clear()
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
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        entry_id = f"kvc-{id(self)}-{self._entry_counter}"
        self._entry_counter += 1
        self._entry_ids.append(entry_id)
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        if self._remote_tier is not None:
            self._remote_tier.store_async(entry_id, prompt_tokens, cache)
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
        if self._remote_tier is not None:
            self._remote_tier.store_async(self._entry_ids[index], prompt_tokens, cache)
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

        if best_index is None:
            if self._remote_tier is not None:
                remote_result = self._remote_tier.fetch(prompt_tokens, model)
                if remote_result is not None:
                    remote_cache, matched_tokens = remote_result
                    return remote_cache, prompt_tokens[matched_tokens:], None
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
            if self._remote_tier is not None:
                self._remote_tier.remove(self._entry_ids[lru_index])
            self._entry_ids.pop(lru_index)
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._last_used.pop(lru_index)
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

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


def compress_kv_cache(
    cache: KVCacheType,
    threshold: int = 2048,
    anchor_tokens: int = 64,
    recent_tokens: int = 512,
    important_tokens: int = 512,
) -> int:
    """SnapKV-style KV cache compression.

    When the cache exceeds *threshold* tokens, it is compressed to at most
    ``anchor_tokens + recent_tokens + important_tokens`` tokens by:

    1. Keeping the first ``anchor_tokens`` positions (prompt anchor).
    2. Keeping the last ``recent_tokens`` positions (recent context).
    3. Scoring each remaining position by the L2-norm of its key vectors
       (averaged across heads and batch), then keeping the top
       ``important_tokens`` scoring positions.

    Only plain :class:`KVCache` layers are compressed; ``QuantizedKVCache``,
    ``RotatingKVCache``, and ``ArraysCache`` layers are left untouched because
    their internal invariants differ.  The function is a no-op when the cache
    length is at or below *threshold*.

    Returns the new cache length after compression (or the original length if
    compression was skipped).
    """
    current_length = cache_length(cache)
    if current_length <= threshold:
        return current_length

    budget = anchor_tokens + recent_tokens + important_tokens
    if current_length <= budget:
        return current_length

    # Middle band start: everything after the anchor is a candidate.
    middle_start = anchor_tokens

    for layer_cache in cache:
        if not isinstance(layer_cache, KVCache):
            # Skip non-plain caches — their invariants differ.
            continue
        # KVCache.keys starts as None at runtime despite the stub type being mx.array.
        # getattr with sentinel is the only way to check without a type-ignore.
        layer_keys = getattr(layer_cache, "keys", None)
        layer_values = getattr(layer_cache, "values", None)
        if not isinstance(layer_keys, mx.array) or not isinstance(
            layer_values, mx.array
        ):
            continue

        effective_length = layer_cache.offset
        if effective_length <= threshold:
            continue

        keys = layer_keys[..., :effective_length, :]  # (B, H, T, D)
        values = layer_values[..., :effective_length, :]

        # --- Build the index set of tokens to keep ---
        anchor_idx = mx.arange(anchor_tokens)
        recent_start = effective_length - recent_tokens
        recent_idx = mx.arange(recent_start, effective_length)

        mid_end = effective_length - recent_tokens
        mid_len = mid_end - middle_start

        if mid_len > 0 and important_tokens > 0:
            mid_keys = keys[..., middle_start:mid_end, :]  # (B, H, mid_len, D)
            # Score = mean L2-norm across batch and heads → shape (mid_len,)
            scores = mx.mean(
                mx.sqrt(mx.sum(mid_keys * mid_keys, axis=-1)),
                axis=(0, 1),
            )  # (mid_len,)
            mx.eval(scores)
            scores_list = cast(list[float], scores.tolist())
            # Top-k indices within the middle band
            k = min(important_tokens, mid_len)
            top_local = sorted(range(mid_len), key=lambda i: scores_list[i], reverse=True)[:k]
            top_local_sorted = sorted(top_local)
            important_idx = mx.array([middle_start + i for i in top_local_sorted])
        else:
            important_idx = mx.array([], dtype=mx.int32)

        # Concatenate all kept indices (already sorted: anchor < important < recent)
        if len(important_idx) > 0:
            keep_idx = mx.concatenate([anchor_idx, important_idx, recent_idx])
        else:
            keep_idx = mx.concatenate([anchor_idx, recent_idx])

        mx.eval(keep_idx)
        new_len = int(keep_idx.shape[0])

        # Gather selected keys and values along the sequence axis.
        new_keys = keys[..., keep_idx, :]  # (B, H, new_len, D)
        new_values = values[..., keep_idx, :]
        mx.eval(new_keys, new_values)

        layer_cache.keys = new_keys
        layer_cache.values = new_values
        layer_cache.offset = new_len

    new_length = cache_length(cache)
    logger.info(
        f"KV cache compressed: {current_length} → {new_length} tokens "
        f"(anchor={anchor_tokens}, important={important_tokens}, recent={recent_tokens})"
    )
    return new_length


def snapkv_compress(
    cache: KVCacheType,
    *,
    threshold: int = SNAPKV_THRESHOLD,
    anchor_tokens: int = SNAPKV_ANCHOR_TOKENS,
    local_window: int = SNAPKV_LOCAL_WINDOW,
    max_capacity_prompt: int = SNAPKV_MAX_CAPACITY_PROMPT,
    pooling_kernel_size: int = SNAPKV_POOLING_KERNEL_SIZE,
) -> int:
    """SnapKV-style KV cache compression with average-pooling score smoothing.

    Identical to :func:`compress_kv_cache` but applies a sliding-window
    average-pool over neighbouring positions (kernel size = *pooling_kernel_size*)
    before the top-k selection step.  This matches the smoothing strategy in the
    original SnapKV paper and avoids keeping isolated high-norm "spike" positions
    that carry little semantic weight.

    Parameters
    ----------
    cache:
        The KV cache to compress in-place.
    threshold:
        Minimum cache length before compression is attempted.
    anchor_tokens:
        Number of leading tokens always kept (system prompt anchor).
    local_window:
        Number of trailing tokens always kept (recency window).
    max_capacity_prompt:
        Maximum number of "important" middle-band tokens to keep.
    pooling_kernel_size:
        Width of the average-pooling kernel applied to per-position L2-norm
        scores before top-k selection.  Must be >= 1.  When 1 this function
        is equivalent to :func:`compress_kv_cache`.

    Returns
    -------
    int
        New cache length (unchanged if compression was skipped).
    """
    current_length = cache_length(cache)
    if current_length <= threshold:
        return current_length

    budget = anchor_tokens + local_window + max_capacity_prompt
    if current_length <= budget:
        return current_length

    kernel = max(1, pooling_kernel_size)
    middle_start = anchor_tokens

    for layer_cache in cache:
        if not isinstance(layer_cache, KVCache):
            # Skip QuantizedKVCache, RotatingKVCache, ArraysCache — their
            # internal invariants differ from plain KVCache.
            continue

        layer_keys = getattr(layer_cache, "keys", None)
        layer_values = getattr(layer_cache, "values", None)
        if not isinstance(layer_keys, mx.array) or not isinstance(layer_values, mx.array):
            continue

        effective_length = layer_cache.offset
        if effective_length <= threshold:
            continue

        keys = layer_keys[..., :effective_length, :]  # (B, H, T, D)
        values = layer_values[..., :effective_length, :]

        mid_end = effective_length - local_window
        mid_len = mid_end - middle_start

        # Work in numpy for scoring and gathering to avoid Metal fancy-index
        # kernel compilation issues on some macOS versions.
        keys_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.array(
            keys, dtype=np.float32
        )  # (B, H, T, D)
        values_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.array(
            values, dtype=np.float32
        )

        anchor_keys_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
            keys_np[..., :anchor_tokens, :]
        )
        anchor_values_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
            values_np[..., :anchor_tokens, :]
        )
        recent_keys_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
            keys_np[..., (effective_length - local_window):, :]
        )
        recent_values_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
            values_np[..., (effective_length - local_window):, :]
        )

        if mid_len > 0 and max_capacity_prompt > 0:
            mid_keys_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
                keys_np[..., middle_start:mid_end, :]  # (B, H, mid_len, D)
            )
            # L2-norm per position, averaged across batch and heads → (mid_len,)
            # cast() is needed because numpy ufunc stubs return Any.
            squared_sum = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.float32]],
                np.sum(mid_keys_np * mid_keys_np, axis=-1),
            )
            norms = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.float32]],
                np.sqrt(squared_sum),
            )
            raw_scores_np = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.float32]],
                np.mean(norms, axis=(0, 1)),
            )
            scores_list: list[float] = [float(v) for v in raw_scores_np.ravel()]

            # Average-pool with a sliding window to smooth isolated spikes.
            if kernel > 1 and mid_len >= kernel:
                half = kernel // 2
                smoothed: list[float] = []
                for position in range(mid_len):
                    lo = max(0, position - half)
                    hi = min(mid_len, position + half + 1)
                    window_scores = scores_list[lo:hi]
                    smoothed.append(sum(window_scores) / len(window_scores))
            else:
                smoothed = scores_list

            k = min(max_capacity_prompt, mid_len)
            top_local = sorted(range(mid_len), key=lambda i: smoothed[i], reverse=True)[:k]
            top_local_sorted = sorted(top_local)

            mid_keys_selected: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
                mid_keys_np[..., top_local_sorted, :]
            )
            mid_vals_band: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
                values_np[..., middle_start:mid_end, :]
            )
            mid_values_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
                mid_vals_band[..., top_local_sorted, :]
            )
            new_keys_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.concatenate(
                [anchor_keys_np, mid_keys_selected, recent_keys_np], axis=-2
            )
            new_values_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.concatenate(
                [anchor_values_np, mid_values_np, recent_values_np], axis=-2
            )
        else:
            new_keys_np = np.concatenate([anchor_keys_np, recent_keys_np], axis=-2)
            new_values_np = np.concatenate([anchor_values_np, recent_values_np], axis=-2)

        new_len = int(new_keys_np.shape[-2])
        layer_cache.keys = mx.array(new_keys_np)
        layer_cache.values = mx.array(new_values_np)
        layer_cache.offset = new_len

    new_length = cache_length(cache)
    logger.info(
        f"SnapKV compressed: {current_length} → {new_length} tokens "
        f"(anchor={anchor_tokens}, important={max_capacity_prompt}, "
        f"local_window={local_window}, kernel={kernel})"
    )
    return new_length


def snapkv_maybe_compress(cache: KVCacheType) -> int:
    """Apply SnapKV compression when EXO_SNAPKV=1 and the cache is long enough.

    This is the single call-site integration point: call it immediately after
    prefill completes.  Returns the new cache length (unchanged if disabled or
    below the threshold).

    Side-effect: updates the module-level :data:`_SNAPKV_STATS` accumulator.
    """
    if not _SNAPKV_ENABLED:
        return cache_length(cache)

    length_before = cache_length(cache)
    length_after = snapkv_compress(cache)

    if length_before > length_after:
        tokens_removed = length_before - length_after
        # Heuristic memory estimate: 2 (K+V) × float32 bytes × removed tokens.
        # Head count and head-dim are not available here without the model, so
        # we conservatively use 1 as a multiplier (callers may scale separately).
        _SNAPKV_STATS.tokens_before += length_before
        _SNAPKV_STATS.tokens_after += length_after
        _SNAPKV_STATS.compression_calls += 1
        _SNAPKV_STATS.memory_saved_bytes += tokens_removed * 2 * _BYTES_PER_ELEMENT

    return length_after


def snapkv_stats() -> dict[str, float | int]:
    """Return a snapshot of SnapKV compression statistics since process start.

    Returns a plain dict with the following keys:

    * ``compression_ratio`` – ratio of tokens retained to tokens originally
      seen (lower is better compression).  ``1.0`` if no compression has
      occurred yet.
    * ``tokens_processed`` – total number of input tokens seen across all
      compression calls.
    * ``tokens_retained`` – total output tokens after compression.
    * ``compression_calls`` – number of times compression was triggered.
    * ``memory_saved_mb`` – approximate device-memory saved in MiB based on
      a heuristic of 2 × float32 bytes per removed token position.  This is a
      *lower bound*; actual savings scale with number of layers, heads, and
      head dimension.
    """
    s = _SNAPKV_STATS
    ratio = (s.tokens_after / s.tokens_before) if s.tokens_before > 0 else 1.0
    memory_saved_mb = s.memory_saved_bytes / (1024 * 1024)
    return {
        "compression_ratio": ratio,
        "tokens_processed": s.tokens_before,
        "tokens_retained": s.tokens_after,
        "compression_calls": s.compression_calls,
        "memory_saved_mb": memory_saved_mb,
    }


def snapkv_benchmark(seq_len: int = 16384) -> dict[str, float]:
    """Measure SnapKV compression time and approximate memory delta.

    Constructs a synthetic single-layer KV cache of *seq_len* tokens (float32,
    1 batch, 32 heads, 128 head-dim), runs :func:`snapkv_compress` once, and
    reports wall-clock time and the memory-usage delta measured via
    ``psutil``.

    Parameters
    ----------
    seq_len:
        Number of tokens in the synthetic cache.  Default 16 384.

    Returns
    -------
    dict with keys:
        * ``seq_len`` – the requested sequence length.
        * ``new_len`` – compressed length in tokens.
        * ``compression_ratio`` – ``new_len / seq_len``.
        * ``elapsed_seconds`` – wall-clock time for the compress call.
        * ``rss_before_mb`` – RSS before compression (MiB).
        * ``rss_after_mb`` – RSS after compression (MiB).
        * ``rss_delta_mb`` – RSS change (positive = memory freed).
    """
    batch_size = 1
    num_heads = 32
    head_dim = 128
    shape = (batch_size, num_heads, seq_len, head_dim)

    # Build a synthetic single-layer cache.
    synthetic_cache: KVCache = KVCache()
    synthetic_cache.keys = mx.random.normal(shape).astype(mx.float32)
    synthetic_cache.values = mx.random.normal(shape).astype(mx.float32)
    synthetic_cache.offset = seq_len
    mx.eval(synthetic_cache.keys, synthetic_cache.values)

    cache: KVCacheType = [synthetic_cache]

    process = psutil.Process()
    rss_before = process.memory_info().rss

    start = time.perf_counter()
    new_len = snapkv_compress(cache, threshold=0)  # threshold=0 forces compression
    elapsed = time.perf_counter() - start

    rss_after = process.memory_info().rss

    rss_before_mb = rss_before / (1024 * 1024)
    rss_after_mb = rss_after / (1024 * 1024)
    rss_delta_mb = (rss_before - rss_after) / (1024 * 1024)

    return {
        "seq_len": float(seq_len),
        "new_len": float(new_len),
        "compression_ratio": new_len / seq_len if seq_len > 0 else 1.0,
        "elapsed_seconds": elapsed,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": rss_delta_mb,
    }


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
