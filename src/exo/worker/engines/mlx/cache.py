import os
from copy import deepcopy
from typing import TYPE_CHECKING

import mlx.core as mx
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion

# Fraction of device memory above which LRU eviction kicks in
_DEFAULT_MEMORY_THRESHOLD = 0.9
_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _DEFAULT_MEMORY_THRESHOLD)
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
    def __init__(self, group: mx.distributed.Group | None = None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []  # per-entry media regions
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._group = group

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
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

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region. This prevents false
        cache hits when the same pad-token IDs encode different images/frames.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                # Validate media regions within the matched prefix
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length > best_length:
                best_index, best_length = i, length
            if length == max_length:
                is_exact = True
                best_index, best_length = i, length
                break

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        target = (max_length - 1) if is_exact else best_length
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_non_kv_caches(self.caches[best_index]):
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

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        """Truncate match_length if mismatched media regions are in the prefix.

        For each cached media region that falls within [0, match_length),
        check if the query has a region at the same position with the same
        content hash.  On first mismatch, truncate to the start of that
        cached region — the KV values beyond that point encode different
        media content and must not be reused.

        If the query has *no* media regions (text-only follow-up), cached
        regions that are fully below match_length are fine — their KV
        values are already baked in from a prior turn and the conversation
        token history includes them.
        """
        if not cached_regions:
            # Nothing to validate — cached entry is text-only
            return match_length

        # Build a lookup: start_pos → query region
        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                # Cached region is outside the matched prefix — irrelevant
                break
            # The matched prefix includes (part of) this cached region.
            # Check if the query has the same content at the same position.
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                # Query has no media here. This is OK — the cached KV
                # already includes the vision features from a prior turn,
                # and the text tokens after that region are what diverge.
                # But if the match extends *into* a region that the query
                # doesn't have at all, we should still truncate because
                # the token-ids match (pad tokens) but meanings differ.
                # However, if this is a text-only follow-up the query
                # tokens would diverge before the pad region, so
                # match_length wouldn't reach here. Safe to continue.
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}… "
                    f"query={query_r.content_hash[:12]}… — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break  # first mismatch truncates

        return match_length

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
            self._media_regions.pop(lru_index)
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
                c.state = [None] * len(c.state)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        else:
            c.trim(num_tokens)  # pyright: ignore[reportUnknownMemberType]


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    return max(getattr(c, "offset", 0) for c in cache)


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
