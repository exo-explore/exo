import os
from copy import deepcopy

import mlx.core as mx
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.runner.bootstrap import logger

# Fraction of device memory above which LRU eviction kicks in
_DEFAULT_MEMORY_THRESHOLD = 0.9
_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _DEFAULT_MEMORY_THRESHOLD)
)


class SSMSnapshot:
    """Snapshot of ArraysCache states at a known token position."""

    def __init__(self, states: list[list[object] | None], token_count: int):
        self.states = states
        self.token_count = token_count


def snapshot_ssm_states(cache: KVCacheType) -> SSMSnapshot:
    states: list[list[object] | None] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(deepcopy(c.state))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        else:
            states.append(None)
    token_count = cache_length(cache)
    return SSMSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[SSMSnapshot],
    target_token_count: int,
) -> SSMSnapshot | None:
    best: SSMSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def has_ssm_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(isinstance(c, ArraysCache) for c in cache)


class KVPrefixCache:
    def __init__(
        self, tokenizer: TokenizerWrapper, group: mx.distributed.Group | None = None
    ):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._ssm_snapshots: list[list[SSMSnapshot] | None] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._tokenizer: TokenizerWrapper = tokenizer
        self._group = group

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._ssm_snapshots.clear()
        self._last_used.clear()

    def add_kv_cache(
        self,
        prompt: str,
        cache: KVCacheType,
        ssm_snapshots: list[SSMSnapshot] | None = None,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        tokenized_prompt = encode_prompt(self._tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))
        self._ssm_snapshots.append(ssm_snapshots)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(f"KV cache added: {len(tokenized_prompt)} tokens")

    def update_kv_cache(
        self,
        index: int,
        prompt: str,
        cache: KVCacheType,
        ssm_snapshots: list[SSMSnapshot] | None,
        restore_pos: int,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._ssm_snapshots[index]
        merged: list[SSMSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if ssm_snapshots:
            merged.extend(ssm_snapshots)

        tokenized_prompt = encode_prompt(self._tokenizer, prompt)
        self.prompts[index] = tokenized_prompt
        self.caches[index] = deepcopy(cache)
        self._ssm_snapshots[index] = merged or None
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        logger.info(f"KV cache updated (index {index}): {len(tokenized_prompt)} tokens")

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, SSMSnapshot | None]:
        if not has_ssm_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._ssm_snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt: str,
    ) -> tuple[KVCacheType, mx.array, int | None]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        """
        tokenized_prompt = encode_prompt(self._tokenizer, prompt)
        max_length = len(tokenized_prompt)

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(tokenized_prompt, cached_prompt)
            if length > best_length:
                best_index, best_length = i, length
            if length == max_length:
                is_exact = True
                best_index, best_length = i, length
                break

        if best_index is None:
            if len(self.prompts) == 0:
                logger.info(f"KV cache empty, need to prefill {max_length} tokens")
            else:
                logger.info(
                    f"KV cache no prefix match, need to prefill {max_length} tokens"
                )
            return make_kv_cache(model), tokenized_prompt, None

        # For exact match we trim to max_length-1
        target = (max_length - 1) if is_exact else best_length
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # SSM model with no usable snapshot — need fresh cache
        if (
            restore_pos == 0
            and restore_snap is None
            and has_ssm_caches(self.caches[best_index])
        ):
            match_kind = (
                "exact match"
                if is_exact
                else f"prefix match at {best_length}/{max_length}"
            )
            logger.info(
                f"KV cache {match_kind} but no SSM snapshot, "
                f"need to prefill {max_length} tokens"
            )
            return make_kv_cache(model), tokenized_prompt, None

        prompt_cache = deepcopy(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = tokenized_prompt[restore_pos:]

        if is_exact:
            logger.info(
                f"KV cache exact match: {max_length} tokens "
                f"(reusing {restore_pos}, re-processing {len(remaining)})"
            )
        else:
            logger.info(
                f"KV cache prefix match: {best_length}/{max_length} tokens "
                f"(restoring to {restore_pos}, need to prefill {len(remaining)})"
            )
        return prompt_cache, remaining, best_index

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0 and
            self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._ssm_snapshots.pop(lru_index)
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


def _fix_unmatched_think_end_tokens(tokens: mx.array, tokenizer: TokenizerWrapper) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    think_start_id: int = tokenizer.think_start_id  # type: ignore[attr-defined]
    think_end_id: int = tokenizer.think_end_id  # type: ignore[attr-defined]
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []
    depth = 0
    for token in token_list:
        if token == think_start_id:
            depth += 1
        elif token == think_end_id:
            if depth == 0:
                result.append(think_start_id)
            else:
                depth -= 1
        result.append(token)
    if len(result) == len(token_list):
        return tokens
    return mx.array(result)


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    ssm_snapshot: SSMSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            if ssm_snapshot is not None and ssm_snapshot.states[i] is not None:
                c.state = deepcopy(ssm_snapshot.states[i])
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
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = mx.array(tokenized_prompt)
    return _fix_unmatched_think_end_tokens(tokens, tokenizer)


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

    # TODO: Do this for all models
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
