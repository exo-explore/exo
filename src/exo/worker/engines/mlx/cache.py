from copy import deepcopy
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.mlx import KVCacheType
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.utils_mlx import make_kv_cache
from exo.worker.runner.bootstrap import logger

# Fraction of device memory above which LRU eviction kicks in
_MEMORY_PRESSURE_THRESHOLD = 0.85


class KVPrefixCache:
    def __init__(self):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._last_used.clear()

    def add_kv_cache(
        self, tokenizer: TokenizerWrapper, prompt: str, cache: KVCacheType
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        tokenized_prompt = encode_prompt(tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(f"KV cache added: {len(tokenized_prompt)} tokens")

    def update_kv_cache(
        self,
        index: int,
        tokenizer: TokenizerWrapper,
        prompt: str,
        cache: KVCacheType,
    ):
        """Update an existing cache entry in-place."""
        tokenized_prompt = encode_prompt(tokenizer, prompt)
        self.prompts[index] = tokenized_prompt
        self.caches[index] = deepcopy(cache)
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        logger.info(f"KV cache updated (index {index}): {len(tokenized_prompt)} tokens")

    def get_kv_cache(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        prompt: str,
    ) -> tuple[KVCacheType, mx.array, int | None]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
        """
        tokenized_prompt = encode_prompt(tokenizer, prompt)
        max_length = len(tokenized_prompt)

        best_snapshot_index, best_snapshot_length = None, 0

        for i, cached_prompt in enumerate(self.prompts):
            length = _get_prefix_length(tokenized_prompt, cached_prompt)

            if length == max_length:
                # Exact match - cached prompt starts with our entire prompt
                # Trim cache to prompt length - 1, return last token for stream_generate
                prompt_cache = deepcopy(self.caches[i])
                cached_length = _cache_length(self.caches[i])
                tokens_to_trim = cached_length - (max_length - 1)
                if tokens_to_trim > 0:
                    trim_prompt_cache(cast(list[Any], prompt_cache), tokens_to_trim)
                self._access_counter += 1
                self._last_used[i] = self._access_counter
                logger.info(f"KV cache exact match: {max_length} tokens (instant)")
                return prompt_cache, tokenized_prompt[-1:], i

            if length > best_snapshot_length:
                best_snapshot_index, best_snapshot_length = i, length

        if best_snapshot_index is not None:
            new_tokens = max_length - best_snapshot_length
            logger.info(
                f"KV cache prefix match: {best_snapshot_length}/{max_length} tokens "
                f"(reusing {best_snapshot_length}, need to prefill {new_tokens})"
            )

            prompt_cache = deepcopy(self.caches[best_snapshot_index])

            # Trim removes tokens from the end, so we trim (cached_length - prefix_length) to keep the prefix
            cached_length = _cache_length(self.caches[best_snapshot_index])
            tokens_to_trim = cached_length - best_snapshot_length
            if tokens_to_trim > 0:
                trim_prompt_cache(cast(list[Any], prompt_cache), tokens_to_trim)

            self._access_counter += 1
            self._last_used[best_snapshot_index] = self._access_counter
            remaining_tokens = tokenized_prompt[best_snapshot_length:]
            return prompt_cache, remaining_tokens, best_snapshot_index

        else:
            prompt_cache = make_kv_cache(model)
            if len(self.prompts) == 0:
                logger.info(f"KV cache empty, need to prefill {max_length} tokens")
            else:
                logger.info(
                    f"KV cache no prefix match, need to prefill {max_length} tokens"
                )

            return prompt_cache, tokenized_prompt, None

    def _evict_if_needed(self):
        """Evict least recently used entries while memory pressure is high."""
        if len(self.caches) == 0:
            return

        active: int = mx.metal.get_active_memory()
        limit = int(mx.metal.device_info()["max_recommended_working_set_size"])
        if active < limit * _MEMORY_PRESSURE_THRESHOLD:
            return

        # Evict LRU entries until below threshold or only one entry left
        while len(self.caches) > 0:
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._last_used.pop(lru_index)
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory pressure"
            )

            active = mx.metal.get_active_memory()
            if active < limit * _MEMORY_PRESSURE_THRESHOLD:
                break


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(tokenized_prompt)


def _cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    # Use .offset attribute which all cache types have (len() not implemented in older QuantizedKVCache)
    return max(c.offset for c in cache)  # type: ignore


def _get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())
