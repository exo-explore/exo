"""KV prefix cache for reusing computed prompt prefixes across requests."""

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Protocol

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import trim_prompt_cache

from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.utils_mlx import make_kv_cache
from exo.worker.runner.bootstrap import logger

# Type alias for KV cache - the actual type is _BaseCache but it's private
KVCacheType = Any


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers used with KVPrefixCache."""

    bos_token: str | None

    def encode(self, text: str, **kwargs: bool) -> list[int]: ...


class KVPrefixCache:
    """Cache for common prompt prefixes to avoid re-processing.

    Uses LRU eviction when capacity is reached. Stores tokenized prompts
    and their corresponding KV caches for reuse.
    """

    def __init__(self, max_size: int = 10):
        """Initialize prefix cache.

        Args:
            max_size: Maximum number of cached entries before LRU eviction.
        """
        self.max_size = max_size
        # OrderedDict maintains insertion order for LRU - most recent at end
        # Key: token bytes, Value: (tokens as mx.array, KV cache)
        self._cache: OrderedDict[bytes, tuple[mx.array, list[KVCacheType]]] = (
            OrderedDict()
        )

    def _token_key(self, tokens: mx.array) -> bytes:
        """Create hashable key from token array."""
        return np.array(tokens.tolist(), dtype=np.int32).tobytes()

    def _encode_prompt(self, tokenizer: TokenizerProtocol, prompt: str) -> mx.array:
        """Tokenize prompt string to mx.array."""
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        tokenized = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokenized)

    def _find_best_prefix(self, tokens: mx.array) -> tuple[bytes | None, int]:
        """Find cached entry with longest matching prefix.

        Returns:
            Tuple of (cache_key, prefix_length). cache_key is None if no match found.
        """
        best_key: bytes | None = None
        best_length = 0
        target_len = tokens.shape[0]

        for key, (cached_tokens, _cache) in self._cache.items():
            prefix_len = get_prefix_length(tokens, cached_tokens)

            # Exact match - return immediately
            if prefix_len == target_len and prefix_len == cached_tokens.shape[0]:
                return key, prefix_len

            # Better prefix match
            if prefix_len > best_length:
                best_key = key
                best_length = prefix_len

        return best_key, best_length

    def get_kv_cache(
        self,
        model: Model,
        tokenizer: TokenizerProtocol,
        sampler: Callable[[mx.array], mx.array],
        prompt: str,
    ) -> tuple[list[KVCacheType], int]:
        """Get KV cache for prompt, reusing prefix if available.

        Args:
            model: The model to create cache for.
            tokenizer: Tokenizer for encoding prompt.
            sampler: Sampler function for prefill.
            prompt: The prompt string to process.

        Returns:
            Tuple of (kv_cache, tokens_reused). tokens_reused indicates how many
            tokens were reused from cache (0 if no cache hit).
        """
        tokens = self._encode_prompt(tokenizer, prompt)
        target_len = int(tokens.shape[0])

        # Find best prefix match
        best_key, prefix_len = self._find_best_prefix(tokens)

        if best_key is not None and prefix_len > 0:
            cached_tokens, cached_kv = self._cache[best_key]
            cached_len = int(cached_tokens.shape[0])

            # Move to end (most recently used)
            self._cache.move_to_end(best_key)

            if prefix_len == target_len and prefix_len == cached_len:
                # Exact match - return deepcopy directly
                logger.debug(f"Prefix cache: exact match, reusing {prefix_len} tokens")
                return deepcopy(cached_kv), prefix_len

            # Partial match - need to trim and/or extend
            prompt_cache = deepcopy(cached_kv)

            if cached_len > prefix_len:
                # Cached prompt is longer - trim to prefix length
                num_to_trim = cached_len - prefix_len
                trim_prompt_cache(prompt_cache, num_to_trim)
                logger.debug(
                    f"Prefix cache: trimmed {num_to_trim} tokens from cached entry"
                )

            # Note: We don't prefill remaining tokens here - stream_generate will do it
            # when processing the full prompt with this partial cache
            return prompt_cache, prefix_len

        # No cache hit - return fresh cache (stream_generate will prefill)
        logger.debug(
            f"Prefix cache: miss, will prefill {target_len} tokens during generation"
        )
        prompt_cache = make_kv_cache(model=model)

        return prompt_cache, 0

    def put(
        self, tokenizer: TokenizerProtocol, prompt: str, cache: list[KVCacheType]
    ) -> None:
        """Store KV cache for prompt after generation completes.

        Args:
            tokenizer: Tokenizer for encoding prompt.
            prompt: The prompt string that was processed.
            cache: The KV cache to store.
        """
        tokens = self._encode_prompt(tokenizer, prompt)
        key = self._token_key(tokens)

        # If already in cache, just move to end
        if key in self._cache:
            self._cache.move_to_end(key)
            return

        # Evict LRU entry if at capacity
        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(
                f"Prefix cache: evicted LRU entry ({len(evicted_key)} token bytes)"
            )

        # Store deepcopy
        self._cache[key] = (tokens, deepcopy(cache))
        logger.debug(f"Prefix cache: stored entry with {tokens.shape[0]} tokens")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Calculate length of matching prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())
