# type: ignore
# TODO: Fix this file, including types!
from copy import deepcopy
from typing import Callable

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    trim_prompt_cache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE
from exo.worker.engines.mlx.utils_mlx import make_kv_cache
from exo.worker.runner.bootstrap import logger

# Type alias matching make_kv_cache return type
KVCacheType = list[KVCache | RotatingKVCache | QuantizedKVCache]


class KVPrefixCache:
    def __init__(self):
        # Only one prefix cache per runner.
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()

    def add_kv_cache(
        self, tokenizer: TokenizerWrapper, prompt: str, cache: KVCacheType
    ):
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))
        logger.info(f"KV cache saved: {len(tokenized_prompt)} tokens")

    def get_kv_cache(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        sampler: Callable[[mx.array], mx.array],
        prompt: str,
    ) -> tuple[KVCacheType, mx.array]:
        """Get KV cache for prompt, prefilling any new tokens.

        Returns:
            Tuple of (cache, remaining_tokens) where remaining_tokens are the
            tokens that still need to be processed by stream_generate.
        """
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
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
                    trim_prompt_cache(prompt_cache, tokens_to_trim)
                logger.info(f"KV cache exact match: {max_length} tokens (instant)")
                return prompt_cache, tokenized_prompt[-1:]

            if length > best_snapshot_length:
                best_snapshot_index, best_snapshot_length = i, length

        if best_snapshot_index is not None:
            new_tokens = max_length - best_snapshot_length
            logger.info(
                f"KV cache prefix match: {best_snapshot_length}/{max_length} tokens "
                f"(reusing {best_snapshot_length}, processing {new_tokens} new)"
            )

            prompt_cache = deepcopy(self.caches[best_snapshot_index])

            # Trim removes tokens from the end, so we trim (cached_length - prefix_length) to keep the prefix
            cached_length = _cache_length(self.caches[best_snapshot_index])
            tokens_to_trim = cached_length - best_snapshot_length
            if tokens_to_trim > 0:
                trim_prompt_cache(prompt_cache, tokens_to_trim)

            # Prefill the remaining tokens (except the last one which stream_generate needs)
            remaining_tokens = tokenized_prompt[best_snapshot_length:]
            if len(remaining_tokens) > 1:
                prefill(model, tokenizer, sampler, remaining_tokens[:-1], prompt_cache)

            # Return last token for stream_generate to start from
            return prompt_cache, remaining_tokens[-1:]

        else:
            prompt_cache = make_kv_cache(model)
            if len(self.prompts) == 0:
                logger.info("KV cache empty, created new")
            else:
                logger.info("KV cache no prefix match, starting fresh")

            # Prefill all but last token, return last token for stream_generate
            if len(tokenized_prompt) > 1:
                prefill(model, tokenizer, sampler, tokenized_prompt[:-1], prompt_cache)
            return prompt_cache, tokenized_prompt[-1:]

    def encode_prompt(self, tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
        return encode_prompt(tokenizer, prompt)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array."""
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    return mx.array(tokenized_prompt)


def _cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    # Use .offset attribute which all cache types have (len() not implemented in older QuantizedKVCache)
    return max(c.offset for c in cache)


def _get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = (prompt[:n] == cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt: mx.array,
    cache: KVCacheType,
) -> None:
    # Use max_tokens=1 because max_tokens=0 is buggy in some mlx_lm versions
    # We just throw away the generated token - we only care about filling the cache
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        break  # Stop after first iteration - cache is now filled
    # Trim the extra token we generated (max_tokens=1 workaround)
    trim_prompt_cache(cache, 1)
