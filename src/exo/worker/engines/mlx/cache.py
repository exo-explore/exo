from copy import deepcopy

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.mlx import KVCacheType
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.utils_mlx import make_kv_cache
from exo.worker.runner.bootstrap import logger


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
        prompt: str,
    ) -> tuple[KVCacheType, mx.array]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        This method finds the best matching cached prefix and returns:
        - A copy of the cache trimmed to the prefix length
        - The remaining tokens that need to be prefilled before generation

        The caller is responsible for prefilling the remaining tokens.

        Returns:
            Tuple of (cache, remaining_tokens) where remaining_tokens are the
            tokens that still need to be prefilled/processed.
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
                f"(reusing {best_snapshot_length}, need to prefill {new_tokens})"
            )

            prompt_cache = deepcopy(self.caches[best_snapshot_index])

            # Trim removes tokens from the end, so we trim (cached_length - prefix_length) to keep the prefix
            cached_length = _cache_length(self.caches[best_snapshot_index])
            tokens_to_trim = cached_length - best_snapshot_length
            if tokens_to_trim > 0:
                trim_prompt_cache(prompt_cache, tokens_to_trim)

            # Return remaining tokens for caller to prefill
            remaining_tokens = tokenized_prompt[best_snapshot_length:]
            return prompt_cache, remaining_tokens

        else:
            prompt_cache = make_kv_cache(model)
            if len(self.prompts) == 0:
                logger.info(f"KV cache empty, need to prefill {max_length} tokens")
            else:
                logger.info(
                    f"KV cache no prefix match, need to prefill {max_length} tokens"
                )

            # Return all tokens for caller to prefill
            return prompt_cache, tokenized_prompt

    def encode_prompt(self, tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
        return encode_prompt(tokenizer, prompt)


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
    return max(c.offset for c in cache)


def _get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())
