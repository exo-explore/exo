# type: ignore
# TODO: Fix this file, including types!
from copy import deepcopy
from typing import Callable

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import _BaseCache, trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KEEP_KV_SIZE, KV_BITS, KV_GROUP_SIZE
from exo.worker.engines.mlx.utils_mlx import make_kv_cache


class KVPrefixCache:
    def __init__(self):
        # Only one prefix cache per runner.
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[list[_BaseCache]] = []

    def add_kv_cache(
        self, tokenizer: TokenizerWrapper, prompt: str, cache: list[_BaseCache]
    ):
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))

    def get_kv_cache(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        sampler: Callable[[mx.array], mx.array],
        prompt: str,
    ) -> list[_BaseCache]:
        tokenized_prompt = self.encode_prompt(tokenizer, prompt)
        max_length = len(tokenized_prompt)

        best_snapshot_index, best_snapshot_length = None, 0

        for i, cached_prompt in enumerate(self.prompts):
            length = _get_prefix_length(tokenized_prompt, cached_prompt)

            if length == max_length:
                return self.caches[i]

            if length > best_snapshot_length:
                best_snapshot_index, best_snapshot_length = i, length

        if best_snapshot_index is not None:
            prompt_cache = deepcopy(self.caches[best_snapshot_index])
            trim_prompt_cache(prompt_cache, max_length - best_snapshot_length)
            tokenized_prompt = tokenized_prompt[best_snapshot_index:]

        else:
            prompt_cache = make_kv_cache(
                model,
                # max_kv_size=MAX_KV_SIZE,
                # keep=KEEP_KV_SIZE
            )

        prefill(model, tokenizer, sampler, tokenized_prompt, prompt_cache)

        return prompt_cache

    def encode_prompt(self, tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        tokenized_prompt = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )
        return mx.array(tokenized_prompt)


def _get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]), KEEP_KV_SIZE)
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
    cache: list[_BaseCache],
) -> None:
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=0,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        pass
