# type: ignore
import time
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    _cache_length,
    _get_prefix_length,
    encode_prompt,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate, prefill
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, make_kv_cache
from exo.worker.tests.unittests.test_mlx.conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    DEFAULT_GPT_OSS_MODEL_ID,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert _get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert _get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert _get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert _get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert _get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert _get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert _get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert _get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert _get_prefix_length(a, b) == 0


class TestKVPrefix:
    def test_starts_empty(self):
        cache = KVPrefixCache()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self):
        cache = KVPrefixCache()
        cache.prompts.append(mx.array([1, 2, 3]))
        cache.caches.append([KVCache()])
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self):
        cache = KVPrefixCache()
        cache.clear()
        assert len(cache.prompts) == 0


def _load_gpt_oss() -> tuple[Model, object]:
    from mlx_lm.utils import load_model

    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    model_path = DEFAULT_GPT_OSS_CONFIG.model_path
    model_id = ModelId(DEFAULT_GPT_OSS_MODEL_ID)

    model, _ = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    return cast(Model, model), tokenizer


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_model_exists(),
    reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
)
class TestKVPrefixCacheWithModel:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = _load_gpt_oss()
        return model, tokenizer

    def test_prefill_populates_cache(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello!!")],
            max_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        prefill(model, tokenizer, make_sampler(0.0), tokens, cache)

        # Cache should now hold the prompt tokens
        assert _cache_length(cache) == len(tokens)

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Test exact")],
            max_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        prefill(model, tokenizer, make_sampler(0.0), tokens, cache)

        kv_prefix_cache = KVPrefixCache()
        kv_prefix_cache.add_kv_cache(tokenizer, prompt, cache)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = _cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # Retrieve with same prompt: exact match
        result_cache, remaining_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, tokenizer, prompt
        )
        assert matched_index == 0

        # Exact match returns only last token
        assert len(remaining_tokens) == 1
        assert mx.array_equal(remaining_tokens, tokens[-1:])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """get_kv_cache with a longer prompt sharing prefix should return partial match."""
        model, tokenizer = model_and_tokenizer

        short_task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hi")],
            max_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        prefill(model, tokenizer, make_sampler(0.0), short_tokens, cache)

        kv_prefix_cache = KVPrefixCache()
        kv_prefix_cache.add_kv_cache(tokenizer, short_prompt, cache)

        # Query with longer prompt that shares the chat template prefix
        long_task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[
                ChatCompletionMessage(role="user", content="Hi there, how are you?")
            ],
            max_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # The prompts share a prefix (chat template preamble + "Hi")
        expected_prefix = _get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, tokenizer, long_prompt
        )
        assert matched_index == 0

        # remaining_tokens should be the suffix after the shared prefix
        assert len(remaining_tokens) == len(long_tokens) - expected_prefix
        assert mx.array_equal(remaining_tokens, long_tokens[expected_prefix:])

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """Getting a cache and then mutating it (as generation does) must not corrupt stored cache."""
        model, tokenizer = model_and_tokenizer

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Mutation test")],
            max_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        prefill(model, tokenizer, make_sampler(0.0), tokens, cache)

        kv_prefix_cache = KVPrefixCache()
        kv_prefix_cache.add_kv_cache(tokenizer, prompt, cache)

        stored_length = _cache_length(kv_prefix_cache.caches[0])

        # Get cache and mutate it (simulating what generation does)
        result_cache, _, matched_index = kv_prefix_cache.get_kv_cache(
            model, tokenizer, prompt
        )
        assert matched_index == 0

        # Simulate generation: feed many additional tokens through the cache
        head_dim = result_cache[0].keys.shape[-1]
        num_heads = result_cache[0].keys.shape[1]
        extra_keys = mx.random.normal((1, num_heads, 50, head_dim))
        extra_values = mx.random.normal((1, num_heads, 50, head_dim))
        for layer_cache in result_cache:
            layer_cache.update_and_fetch(extra_keys, extra_values)
        mx.eval([c.keys for c in result_cache])

        # Stored cache must be unchanged
        assert _cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """Multiple get+mutate cycles (like repeated user requests) must not corrupt cache."""
        model, tokenizer = model_and_tokenizer

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Repeat test")],
            max_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        prefill(model, tokenizer, make_sampler(0.0), tokens, cache)

        kv_prefix_cache = KVPrefixCache()
        kv_prefix_cache.add_kv_cache(tokenizer, prompt, cache)

        stored_length = _cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _ = kv_prefix_cache.get_kv_cache(model, tokenizer, prompt)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert _cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_cache(self, model_and_tokenizer):
        """mlx_generate should save the cache after generation completes."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache()
        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            max_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # Consume the entire generator so the cache-saving code after yield runs
        generated_tokens = 0
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
        ):
            generated_tokens += 1

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        # Cache should contain prompt + generated tokens
        expected_length = len(prompt_tokens) + generated_tokens
        assert _cache_length(kv_prefix_cache.caches[0]) == expected_length

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """Second mlx_generate call with same prompt should get a prefix hit from stored cache."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache()
        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Reuse test")],
            max_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # Second call should find a prefix match (the stored cache contains
        # prompt + generated tokens, which shares the prompt prefix)
        result_cache, remaining_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, tokenizer, prompt
        )
        # The stored cache is longer than the prompt (it includes generated tokens),
        # so this is a prefix match where our prompt is fully contained
        assert matched_index == 0
        # Exact match: remaining_tokens is just the last token
        assert len(remaining_tokens) == 1
        assert mx.array_equal(remaining_tokens, prompt_tokens[-1:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """With a prompt > 1000 tokens, second generation should update the cache entry in-place."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache()

        # Build a long user message (> 1000 tokens) to exceed _MIN_PREFIX_HIT_TO_UPDATE
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content=long_content)],
            max_tokens=5,
        )
        prompt1 = apply_chat_template(tokenizer, task1)
        prompt1_tokens = encode_prompt(tokenizer, prompt1)
        assert len(prompt1_tokens) > 1000, (
            "Prompt must exceed _MIN_PREFIX_HIT_TO_UPDATE"
        )

        # First generation populates the cache (must prefill all tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task1,
            prompt=prompt1,
            kv_prefix_cache=kv_prefix_cache,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = _cache_length(kv_prefix_cache.caches[0])

        # Second generation: same long prompt + extra content (simulating multi-turn)
        task2 = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[
                ChatCompletionMessage(role="user", content=long_content),
                ChatCompletionMessage(role="assistant", content="Sure, I can help."),
                ChatCompletionMessage(role="user", content="Tell me more."),
            ],
            max_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # Verify the prompts share a long prefix
        prefix_len = _get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # Second generation should reuse the cached prefix (only prefill new tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
        ):
            pass
        second_gen_time = time.perf_counter() - t0

        # Second generation should be significantly faster due to prefix cache hit - hopefully not flaky
        assert second_gen_time < first_gen_time * 0.5, (
            f"Expected prefix cache speedup: "
            f"first={first_gen_time:.2f}s, second={second_gen_time:.2f}s"
        )

        # With prefix_hit > 1000, should update in-place (not add a second entry)
        assert len(kv_prefix_cache.prompts) == 1
        # Updated cache should be longer (prompt2 + generated > prompt1 + generated)
        updated_cache_length = _cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """After mlx_generate saves a cache, a second generation must not corrupt the stored copy."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache()
        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Immutable test")],
            max_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
        ):
            pass

        first_cache_length = _cache_length(kv_prefix_cache.caches[0])

        # Second generation gets the cache and mutates it during generation
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
        ):
            pass

        # The first stored cache must not have been mutated by the second generation
        assert _cache_length(kv_prefix_cache.caches[0]) == first_cache_length
