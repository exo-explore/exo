# type: ignore
import time
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    encode_prompt,
    get_prefix_length,
    make_kv_cache,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate, prefill
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
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
        assert get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0


class TestKVPrefix:
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a minimal mock tokenizer for tests that don't need real tokenization."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    def test_starts_empty(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.add_kv_cache(mx.array([1, 2, 3]), [KVCache()])
        assert len(cache.prompts) == 1
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.clear()
        assert len(cache.prompts) == 0


def _fake_kv_cache(num_layers: int, num_tokens: int) -> list[KVCache]:
    """Build a KV cache of `num_layers` KVCache entries each holding a small
    but valid K/V tensor covering `num_tokens` tokens along the sequence axis.
    Shape [B=1, H=2, S=num_tokens, D=4].
    """
    caches: list[KVCache] = []
    for layer_idx in range(num_layers):
        c = KVCache()
        k = mx.arange(num_tokens, dtype=mx.float32)
        k = mx.broadcast_to(k.reshape(1, 1, num_tokens, 1), (1, 2, num_tokens, 4))
        # Layer-dependent offset so different layers aren't identical.
        k = k + float(layer_idx)
        v = k + 0.5
        c.keys = mx.array(k)
        c.values = mx.array(v)
        c.offset = num_tokens
        caches.append(c)
    return caches


class TestRadixTrieStorage:
    """Verify that storage is actually deduplicated across sessions sharing a prefix."""

    def test_two_sessions_share_prefix_node(self):
        cache = KVPrefixCache(None)
        # Two prompts sharing [1,2,3,4,5] then diverging.
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        tokens_b = mx.array(shared + [20, 21], dtype=mx.int32)

        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))
        id_b = cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=7))

        # Both leaves reachable under the shared node at depth 5.
        root = cache._root  # pyright: ignore[reportPrivateUsage]
        first = root.children.get(1)
        assert first is not None
        # The first child's edge should cover exactly the shared prefix, then
        # split into two children (10 and 20).
        assert first.depth == 5
        assert first.edge_length == 5
        assert set(first.children.keys()) == {10, 20}
        # Ref count == 2 leaves under the shared node.
        assert first.ref_count == 2
        assert id_a != id_b

    def test_prefix_hit_reuses_stored_kv(self):
        cache = KVPrefixCache(None)
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))

        # Query with a longer prompt sharing the prefix.
        tokens_b = mx.array(shared + [20, 21, 22], dtype=mx.int32)
        # Model mock: _materialize_full_leaf_cache doesn't need the model, and
        # get_kv_cache constructs a fresh cache for remaining tokens via
        # make_kv_cache(model). Provide a stub with .layers that routes to
        # make_kv_cache's fallback branch.
        #
        # In practice, miss case is what matters here: verify the hit branch
        # returns the expected match depth and leaf id.
        from unittest.mock import MagicMock

        model = MagicMock()
        model.layers = [None, None]

        result_cache, remaining, matched_id, is_exact = cache.get_kv_cache(
            model, tokens_b
        )
        assert matched_id == 0
        assert int(remaining.shape[0]) == 3  # [20, 21, 22]
        assert is_exact is False
        # Materialized cache should have 5 tokens along the sequence axis.
        assert result_cache[0].offset == 5

    def test_eviction_of_one_leaf_frees_only_unique_branch(self):
        cache = KVPrefixCache(None, max_sessions=2)
        shared = [1, 2, 3, 4, 5]
        tokens_a = mx.array(shared + [10, 11], dtype=mx.int32)
        tokens_b = mx.array(shared + [20, 21], dtype=mx.int32)
        tokens_c = mx.array(shared + [30, 31], dtype=mx.int32)

        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=7))
        id_b = cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=7))
        # Adding C forces eviction of A (the LRU non-pinned).
        id_c = cache.add_kv_cache(tokens_c, _fake_kv_cache(num_layers=2, num_tokens=7))

        assert id_a not in cache.prompts
        assert id_b in cache.prompts
        assert id_c in cache.prompts

        # Shared prefix node must still exist (referenced by B and C).
        root = cache._root  # pyright: ignore[reportPrivateUsage]
        first = root.children.get(1)
        assert first is not None
        assert first.ref_count == 2
        # A's unique branch (first token 10) is gone; B and C's branches remain.
        assert set(first.children.keys()) == {20, 30}

    def test_pin_prevents_eviction(self):
        cache = KVPrefixCache(None, max_sessions=1)
        tokens_a = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        tokens_b = mx.array([6, 7, 8, 9, 10], dtype=mx.int32)
        id_a = cache.add_kv_cache(tokens_a, _fake_kv_cache(num_layers=2, num_tokens=5))
        cache.pin(id_a)
        cache.add_kv_cache(tokens_b, _fake_kv_cache(num_layers=2, num_tokens=5))
        # Pinned A survives; the cap is "soft" under pinning.
        assert id_a in cache.prompts

    def test_update_extends_existing_leaf(self):
        cache = KVPrefixCache(None)
        tokens_short = mx.array([1, 2, 3], dtype=mx.int32)
        id_s = cache.add_kv_cache(
            tokens_short, _fake_kv_cache(num_layers=2, num_tokens=3)
        )
        tokens_long = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        cache.update_kv_cache(
            leaf_id=id_s,
            prompt_tokens=tokens_long,
            cache=_fake_kv_cache(num_layers=2, num_tokens=5),
            snapshots=None,
            restore_pos=3,
        )
        assert list(cache.prompts.keys()) == [id_s]
        stored = cache.prompts[id_s]
        assert int(stored.shape[0]) == 5

    def test_media_region_mismatch_truncates_match(self):
        from exo.worker.engines.mlx.vision import MediaRegion

        cache = KVPrefixCache(None)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7], dtype=mx.int32)
        regions_a = [MediaRegion(content_hash="img-A", start_pos=2, end_pos=5)]
        cache.add_kv_cache(
            tokens, _fake_kv_cache(num_layers=1, num_tokens=7), media_regions=regions_a
        )

        from unittest.mock import MagicMock

        model = MagicMock()
        model.layers = [None]

        # Query with the same tokens but a different image hash at pos 2.
        regions_b = [MediaRegion(content_hash="img-B", start_pos=2, end_pos=5)]
        _, remaining, _, _ = cache.get_kv_cache(model, tokens, media_regions=regions_b)
        # Match should be truncated to pos 2 (start of the mismatching region).
        assert int(remaining.shape[0]) == 5  # tokens 2..6


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

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello!!")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        # Cache should now hold the prompt tokens minus one
        assert cache_length(cache) == len(tokens) - 1
        # Snapshots should be available for models with non-KV caches
        assert len(snapshots) > 0

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Test exact")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # Retrieve with same prompt: exact match
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, tokens
        )
        assert matched_index == 0

        # Exact match returns last token(s) — for models with SSM/rotating caches,
        # snapshot availability constrains how far back we can trim, so remaining
        # may be 1 or 2 tokens depending on the model.
        assert len(remaining_tokens) >= 1
        assert mx.array_equal(remaining_tokens, tokens[-len(remaining_tokens) :])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """get_kv_cache with a longer prompt sharing prefix should return partial match."""
        model, tokenizer = model_and_tokenizer

        short_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi")],
            max_output_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            short_tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(short_tokens, cache, snapshots)

        # Query with longer prompt that shares the chat template prefix
        long_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi there, how are you?")],
            max_output_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # The prompts share a prefix (chat template preamble + "Hi")
        expected_prefix = get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, long_tokens
        )
        assert matched_index == 0

        # remaining_tokens covers from snapshot restore position to end
        assert len(remaining_tokens) >= len(long_tokens) - expected_prefix

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """Getting a cache and then mutating it (as generation does) must not corrupt stored cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Mutation test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        # Get cache and mutate it (simulating what generation does)
        result_cache, _, matched_index, _ = kv_prefix_cache.get_kv_cache(model, tokens)
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
        assert cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """Multiple get+mutate cycles (like repeated user requests) must not corrupt cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Repeat test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _, _ = kv_prefix_cache.get_kv_cache(model, tokens)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_cache(self, model_and_tokenizer):
        """mlx_generate should save the cache after generation completes."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=5,
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
            group=None,
        ):
            generated_tokens += 1

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        # Cache should contain prompt + generated tokens
        expected_length = len(prompt_tokens) + generated_tokens
        assert cache_length(kv_prefix_cache.caches[0]) == expected_length

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """Second mlx_generate call with same prompt should get a prefix hit from stored cache."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Reuse test")],
            max_output_tokens=5,
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
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # Second call should find a prefix match (the stored cache contains
        # prompt + generated tokens, which shares the prompt prefix)
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        # The stored cache is longer than the prompt (it includes generated tokens),
        # so this is a prefix match where our prompt is fully contained
        assert matched_index == 0
        # Exact match: remaining_tokens is just the last token and the one before
        assert len(remaining_tokens) == 2
        assert mx.array_equal(remaining_tokens, prompt_tokens[-2:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """With a prompt > 1000 tokens, second generation should update the cache entry in-place."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Build a long user message (> 1000 tokens) to exceed _MIN_PREFIX_HIT_TO_UPDATE
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content=long_content)],
            max_output_tokens=5,
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
            group=None,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation: same long prompt + extra content (simulating multi-turn)
        task2 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[
                InputMessage(role="user", content=long_content),
                InputMessage(role="assistant", content="Sure, I can help."),
                InputMessage(role="user", content="Tell me more."),
            ],
            max_output_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # Verify the prompts share a long prefix
        prefix_len = get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # Second generation should reuse the cached prefix (only prefill new tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
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
        updated_cache_length = cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """After mlx_generate saves a cache, a second generation must not corrupt the stored copy."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Immutable test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        firstcache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation gets the cache and mutates it during generation
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        # The first stored cache must not have been mutated by the second generation
        assert cache_length(kv_prefix_cache.caches[0]) == firstcache_length

    def test_evicts_lru_entry_under_memory_pressure(self, model_and_tokenizer):
        """Under memory pressure, adding a new cache entry evicts the least recently used one."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Add three cache entries with different prompts
        prompts = ["First entry", "Second entry", "Third entry"]
        for i, content in enumerate(prompts):
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content=content)],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)
            # Stagger _last_used so LRU order is deterministic
            kv_prefix_cache._last_used[i] = float(i)

        assert len(kv_prefix_cache.prompts) == 3

        # Access the third entry to make it most recently used
        kv_prefix_cache._last_used[2] = 100.0
        # Entry 0 (_last_used=0.0) is LRU, entry 1 (_last_used=1.0) is next

        # Simulate memory pressure: return usage above _MEMORY_THRESHOLD (0.9)
        with patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.95,
        ):
            # Trigger eviction by adding a new entry
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content="New entry")],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)

        # LRU entries should have been evicted (entries 0, 1, 2 in order of _last_used)
        # Since fake_active stays above threshold after each eviction (we don't change it),
        # all old entries get evicted, leaving only the newly added one
        assert len(kv_prefix_cache.prompts) == 1
        # The surviving entry should be the newly added one
        assert get_prefix_length(kv_prefix_cache.prompts[0], tokens) == len(tokens)
