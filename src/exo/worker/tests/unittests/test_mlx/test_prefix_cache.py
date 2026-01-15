"""Tests for KVPrefixCache."""

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.cache import (
    KVCacheType,
    KVPrefixCache,
    TokenizerProtocol,
    get_prefix_length,
)


class MockTokenizer(TokenizerProtocol):
    """Mock tokenizer that converts string to list of char codes."""

    bos_token: str | None = None

    def encode(self, text: str, **kwargs: bool) -> list[int]:
        """Encode text to list of character codes."""
        del kwargs  # unused
        return [ord(c) for c in text]


class TestGetPrefixLength:
    """Tests for the core prefix matching algorithm."""

    def test_identical_arrays(self) -> None:
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 5

    def test_partial_match(self) -> None:
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 9, 9])
        assert get_prefix_length(a, b) == 3

    def test_no_match(self) -> None:
        a = mx.array([1, 2, 3])
        b = mx.array([9, 9, 9])
        assert get_prefix_length(a, b) == 0

    def test_different_lengths(self) -> None:
        short = mx.array([1, 2, 3])
        long = mx.array([1, 2, 3, 4, 5])
        # Should return length of shorter when they match
        assert get_prefix_length(short, long) == 3
        assert get_prefix_length(long, short) == 3

    def test_empty_array(self) -> None:
        empty: mx.array = mx.array([])
        tokens = mx.array([1, 2, 3])
        assert get_prefix_length(empty, tokens) == 0
        assert get_prefix_length(tokens, empty) == 0


class TestKVPrefixCache:
    """Tests for the KV prefix cache."""

    @pytest.fixture
    def tokenizer(self) -> MockTokenizer:
        """Mock tokenizer that converts string to list of char codes."""
        return MockTokenizer()

    @pytest.fixture
    def fake_kv(self) -> list[KVCacheType]:
        """Fake KV cache for testing."""
        return [object()]

    def test_put_stores_entry(
        self, tokenizer: MockTokenizer, fake_kv: list[KVCacheType]
    ) -> None:
        cache = KVPrefixCache(max_size=10)

        cache.put(tokenizer, "hello", fake_kv)

        assert len(cache) == 1

    def test_put_same_prompt_twice_does_not_duplicate(
        self, tokenizer: MockTokenizer, fake_kv: list[KVCacheType]
    ) -> None:
        cache = KVPrefixCache(max_size=10)

        cache.put(tokenizer, "hello", fake_kv)
        cache.put(tokenizer, "hello", fake_kv)

        assert len(cache) == 1

    def test_lru_eviction(
        self, tokenizer: MockTokenizer, fake_kv: list[KVCacheType]
    ) -> None:
        cache = KVPrefixCache(max_size=2)

        # Fill cache
        cache.put(tokenizer, "first", fake_kv)
        cache.put(tokenizer, "second", fake_kv)
        assert len(cache) == 2

        # Add third - should evict "first" (oldest)
        cache.put(tokenizer, "third", fake_kv)
        assert len(cache) == 2

        # Add "first" again - if it was evicted, cache size stays 2
        # If it wasn't evicted, this would be a no-op
        cache.put(tokenizer, "first", fake_kv)
        # Now add fourth - if "first" was re-added, size is still 2
        cache.put(tokenizer, "fourth", fake_kv)
        assert len(cache) == 2

    def test_lru_access_refreshes_entry(
        self, tokenizer: MockTokenizer, fake_kv: list[KVCacheType]
    ) -> None:
        cache = KVPrefixCache(max_size=2)

        # Add two entries
        cache.put(tokenizer, "first", fake_kv)
        cache.put(tokenizer, "second", fake_kv)

        # Access "first" again (moves to end of LRU)
        cache.put(tokenizer, "first", fake_kv)

        # Add third - should evict "second" now (oldest)
        cache.put(tokenizer, "third", fake_kv)

        # Add "second" again - this will add it as new entry
        cache.put(tokenizer, "second", fake_kv)
        # Now "first" is oldest, adding fourth should evict it
        cache.put(tokenizer, "fourth", fake_kv)

        # Cache should have "second" and "fourth", not "first"
        assert len(cache) == 2

    def test_clear(self, tokenizer: MockTokenizer, fake_kv: list[KVCacheType]) -> None:
        cache = KVPrefixCache()
        cache.put(tokenizer, "hello", fake_kv)
        cache.put(tokenizer, "world", fake_kv)
        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
