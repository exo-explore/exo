from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import exo.worker.engines.mlx.cache as cache_module
from exo.worker.engines.mlx.cache import KVCacheType, KVPrefixCache
from exo.worker.engines.mlx.tiered_cache import (
    TieredCacheConfiguration,
    TieredKVCacheManager,
    deserialize_single_cache_layer,
    serialize_single_cache_layer,
)


class MockAttentionLayerCache:
    """Mock key-value cache layer simulating an attention layer."""

    def __init__(
        self,
        offset: int = 16,
        keys: np.ndarray | None = None,
        values: np.ndarray | None = None,
        nbytes: int = 1024,
    ) -> None:
        self.offset = offset
        self.keys = (
            keys
            if keys is not None
            else np.random.randn(2, 4, 16, 64).astype(np.float32)
        )
        self.values = (
            values
            if values is not None
            else np.random.randn(2, 4, 16, 64).astype(np.float32)
        )
        self.nbytes = nbytes

    def trim(self, num_tokens: int) -> None:
        """Trim tokens from this cache layer."""
        self.offset = max(0, self.offset - num_tokens)


@pytest.fixture(autouse=True)
def no_memory_pressure() -> Iterator[None]:
    """Default fixture ensuring that host system RAM load does not artificially trigger eviction."""
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.1):
        yield


def test_serialization_and_deserialization_fidelity() -> None:
    """Verify that serializing and deserializing a cache layer preserves exact values and offset."""
    original_keys = np.arange(32, dtype=np.float32).reshape(2, 2, 8)
    original_values = np.arange(32, 64, dtype=np.float32).reshape(2, 2, 8)
    original_layer = MockAttentionLayerCache(
        offset=42,
        keys=original_keys,
        values=original_values,
        nbytes=256,
    )

    serialized_data = serialize_single_cache_layer(original_layer)
    assert serialized_data["cache_type"] == "GenericCache"
    assert serialized_data["attributes"]["offset"] == 42

    restored_layer = deserialize_single_cache_layer(serialized_data)
    assert restored_layer.offset == 42
    np.testing.assert_array_equal(restored_layer.keys, original_keys)
    np.testing.assert_array_equal(restored_layer.values, original_values)


def test_tiered_cache_manager_offload_and_restore() -> None:
    """Verify that TieredKVCacheManager writes compressed files to disk and restores them."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        configuration = TieredCacheConfiguration(
            storage_directory_path=temporary_directory,
            compression_enabled=True,
            compression_level=1,
            cleanup_on_exit=True,
        )
        manager = TieredKVCacheManager(configuration)

        layer_1 = MockAttentionLayerCache(offset=10)
        layer_2 = MockAttentionLayerCache(offset=10)
        cache_layers: list[MockAttentionLayerCache] = [layer_1, layer_2]

        metadata = manager.offload_cache_entry(
            entry_identifier="test_entry_001",
            token_count=10,
            cache_layers=cache_layers,
            snapshots=None,
            access_counter=1,
            prefill_tokens_per_second=120.5,
        )

        assert metadata.token_count == 10
        assert metadata.entry_identifier == "test_entry_001"
        assert Path(metadata.file_path).exists()
        assert metadata.byte_size > 0

        restored_layers, snapshots = manager.restore_cache_entry(metadata)
        assert len(restored_layers) == 2
        assert snapshots is None
        assert restored_layers[0].offset == 10
        np.testing.assert_array_almost_equal(restored_layers[0].keys, layer_1.keys)

        manager.remove_offloaded_entry(metadata)
        assert not Path(metadata.file_path).exists()


def test_memory_pressure_triggers_offload_rather_than_destruction() -> None:
    """Verify that memory pressure moves the least recently used cache to cold storage without deleting it."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        configuration = TieredCacheConfiguration(
            storage_directory_path=temporary_directory,
            compression_enabled=False,
        )
        manager = TieredKVCacheManager(configuration)
        prefix_cache = KVPrefixCache(
            group=None,
            tiered_cache_manager=manager,
        )

        prompt_tokens_1 = np.array([101, 102, 103], dtype=np.int64)
        prompt_tokens_2 = np.array([201, 202, 203], dtype=np.int64)

        cache_1: KVCacheType = [MockAttentionLayerCache(offset=3, nbytes=500)]  # type: ignore[assignment]
        cache_2: KVCacheType = [MockAttentionLayerCache(offset=3, nbytes=500)]  # type: ignore[assignment]

        prefix_cache.add_kv_cache(prompt_tokens_1, cache_1)
        prefix_cache.add_kv_cache(prompt_tokens_2, cache_2)

        assert len(prefix_cache.caches) == 2
        assert not prefix_cache._is_offloaded[0]
        assert not prefix_cache._is_offloaded[1]

        # Simulate memory pressure exceeding threshold
        with patch.object(
            cache_module, "get_memory_used_percentage", side_effect=[0.95, 0.40]
        ):
            prefix_cache._evict_if_needed(incoming_bytes=100)

        # Entry 0 (least recently used) should be offloaded to Tier 2 cold storage
        assert prefix_cache._is_offloaded[0] is True
        assert prefix_cache.caches[0] is None
        assert prefix_cache._offloaded_metadata[0] is not None
        assert Path(prefix_cache._offloaded_metadata[0].file_path).exists()

        # Prompt 0 is still retained in catalog
        assert len(prefix_cache.prompts) == 2
        np.testing.assert_array_equal(prefix_cache.prompts[0], prompt_tokens_1)


def test_transparent_hydration_on_prefix_match() -> None:
    """Verify that querying an offloaded cache transparently restores it from disk."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        configuration = TieredCacheConfiguration(
            storage_directory_path=temporary_directory,
            compression_enabled=False,
        )
        manager = TieredKVCacheManager(configuration)
        prefix_cache = KVPrefixCache(
            group=None,
            tiered_cache_manager=manager,
        )

        prompt_tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        original_layer = MockAttentionLayerCache(offset=5, nbytes=400)
        prefix_cache.add_kv_cache(prompt_tokens, [original_layer])  # type: ignore[assignment]

        # Explicitly offload entry 0
        success = prefix_cache._offload_entry(0)
        assert success is True
        assert prefix_cache._is_offloaded[0] is True
        assert prefix_cache.caches[0] is None

        # Query with prompt that matches prefix
        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]
        retrieved_cache, _remaining_tokens, matched_index, is_exact = (
            prefix_cache.get_kv_cache(
                model=mock_model,
                prompt_tokens=prompt_tokens,
            )
        )

        assert matched_index == 0
        assert is_exact is True
        assert prefix_cache._is_offloaded[0] is False
        assert prefix_cache.caches[0] is not None
        assert retrieved_cache[0].offset == 4  # Trimmed by 1 for stream_generate next token
        assert prefix_cache.caches[0][0].offset == 5  # Hydrated cache in prefix store maintains full 5 tokens
        np.testing.assert_array_almost_equal(
            retrieved_cache[0].keys, original_layer.keys
        )


def test_total_entries_stay_bounded_by_max_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that when entries exceed _MAX_ENTRIES, the overall oldest entry is permanently evicted."""
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    with tempfile.TemporaryDirectory() as temporary_directory:
        configuration = TieredCacheConfiguration(
            storage_directory_path=temporary_directory,
        )
        manager = TieredKVCacheManager(configuration)
        prefix_cache = KVPrefixCache(
            group=None,
            tiered_cache_manager=manager,
        )

        for token_seed in (10, 20, 30):
            tokens = np.array([token_seed, token_seed + 1], dtype=np.int64)
            prefix_cache.add_kv_cache(
                tokens, [MockAttentionLayerCache(offset=2, nbytes=100)]  # type: ignore[assignment]
            )

        assert len(prefix_cache.prompts) == 2
        # First entry (10) was permanently evicted to respect _MAX_ENTRIES limit
        assert int(prefix_cache.prompts[0][0]) == 20
        assert int(prefix_cache.prompts[1][0]) == 30


def test_clear_removes_all_tracked_and_disk_files() -> None:
    """Verify that clear() empties all collections and removes all disk files."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        configuration = TieredCacheConfiguration(
            storage_directory_path=temporary_directory,
        )
        manager = TieredKVCacheManager(configuration)
        prefix_cache = KVPrefixCache(
            group=None,
            tiered_cache_manager=manager,
        )

        tokens = np.array([1, 2, 3], dtype=np.int64)
        prefix_cache.add_kv_cache(tokens, [MockAttentionLayerCache(offset=3)])  # type: ignore[assignment]
        prefix_cache._offload_entry(0)

        assert prefix_cache._is_offloaded[0] is True
        assert len(list(Path(temporary_directory).glob("*.kvcache"))) == 1

        prefix_cache.clear()
        assert len(prefix_cache.prompts) == 0
        assert len(prefix_cache.caches) == 0
        assert len(prefix_cache._offloaded_metadata) == 0
        assert len(list(Path(temporary_directory).glob("*.kvcache"))) == 0
