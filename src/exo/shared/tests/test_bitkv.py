import numpy as np

from exo.shared.types.bitkv import (
    BitKVConfig,
    QuantizedKVTensors,
)
from exo.worker.engines.kv_cache.bitkv_manager import BitKVCacheManager, BitKVQuantizer


def test_2bit_packing_unpacking() -> None:
    # Test array of 2-bit values with length not a multiple of 4
    original = np.array([0, 1, 2, 3, 3, 2, 1, 0, 2], dtype=np.uint8)
    packed = BitKVQuantizer.pack_2bit(original)
    unpacked = BitKVQuantizer.unpack_2bit(packed, len(original))
    np.testing.assert_array_equal(unpacked, original)

    # Empty array edge case
    empty_packed = BitKVQuantizer.pack_2bit(np.empty(0, dtype=np.uint8))
    assert empty_packed == b""
    assert len(BitKVQuantizer.unpack_2bit(empty_packed, 0)) == 0


def test_key_per_channel_quantization_and_dequantization() -> None:
    np.random.seed(42)
    seq_len, num_heads, head_dim = 64, 4, 32
    # Generate Key tensor with channel-specific bias (realistic LLM Key distribution)
    channel_bias = np.random.randn(1, num_heads, head_dim).astype(np.float32) * 5.0
    key_tensor = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32) + channel_bias

    packed, scales, zeros = BitKVQuantizer.quantize_key_per_channel(key_tensor)
    assert len(packed) == (seq_len * num_heads * head_dim + 3) // 4

    reconstructed = BitKVQuantizer.dequantize_key(
        packed, scales, zeros, (seq_len, num_heads, head_dim)
    )
    assert reconstructed.shape == key_tensor.shape

    # Cosine similarity check across all elements
    dot_prod = np.sum(key_tensor * reconstructed)
    norm = np.linalg.norm(key_tensor) * np.linalg.norm(reconstructed)
    cos_sim = dot_prod / norm
    assert cos_sim > 0.95


def test_value_per_token_quantization_and_dequantization() -> None:
    np.random.seed(42)
    seq_len, num_heads, head_dim = 64, 4, 32
    # Generate Value tensor with token-specific bias (realistic LLM Value distribution)
    token_bias = np.random.randn(seq_len, num_heads, 1).astype(np.float32) * 3.0
    value_tensor = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32) + token_bias

    packed, scales, zeros = BitKVQuantizer.quantize_value_per_token(value_tensor)
    assert len(packed) == (seq_len * num_heads * head_dim + 3) // 4

    reconstructed = BitKVQuantizer.dequantize_value(
        packed, scales, zeros, (seq_len, num_heads, head_dim)
    )
    assert reconstructed.shape == value_tensor.shape

    dot_prod = np.sum(value_tensor * reconstructed)
    norm = np.linalg.norm(value_tensor) * np.linalg.norm(reconstructed)
    cos_sim = dot_prod / norm
    assert cos_sim > 0.92


def test_bitkv_cache_manager_sliding_window_append() -> None:
    np.random.seed(123)
    window_size = 16
    num_heads = 2
    head_dim = 16
    config = BitKVConfig(window_size=window_size)
    manager = BitKVCacheManager(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )

    total_tokens = 48
    all_keys = []
    all_values = []
    for _ in range(total_tokens):
        k = np.random.randn(num_heads, head_dim).astype(np.float32)
        v = np.random.randn(num_heads, head_dim).astype(np.float32)
        all_keys.append(k)
        all_values.append(v)
        manager.append_token_kv(k, v)

    assert manager.total_tokens == total_tokens
    assert manager._quantized_seq_len == total_tokens - window_size
    assert len(manager._recent_keys) == window_size

    full_k, full_v = manager.get_full_kv()
    assert full_k.shape == (total_tokens, num_heads, head_dim)
    assert full_v.shape == (total_tokens, num_heads, head_dim)

    # Verify that recent window tokens are 100% exact in full precision
    expected_recent_k = np.stack(all_keys[-window_size:], axis=0)
    expected_recent_v = np.stack(all_values[-window_size:], axis=0)
    np.testing.assert_array_equal(full_k[-window_size:], expected_recent_k)
    np.testing.assert_array_equal(full_v[-window_size:], expected_recent_v)


def test_bitkv_telemetry_and_compression_ratio() -> None:
    np.random.seed(999)
    window_size = 32
    num_heads = 4
    head_dim = 64
    config = BitKVConfig(window_size=window_size)
    manager = BitKVCacheManager(
        layer_idx=1,
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )

    # Simulate 256 tokens appended
    for _ in range(256):
        k = np.random.randn(num_heads, head_dim).astype(np.float32)
        v = np.random.randn(num_heads, head_dim).astype(np.float32)
        manager.append_token_kv(k, v)

    telemetry = manager.get_telemetry()
    assert telemetry.total_tokens == 256
    # 256 tokens * 4 heads * 64 dim * 2 (K & V) * 2 bytes (FP16) = 262,144 bytes
    assert telemetry.uncompressed_fp16_bytes == 262144
    assert telemetry.compression_ratio > 3.5
    assert telemetry.effective_bits_per_param < 5.0

    snapshot = manager.snapshot_tensors()
    assert isinstance(snapshot, QuantizedKVTensors)
    assert snapshot.total_seq_len == 256
    assert snapshot.quantized_seq_len == 224
    assert snapshot.recent_seq_len == 32
