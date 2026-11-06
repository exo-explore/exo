import numpy as np

from exo.shared.types.ane_compute import (
    ANEShapeBucket,
    ANETelemetry,
)
from exo.worker.engines.npu import (
    ANEBucketManager,
    AppleNeuralEngineExecutor,
    IOSurfaceMemoryAllocator,
)


def test_ane_shape_bucket_properties() -> None:
    bucket = ANEShapeBucket(
        bucket_id="b-test",
        batch_size=2,
        chunk_size=128,
        hidden_dim=2048,
    )
    assert bucket.max_tokens == 256


def test_ane_bucket_manager_selection() -> None:
    manager = ANEBucketManager(chunk_sizes=(64, 128, 256, 512, 1024), hidden_dim=4096)

    # 0 or negative tokens -> smallest bucket
    assert manager.select_optimal_bucket(0).chunk_size == 64
    assert manager.select_optimal_bucket(50).chunk_size == 64
    assert manager.select_optimal_bucket(64).chunk_size == 64

    # Mid ranges
    assert manager.select_optimal_bucket(65).chunk_size == 128
    assert manager.select_optimal_bucket(128).chunk_size == 128
    assert manager.select_optimal_bucket(300).chunk_size == 512

    # Overflow -> capped at largest bucket
    assert manager.select_optimal_bucket(2000).chunk_size == 1024


def test_ane_bucket_padding_and_unpadding() -> None:
    manager = ANEBucketManager(chunk_sizes=(64, 128), hidden_dim=8)
    bucket = manager.select_optimal_bucket(10)

    # Generate 10 tokens
    raw_data = np.arange(10 * 8, dtype=np.float32).reshape(10, 8)
    padded, orig_len = manager.pad_activations(raw_data, bucket)

    assert orig_len == 10
    assert padded.shape == (64, 8)
    # Padded tail should be zeros
    assert np.all(padded[10:] == 0.0)

    # Unpadding should restore the original 10 rows exactly
    restored = manager.unpad_activations(padded, orig_len)
    assert restored.shape == (10, 8)
    np.testing.assert_array_equal(restored, raw_data)


def test_iosurface_allocator_stride_and_roundtrip() -> None:
    # Width 33 float32 elements = 132 bytes.
    # Aligned to 64 bytes -> 192 bytes_per_row.
    desc, buf = IOSurfaceMemoryAllocator.allocate_surface(width=33, height=4, bytes_per_element=4, alignment=64)

    assert desc.bytes_per_row % 64 == 0
    assert desc.bytes_per_row == 192
    assert desc.allocation_size_bytes == 192 * 4
    assert len(buf) == desc.allocation_size_bytes

    # Roundtrip test
    test_tensor = np.random.randn(4, 33).astype(np.float32)
    IOSurfaceMemoryAllocator.write_tensor_to_surface(desc, buf, test_tensor)

    read_back = IOSurfaceMemoryAllocator.read_tensor_from_surface(desc, buf, shape=(4, 33))
    np.testing.assert_allclose(read_back, test_tensor, rtol=1e-5, atol=1e-5)


def test_ane_executor_end_to_end() -> None:
    np.random.seed(42)
    hidden_dim = 16
    out_dim = 8
    seq_len = 25

    activations = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    weights = np.random.randn(hidden_dim, out_dim).astype(np.float32)

    executor = AppleNeuralEngineExecutor(hidden_dim=hidden_dim)
    output, telemetry = executor.execute_layer(activations, weights)

    # Output dimensions should match unpadded sequence length
    assert output.shape == (seq_len, out_dim)

    # Output values should match direct reference dot product
    expected = np.matmul(activations, weights)
    np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)

    # Telemetry checks
    assert isinstance(telemetry, ANETelemetry)
    assert telemetry.input_tokens == seq_len
    assert telemetry.padded_tokens == 64  # Smallest bucket >= 25
    assert telemetry.dma_transfer_time_saved_ms > 0
    assert telemetry.estimated_power_watts == 2.4
    assert telemetry.tokens_per_second > 0
    assert 0.0 < telemetry.padding_overhead_ratio < 1.0
