import numpy as np
import pytest

from exo.shared.types.sparse_bitnet import (
    SparseBitNetTensor,
    SparseCompressionTelemetry,
)
from exo.worker.engines.quantization.sparse_bitnet_codec import SparseBitNetCodec
from exo.worker.engines.quantization.sparse_gemm import SparseBitNetGemmEngine


def test_sparse_bitnet_tensor_properties() -> None:
    # 100 elements, 50 non-zero -> 50% sparsity
    bitmask = bytes([0b10101010] * 13)  # 104 bits > 100
    signs = bytes([0b11110000] * 7)  # 56 bits > 50
    tensor = SparseBitNetTensor(
        tensor_name="layers.0.weight",
        shape=(10, 10),
        total_elements=100,
        non_zero_count=50,
        bitmask_bytes=bitmask,
        sign_stream_bytes=signs,
        scale=0.5,
    )
    assert tensor.sparsity_ratio == pytest.approx(0.5)
    assert tensor.total_bytes == 20
    assert tensor.effective_bits_per_param == pytest.approx(1.6)

    telemetry = SparseCompressionTelemetry(
        tensor_name="layers.0.weight",
        uncompressed_fp16_bytes=200,
        compressed_bytes=tensor.total_bytes,
        effective_bits_per_param=tensor.effective_bits_per_param,
        sparsity_ratio=tensor.sparsity_ratio,
    )
    assert telemetry.compression_ratio_vs_fp16 == pytest.approx(10.0)


def test_compress_decompress_roundtrip() -> None:
    np.random.seed(42)
    # Generate random ternary weights {-1.0, 0.0, 1.0} with 50% zeros
    raw = np.random.choice([-1.0, 0.0, 1.0], size=(32, 64), p=[0.25, 0.5, 0.25])
    scale = 0.015

    sparse_tensor, telemetry = SparseBitNetCodec.compress_ternary(
        weights=raw * scale,
        tensor_name="test_proj.weight",
        scale=scale,
    )

    assert sparse_tensor.tensor_name == "test_proj.weight"
    assert sparse_tensor.shape == (32, 64)
    assert sparse_tensor.total_elements == 2048
    assert sparse_tensor.scale == scale
    assert telemetry.compressed_bytes == sparse_tensor.total_bytes
    # At ~50% sparsity, effective bits per param is around 1.0 to 1.6 bits
    assert 0.8 <= sparse_tensor.effective_bits_per_param <= 1.8

    # Lossless decompression check
    decompressed = SparseBitNetCodec.decompress_to_ternary(sparse_tensor)
    assert decompressed.shape == (32, 64)
    np.testing.assert_allclose(decompressed, raw * scale, rtol=1e-5, atol=1e-5)


def test_edge_cases_all_zeros_and_all_ones() -> None:
    # 1. All zeros
    zeros = np.zeros((16, 16), dtype=np.float32)
    sparse_zero, _ = SparseBitNetCodec.compress_ternary(zeros, tensor_name="zero_weight")
    assert sparse_zero.non_zero_count == 0
    assert sparse_zero.sparsity_ratio == 1.0
    decompressed_zero = SparseBitNetCodec.decompress_to_ternary(sparse_zero)
    np.testing.assert_array_equal(decompressed_zero, zeros)

    # 2. All +1
    ones = np.ones((8, 8), dtype=np.float32)
    sparse_one, _ = SparseBitNetCodec.compress_ternary(ones, tensor_name="one_weight")
    assert sparse_one.non_zero_count == 64
    assert sparse_one.sparsity_ratio == 0.0
    decompressed_one = SparseBitNetCodec.decompress_to_ternary(sparse_one)
    np.testing.assert_array_equal(decompressed_one, ones)

    # 3. All -1
    neg_ones = -np.ones((8, 8), dtype=np.float32)
    sparse_neg, _ = SparseBitNetCodec.compress_ternary(neg_ones, tensor_name="neg_weight")
    assert sparse_neg.non_zero_count == 64
    decompressed_neg = SparseBitNetCodec.decompress_to_ternary(sparse_neg)
    np.testing.assert_array_equal(decompressed_neg, neg_ones)

    # 4. Odd dimensions (prime elements)
    odd = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    sparse_odd, _ = SparseBitNetCodec.compress_ternary(odd, tensor_name="odd_weight")
    decompressed_odd = SparseBitNetCodec.decompress_to_ternary(sparse_odd)
    np.testing.assert_array_equal(decompressed_odd, odd)


def test_sparse_gemm_numerical_equivalence() -> None:
    np.random.seed(123)
    k_dim, n_dim = 128, 64
    m_dim = 4

    # Ternary weight
    raw_w = np.random.choice([-1.0, 0.0, 1.0], size=(k_dim, n_dim), p=[0.25, 0.5, 0.25])
    scale = 0.05
    sparse_w, _ = SparseBitNetCodec.compress_ternary(raw_w * scale, scale=scale)

    activations = np.random.randn(m_dim, k_dim).astype(np.float32)
    expected_output = np.matmul(activations, raw_w * scale)

    output, elapsed_ms, flop_reduction = SparseBitNetGemmEngine.execute_sparse_gemm(
        activations=activations,
        sparse_weight=sparse_w,
    )

    assert output.shape == (m_dim, n_dim)
    assert elapsed_ms >= 0.0
    assert flop_reduction == pytest.approx(sparse_w.sparsity_ratio)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    # Test 1D activation vector
    act_1d = activations[0]
    out_1d, _, _ = SparseBitNetGemmEngine.execute_sparse_gemm(act_1d, sparse_w)
    assert out_1d.shape == (1, n_dim)
    np.testing.assert_allclose(out_1d[0], expected_output[0], rtol=1e-5, atol=1e-5)
