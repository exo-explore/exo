import numpy as np
import pytest

from exo.shared.types.bnn import BNNPackedTensor
from exo.worker.engines.quantization.bnn_kernel import BNNCodec, BNNGemmEngine


def test_bnn_binarize_and_pack_properties() -> None:
    # 64 elements -> 64 bits = 8 bytes (1 uint64)
    raw = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(64)], dtype=np.float32)
    packed = BNNCodec.binarize_and_pack(raw, tensor_name="layer1.weight", scale=0.5)

    assert isinstance(packed, BNNPackedTensor)
    assert packed.total_elements == 64
    assert packed.total_bytes == 8
    assert packed.effective_bits_per_param == 1.0
    # 64 * 2 (FP16) = 128 bytes vs 8 bytes packed -> 16.0x
    assert packed.compression_ratio_vs_fp16 == pytest.approx(16.0)


def test_bnn_pack_unpack_lossless() -> None:
    np.random.seed(42)
    shape = (16, 64)
    # Generate binary weights {-1.0, +1.0}
    raw = np.random.choice([-1.0, 1.0], size=shape).astype(np.float32)
    scale = 0.025

    packed = BNNCodec.binarize_and_pack(raw * scale, tensor_name="test.weight", scale=scale)
    assert packed.shape == shape
    assert packed.total_elements == 1024

    unpacked = BNNCodec.unpack_to_floats(packed)
    assert unpacked.shape == shape
    np.testing.assert_allclose(unpacked, raw * scale, rtol=1e-5, atol=1e-5)


def test_bnn_xnor_mathematical_equivalence() -> None:
    np.random.seed(123)
    m_dim, k_dim, n_dim = 4, 128, 64
    activations = np.random.randn(m_dim, k_dim).astype(np.float32)

    # Pure binary weights {-1.0, +1.0}
    raw_w = np.random.choice([-1.0, 1.0], size=(k_dim, n_dim)).astype(np.float32)
    scale = 0.05
    packed_w = BNNCodec.binarize_and_pack(raw_w * scale, scale=scale)

    # 1. BNN GEMM execution
    output, telemetry = BNNGemmEngine.execute_bnn_gemm(
        activations=activations,
        packed_weight=packed_w,
        binarize_activations=True,
    )

    # 2. Ground truth mathematical simulation
    act_scale = np.mean(np.abs(activations), axis=-1, keepdims=True)
    act_sign = np.where(activations >= 0.0, 1.0, -1.0).astype(np.float32) * act_scale
    expected = np.matmul(act_sign, raw_w * scale)

    assert output.shape == (m_dim, n_dim)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert telemetry.compression_ratio_vs_fp16 == pytest.approx(16.0)
    assert telemetry.estimated_energy_reduction_ratio == 0.95
    assert telemetry.popcount_operations_count > 0


def test_bnn_edge_cases_odd_dimensions_and_1d() -> None:
    # Odd dimension: 7x13 = 91 elements (not a multiple of 64)
    odd_w = np.random.choice([-1.0, 1.0], size=(7, 13)).astype(np.float32)
    packed_odd = BNNCodec.binarize_and_pack(odd_w)
    unpacked_odd = BNNCodec.unpack_to_floats(packed_odd)
    np.testing.assert_allclose(unpacked_odd, odd_w, rtol=1e-5, atol=1e-5)

    # 1D activation vector
    act_1d = np.random.randn(7).astype(np.float32)
    out_1d, _ = BNNGemmEngine.execute_bnn_gemm(act_1d, packed_odd)
    assert out_1d.shape == (13,)

    # Shape mismatch error check
    invalid_act = np.random.randn(4, 9).astype(np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        BNNGemmEngine.execute_bnn_gemm(invalid_act, packed_odd)
