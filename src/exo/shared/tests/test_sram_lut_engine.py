import numpy as np
import pytest

from exo.shared.types.sram_cache import SRAMConfig
from exo.worker.engines.npu.sram_lut_engine import (
    NativeINT2LUTEngine,
    OnChipSRAMCacheManager,
)


def test_sram_cache_residency_and_spillover() -> None:
    max_tokens = 16
    config = SRAMConfig(max_sram_tokens=max_tokens)
    manager = OnChipSRAMCacheManager(layer_idx=0, config=config)

    # 1. Append 10 tokens (within SRAM capacity)
    for _ in range(10):
        k = np.ones((4, 32), dtype=np.float32)
        v = np.ones((4, 32), dtype=np.float32)
        manager.append_kv(k, v)

    alloc_10 = manager.get_allocation()
    assert alloc_10.tokens_in_sram == 10
    assert alloc_10.tokens_in_dram == 0
    assert alloc_10.is_sram_saturated is False

    telemetry_10 = manager.get_telemetry()
    assert telemetry_10.sram_hit_ratio == pytest.approx(1.0)
    assert telemetry_10.dram_io_reduction_ratio == pytest.approx(1.0)
    assert telemetry_10.estimated_energy_savings_ratio == pytest.approx(0.90)

    # 2. Append 14 more tokens (total 24 tokens > max_tokens 16)
    for _ in range(14):
        k = np.ones((4, 32), dtype=np.float32) * 2.0
        v = np.ones((4, 32), dtype=np.float32) * 2.0
        manager.append_kv(k, v)

    alloc_24 = manager.get_allocation()
    assert alloc_24.tokens_in_sram == max_tokens
    assert alloc_24.tokens_in_dram == 8
    assert alloc_24.is_sram_saturated is True

    full_k, full_v = manager.get_full_kv()
    assert full_k.shape == (24, 4, 32)
    assert full_v.shape == (24, 4, 32)


def test_lut_ternary_gemm_numerical_equivalence() -> None:
    np.random.seed(42)
    m_dim, k_dim, n_dim = 4, 64, 32
    activations = np.random.randn(m_dim, k_dim).astype(np.float32)
    ternary_weights = np.random.choice([-1, 0, 1], size=(k_dim, n_dim), p=[0.25, 0.5, 0.25])
    scale = 0.05

    expected = np.matmul(activations, ternary_weights.astype(np.float32) * scale)
    lut_output = NativeINT2LUTEngine.execute_lut_ternary_gemm(
        activations=activations,
        ternary_weights=ternary_weights,
        scale=scale,
    )

    np.testing.assert_allclose(lut_output, expected, rtol=1e-5, atol=1e-5)

    # Test 1D activation vector
    act_1d = activations[0]
    expected_1d = np.matmul(act_1d, ternary_weights.astype(np.float32) * scale)
    lut_output_1d = NativeINT2LUTEngine.execute_lut_ternary_gemm(
        activations=act_1d,
        ternary_weights=ternary_weights,
        scale=scale,
    )
    np.testing.assert_allclose(lut_output_1d, expected_1d, rtol=1e-5, atol=1e-5)


def test_lut_2bit_gemm_numerical_equivalence() -> None:
    np.random.seed(123)
    m_dim, k_dim, n_dim = 4, 64, 32
    activations = np.random.randn(m_dim, k_dim).astype(np.float32)
    uint8_weights = np.random.choice([0, 1, 2, 3], size=(k_dim, n_dim)).astype(np.uint8)

    scale = 0.02
    zero_point = -0.03

    # Ground truth dequantized weights
    w_dequant = uint8_weights.astype(np.float32) * scale + zero_point
    expected = np.matmul(activations, w_dequant)

    lut_output = NativeINT2LUTEngine.execute_lut_2bit_gemm(
        activations=activations,
        uint8_weights=uint8_weights,
        scale=scale,
        zero_point=zero_point,
    )

    np.testing.assert_allclose(lut_output, expected, rtol=1e-5, atol=1e-5)
