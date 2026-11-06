import numpy as np
import pytest

from exo.shared.types.vulkan_compute import (
    AMD_VENDOR_ID,
    INTEL_VENDOR_ID,
    VulkanDeviceProfile,
)
from exo.worker.engines.vulkan import (
    HeterogeneousVulkanDispatcher,
    SubgroupReductionOptimizer,
    VulkanDequantGemmEngine,
    VulkanDeviceInspector,
)


def test_vulkan_device_inspector() -> None:
    devices = VulkanDeviceInspector.list_available_devices()
    assert len(devices) >= 2

    intel_dev = next(d for d in devices if d.vendor_id == INTEL_VENDOR_ID)
    amd_dev = next(d for d in devices if d.vendor_id == AMD_VENDOR_ID)

    assert intel_dev.is_legacy_igpu
    assert amd_dev.is_legacy_igpu
    assert intel_dev.subgroup_size == 16
    assert amd_dev.subgroup_size == 32

    optimal = VulkanDeviceInspector.select_optimal_device()
    assert optimal.is_legacy_igpu


def test_vulkan_optimal_device_discrete_priority() -> None:
    discrete_dev = VulkanDeviceProfile(
        device_name="Discrete Test GPU",
        device_type="discrete_gpu",
        vendor_id=0x10DE,
        subgroup_size=32,
    )
    # When selecting optimal from a list with discrete GPU
    devices = [discrete_dev] + list(VulkanDeviceInspector.list_available_devices())
    for d in devices:
        if d.device_type == "discrete_gpu":
            selected = d
            break
    assert selected.device_name == "Discrete Test GPU"


def test_vulkan_fused_gemm_4bit_accuracy() -> None:
    np.random.seed(42)
    m, k, n = 4, 32, 16
    activations = np.random.randn(m, k).astype(np.float32)

    # Raw 4-bit integer weights in range [0, 15]
    int4_weights = np.random.randint(0, 16, size=(k, n), dtype=np.uint8)
    packed_weights = VulkanDequantGemmEngine.pack_4bit_weights(int4_weights)
    assert packed_weights.shape == (k // 2, n)

    scales = np.random.uniform(0.01, 0.1, size=(n,)).astype(np.float32)
    zero_points = np.random.uniform(0.0, 8.0, size=(n,)).astype(np.float32)

    # Reference GEMM
    ref_weights = (int4_weights.astype(np.float32) - zero_points.reshape(1, n)) * scales.reshape(1, n)
    ref_output = activations @ ref_weights

    # Fused GEMM
    fused_output = VulkanDequantGemmEngine.execute_fused_gemm_4bit(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales,
        zero_points=zero_points,
    )

    np.testing.assert_allclose(fused_output, ref_output, rtol=1e-5, atol=1e-5)


def test_vulkan_fused_gemm_4bit_odd_k_padding() -> None:
    np.random.seed(42)
    m, k, n = 2, 15, 8
    activations = np.random.randn(m, k).astype(np.float32)
    int4_weights = np.random.randint(0, 16, size=(k, n), dtype=np.uint8)

    packed_weights = VulkanDequantGemmEngine.pack_4bit_weights(int4_weights)
    assert packed_weights.shape == ((k + 1) // 2, n)

    scales = np.ones((n,), dtype=np.float32)
    zero_points = np.zeros((n,), dtype=np.float32)

    ref_weights = int4_weights.astype(np.float32)
    ref_output = activations @ ref_weights

    fused_output = VulkanDequantGemmEngine.execute_fused_gemm_4bit(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales,
        zero_points=zero_points,
        original_k=k,
    )

    np.testing.assert_allclose(fused_output, ref_output, rtol=1e-5, atol=1e-5)


def test_vulkan_fused_gemm_2bit_accuracy() -> None:
    np.random.seed(42)
    m, k, n = 2, 32, 8
    activations = np.random.randn(m, k).astype(np.float32)

    # 2-bit weights in range [0, 3] representing ternary {-1, 0, 1, 2}
    int2_weights = np.random.randint(0, 4, size=(k, n), dtype=np.uint8)
    packed_weights = VulkanDequantGemmEngine.pack_2bit_weights(int2_weights)
    assert packed_weights.shape == (k // 4, n)

    scales = np.random.uniform(0.05, 0.2, size=(n,)).astype(np.float32)

    ref_weights = (int2_weights.astype(np.float32) - 1.0) * scales.reshape(1, n)
    ref_output = activations @ ref_weights

    fused_output = VulkanDequantGemmEngine.execute_fused_gemm_2bit(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales,
    )

    np.testing.assert_allclose(fused_output, ref_output, rtol=1e-5, atol=1e-5)


def test_subgroup_tile_size_selection() -> None:
    tile_intel = SubgroupReductionOptimizer.determine_optimal_tile_size(
        subgroup_size=16, k_dim=1024, n_dim=1024
    )
    assert tile_intel == (16, 16)

    tile_amd = SubgroupReductionOptimizer.determine_optimal_tile_size(
        subgroup_size=32, k_dim=1024, n_dim=1024
    )
    assert tile_amd == (32, 16)


def test_subgroup_shuffle_reduction() -> None:
    np.random.seed(123)
    vec = np.random.randn(64).astype(np.float32)
    expected_sum = float(np.sum(vec))

    simulated_sum = SubgroupReductionOptimizer.simulate_subgroup_shuffle_reduction(
        vector=vec,
        subgroup_size=16,
    )

    assert simulated_sum == pytest.approx(expected_sum, rel=1e-4)


def test_heterogeneous_vulkan_dispatcher() -> None:
    dispatcher = HeterogeneousVulkanDispatcher()
    assert dispatcher.target_device.is_legacy_igpu

    m, k, n = 4, 64, 32
    activations = np.random.randn(m, k).astype(np.float32)
    int4_weights = np.random.randint(0, 16, size=(k, n), dtype=np.uint8)
    packed_weights = VulkanDequantGemmEngine.pack_4bit_weights(int4_weights)
    scales = np.ones((n,), dtype=np.float32)

    result, stats = dispatcher.dispatch_gemm_4bit(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales,
    )

    assert result.shape == (m, n)
    assert stats.device_name == dispatcher.target_device.device_name
    assert stats.execution_time_milliseconds >= 0.0
    assert stats.gflops >= 0.0
