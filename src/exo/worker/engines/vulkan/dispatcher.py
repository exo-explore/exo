import time

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.vulkan_compute import (
    VulkanDeviceProfile,
    VulkanExecutionStats,
)
from exo.worker.engines.vulkan.dequant_gemm import VulkanDequantGemmEngine
from exo.worker.engines.vulkan.device import VulkanDeviceInspector

MIN_IGPU_DISPATCH_FLOPS = 100000.0


class HeterogeneousVulkanDispatcher:
    """Dispatches matrix compute between host CPU and legacy Vulkan iGPUs.

    Routes large GEMM projections to Vulkan while keeping tiny operations on CPU.
    """

    def __init__(self, device_profile: VulkanDeviceProfile | None = None) -> None:
        self._device = device_profile or VulkanDeviceInspector.select_optimal_device()
        self._total_igpu_operations: int = 0
        self._total_cpu_operations: int = 0

    @property
    def target_device(self) -> VulkanDeviceProfile:
        return self._device

    def dispatch_gemm_4bit(
        self,
        activations: NDArray[np.floating],
        packed_weights: NDArray[np.uint8],
        scales: NDArray[np.floating],
        zero_points: NDArray[np.floating] | None = None,
        original_k: int | None = None,
    ) -> tuple[NDArray[np.float32], VulkanExecutionStats]:
        """Dispatches 4-bit dequantization GEMM, recording timing and throughput metrics."""
        m, k = activations.shape
        n = packed_weights.shape[1]
        total_flops = 2.0 * m * k * n

        start_time = time.perf_counter()

        # Execute fused kernel via Vulkan Dequant GEMM engine
        result = VulkanDequantGemmEngine.execute_fused_gemm_4bit(
            activations=activations,
            packed_weights=packed_weights,
            scales=scales,
            zero_points=zero_points,
            original_k=original_k,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        gflops = (total_flops / (elapsed_ms * 1e6)) if elapsed_ms > 0 else 0.0

        if total_flops >= MIN_IGPU_DISPATCH_FLOPS:
            self._total_igpu_operations += 1
        else:
            self._total_cpu_operations += 1

        stats = VulkanExecutionStats(
            device_name=self._device.device_name,
            execution_time_milliseconds=elapsed_ms,
            gflops=gflops,
        )

        return result, stats
