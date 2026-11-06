from typing import Literal

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

VulkanDeviceType = Literal[
    "integrated_gpu",
    "discrete_gpu",
    "cpu",
    "virtual_gpu",
    "other",
]

QuantizationBitWidth = Literal[2, 4, 8]

# Standard Intel/AMD vendor IDs for legacy iGPU detection
INTEL_VENDOR_ID = 0x8086
AMD_VENDOR_ID = 0x1002


class VulkanDeviceProfile(FrozenModel):
    """Device profile and feature capabilities for Vulkan compute execution."""

    device_name: str
    device_type: VulkanDeviceType
    vendor_id: int
    api_version: str = "1.2.0"
    subgroup_size: int = 16
    supports_fp16: bool = True
    supports_subgroup_shuffle: bool = True
    dedicated_memory_mb: int = 0
    shared_memory_mb: int = 4096

    @property
    def is_legacy_igpu(self) -> bool:
        """Determines if the device is an integrated GPU from Intel or AMD."""
        return (
            self.device_type == "integrated_gpu"
            and self.vendor_id in (INTEL_VENDOR_ID, AMD_VENDOR_ID)
        )


class VulkanGemmConfig(FrozenModel):
    """Configuration specification for fused dequantization matrix multiplication on Vulkan."""

    m: int
    k: int
    n: int
    bit_width: QuantizationBitWidth = 4
    tile_size_m: int = 16
    tile_size_n: int = 16


class VulkanExecutionStats(FrozenModel):
    """Performance telemetry from Vulkan compute shader execution."""

    device_name: str
    execution_time_milliseconds: float
    gflops: float
    memory_bandwidth_gigabytes_per_second: float = Field(default=0.0)
