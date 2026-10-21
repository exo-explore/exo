from __future__ import annotations

import platform
import sys
from enum import Enum

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel


class NpuArchitecture(str, Enum):
    AppleANE = "AppleANE"
    IntelNPU = "IntelNPU"
    QualcommHexagon = "QualcommHexagon"
    AmdRyzenAI = "AmdRyzenAI"
    GenericDirectML = "GenericDirectML"
    SimulatedNpu = "SimulatedNpu"


class NpuDeviceProfile(FrozenModel):
    """
    Profile representing dedicated Neural Processing Unit hardware capabilities.
    """

    architecture: NpuArchitecture
    tera_operations_per_second: float = Field(ge=0.0)
    memory_bandwidth_gigabytes: float = Field(ge=0.0)
    thermal_efficiency_factor: float = Field(default=3.0, ge=1.0)
    supports_float16: bool = True
    supports_int8: bool = True
    supports_dynamic_shapes: bool = False

    def estimate_energy_joules(
        self,
        token_count: int,
        parameter_count_billions: float,
    ) -> float:
        """
        Estimate electrical energy consumption in Joules for a given token count.
        NPU efficiency reduces energy consumption relative to GPU baselines.
        """
        # Baseline rough energy for GPU: ~0.05 Joules per token per billion parameters
        baseline_joules = 0.05 * token_count * parameter_count_billions
        return baseline_joules / self.thermal_efficiency_factor

    @classmethod
    def detect_local_device(cls) -> NpuDeviceProfile:
        """
        Probe and build the local NPU device profile.
        """
        if sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
            return cls(
                architecture=NpuArchitecture.AppleANE,
                tera_operations_per_second=15.8,
                memory_bandwidth_gigabytes=100.0,
                thermal_efficiency_factor=4.5,
                supports_float16=True,
                supports_int8=True,
                supports_dynamic_shapes=False,
            )

        # Non-Apple or default fallback profile
        return cls(
            architecture=NpuArchitecture.SimulatedNpu,
            tera_operations_per_second=10.0,
            memory_bandwidth_gigabytes=50.0,
            thermal_efficiency_factor=2.5,
            supports_float16=True,
            supports_int8=True,
            supports_dynamic_shapes=True,
        )
