from exo.utils.pydantic_ext import FrozenModel

DEFAULT_SRAM_BYTES: int = 32 * 1024 * 1024  # 32 MB default SRAM budget
DEFAULT_SRAM_TOKENS: int = 256


class SRAMConfig(FrozenModel):
    """Configuration for on-chip NPU SRAM KV cache residency and INT2 LUT execution."""

    max_sram_bytes: int = DEFAULT_SRAM_BYTES
    max_sram_tokens: int = DEFAULT_SRAM_TOKENS
    enable_lut_acceleration: bool = True


class SRAMAllocation(FrozenModel):
    """Allocation status of on-chip SRAM vs external DRAM spillover for a layer."""

    layer_idx: int
    tokens_in_sram: int
    tokens_in_dram: int
    sram_bytes_used: int
    is_sram_saturated: bool


class SRAMTelemetry(FrozenModel):
    """Telemetry report capturing SRAM hit ratio, DRAM I/O reduction, and energy savings."""

    total_attention_tokens: int
    sram_hit_ratio: float
    dram_io_reduction_ratio: float
    estimated_energy_savings_ratio: float
    lut_operations_count: int
