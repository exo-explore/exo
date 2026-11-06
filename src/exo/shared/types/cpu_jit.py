from dataclasses import dataclass, field
from typing import Literal

InstructionSetLevel = Literal[
    "AVX512",
    "AVX2",
    "AVX",
    "SSE4_2",
    "SSE4_1",
    "ARM_NEON",
    "SCALAR",
]


@dataclass(frozen=True)
class CPUCacheGeometry:
    """CPU cache hierarchy geometry in bytes."""

    l1d_cache_bytes: int = 32 * 1024  # 32 KB default L1 Data cache
    l2_cache_bytes: int = 256 * 1024  # 256 KB default L2 cache
    l3_cache_bytes: int = 8 * 1024 * 1024  # 8 MB default L3 cache
    cache_line_bytes: int = 64  # Standard 64-byte cache line

    def validate(self) -> bool:
        return (
            self.l1d_cache_bytes > 0
            and self.l2_cache_bytes >= self.l1d_cache_bytes
            and self.l3_cache_bytes >= self.l2_cache_bytes
            and self.cache_line_bytes > 0
        )


@dataclass(frozen=True)
class CPUCapabilityInfo:
    """Detailed hardware CPU capability and supported SIMD vector sets."""

    architecture: str
    vendor: str
    active_level: InstructionSetLevel
    supported_instruction_sets: list[str] = field(default_factory=list)
    has_avx512: bool = False
    has_avx2: bool = False
    has_avx: bool = False
    has_fma: bool = False
    has_sse4_2: bool = False
    has_sse4_1: bool = False
    has_neon: bool = False
    physical_cores: int = 1
    logical_cores: int = 1


@dataclass(frozen=True)
class CacheTilingPlan:
    """Cache tiling block sizes tailored to L1/L2 cache capacities."""

    tile_m: int
    tile_n: int
    tile_k: int
    l1_working_set_bytes: int
    l2_working_set_bytes: int
    estimated_cache_hit_rate: float
    vector_width_elements: int = 4

    @property
    def total_tile_elements(self) -> int:
        return self.tile_m * self.tile_n * self.tile_k


@dataclass(frozen=True)
class JITExecutionReport:
    """Telemetry report produced by vectorized CPU JIT kernel runs."""

    kernel_name: str
    instruction_level: InstructionSetLevel
    tiling_plan: CacheTilingPlan
    m: int
    k: int
    n: int
    elapsed_ms: float
    gflops: float
    is_fallback: bool = False
